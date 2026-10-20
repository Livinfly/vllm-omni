"""CPU-only per-model reports; optional history is another suite's summary.json."""

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    FIELDS,
    measurements_complete,
    read_json,
    review_measurements,
    summarize,
    write_json,
)
from lifecycle import completed_groups


def collect(suite, plan) -> list[dict[str, Any]]:
    completed = completed_groups(suite)
    rows = []
    for group in plan["groups"]:
        directory = suite / group["directory"]
        metrics = read_json(directory / "metrics.json") if (directory / "metrics.json").exists() else {}
        effective = (
            read_json(directory / "effective_config.json") if (directory / "effective_config.json").exists() else {}
        )
        for mode in plan["modes"]:
            records = [r for r in metrics.get("records", []) if r["mode"] == mode]
            formal = sorted((r for r in records if not r["warmup"]), key=lambda r: r["repeat"])
            warmup = [r for r in records if r["warmup"]]
            rows.append(
                dict(
                    group,
                    mode=mode,
                    count=len(formal),
                    effective=effective,
                    complete=group["directory"] in completed and measurements_complete(metrics, plan, mode),
                    warmup=summarize(warmup),
                    **summarize(formal),
                )
            )
    for row in rows:
        base = next(
            (
                r
                for r in rows
                if r["model"] == row["model"] and r["mode"] == row["mode"] and r["gpus"] == 1 and r["complete"]
            ),
            None,
        )
        for field, key in [(FIELDS[0], "e2e_speedup"), (FIELDS[2], "dit_speedup")]:
            if base and row["complete"] and field in base and field in row:
                row[key] = base[field]["mean"] / row[field]["mean"]
    return rows


def fmt(row, field) -> str:
    if field not in row:
        return "—"
    stat = row[field]
    if stat["stdev"] is None:
        return f"{stat['mean']:.3f}（单次）"
    return f"{stat['mean']:.3f} ± {stat['stdev']:.3f}"


def execution_policy(value):
    if isinstance(value, dict):
        return {k: execution_policy(v) for k, v in value.items() if k not in {"master_port", "nccl_port"}}
    if isinstance(value, list):
        return [execution_policy(v) for v in value]
    return value


def history_mismatches(plan, row, history, old) -> list[str]:
    reasons = []
    previous = history["plan"]
    for key in ["timing_schema_version", "source_commit", "warmup"]:
        if plan[key] != previous.get(key):
            reasons.append(key)
    if plan["inference_source_modified"] or previous.get("inference_source_modified", True):
        reasons.append("inference source modified")
    previous_model = previous.get("models", {}).get(row["model"], {})
    if plan["models"][row["model"]]["profile"] != previous_model.get("profile"):
        reasons.append("model/profile")
    manifest = plan.get("model_manifest_sha256", {}).get(row["model"])
    if not manifest or manifest != previous.get("model_manifest_sha256", {}).get(row["model"]):
        reasons.append("model manifest")
    current, prior = row["effective"], old.get("effective", {})
    if not current or not prior:
        reasons.append("effective config missing")
    else:
        for key in [
            "packages",
            "cuda",
            "pipeline_config",
            "resolved_execution",
            "attention_implementations",
            "native_vae_manifest",
        ]:
            if execution_policy(current.get(key)) != execution_policy(prior.get(key)):
                reasons.append(key)
        if sorted(d["name"] for d in current["devices"]) != sorted(d["name"] for d in prior["devices"]):
            reasons.append("GPU model")
    return reasons


def generate_report(suite, baseline=None) -> None:
    suite = Path(suite)
    plan, status = read_json(suite / "plan.json"), read_json(suite / "status.json")
    rows = collect(suite, plan)
    lines = [
        "# vLLM-Omni 性能报告",
        "",
        f"状态：**{status['status']}**；完成组 {len(status['completed'])}/{status['total']}。",
        "",
        "E2E：客户端调用 Omni.generate 前至原生 mux_video_audio_bytes 封装和 MP4 写盘返回，"
        "包含文本编码、噪声、DiT、scheduler、通信、视频/音频 VAE、编码封装和保存。"
        "加载、排队、warmup、检查及指标聚合排除。",
        "",
        "DiT：完整 transformer forward，pre-hook 位于原生 root pre-hooks 前，"
        "post-hook 位于原生 root post-hooks 后；覆盖 SP/FSDP 与 VSA-H3，不含 scheduler。"
        "主指标 max(sum) 为各 rank 先累计再取最大，另列 sum(max)。",
        "",
        "dit = E2E + DiT；e2e 无 DiT hooks/events，纯 e2e 不分配 events。事件在计时外预分配和初始化；"
        "两模式均不添加逐步或 pipeline 结束同步，关闭原生阶段 profiler。"
        "distributed E2E：同一客户端起点至保存后的控制 RPC 到达各 rank 并完成 Gloo barrier；"
        "包含额外 RPC 调度开销，不能与其他框架的主线程 barrier 指标直接等同。逐 rank 原始值全部保留。",
        "",
        "## 配置",
        "",
        f"源码：`{plan['source_branch']}` / `{plan['source_commit']}`；"
        f"推理源码改动：{plan['inference_source_modified']}。",
        f"每模式 warmup={plan['warmup']}，正式={plan['repeats']}；额外文件 hash 验证：{plan['validate_video']}。",
        "",
        "有效后端、块大小、topk、fallback 策略逐 rank 保存，"
        "配置差异见下文。模型配置 hash / 权重大小见 [model_manifest.json](model_manifest.json)，"
        "未逐字节核验权重。",
        "",
    ]
    for name, item in plan["models"].items():
        profile = item["profile"]
        if profile["adapter"] == "fastwan21_14b":
            from fastwan import COMPARABILITY, MULTICARD

            lines += ["**FastWan 支持范围**：" + MULTICARD, "", "**计算差异**：" + COMPARABILITY, ""]
        lines += [
            f"### {name}",
            "",
            f"模型路径：`{item['path']}`。",
            "",
        ]
        if profile["engine"]:
            lines += [
                "**非默认执行配置**：`" + json.dumps(profile["engine"], ensure_ascii=False) + "`。",
                "",
                "原因：" + profile.get("execution_note", "未提供，请核对本组启动配置。"),
                "",
            ]
        lines += [
            "```json",
            json.dumps(profile, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    for row in rows:
        effective = row["effective"]
        if not effective or row["mode"] != plan["modes"][0]:
            continue
        devices = sorted({f"{d['name']} / {d['total_memory_gib']:.2f} GiB" for d in effective["devices"]})
        lines += [
            f"{row['model']} g{row['gpus']}：{', '.join(devices)}；DiT forwards={effective['expected_dit_forwards']}。",
            f"Python {effective['python'].split()[0]} / CUDA {effective['cuda']}；包 `{effective['packages']}`。",
            f"实际执行策略：`{effective['resolved_execution']}`。",
            f"完整有效配置：[{row['directory']}/effective_config.json]({row['directory']}/effective_config.json)。",
            "",
        ]
    warnings = sorted({w for row in rows for w in row["effective"].get("warnings", [])})
    if warnings:
        lines += ["## 原生配置差异", "", *["- " + w for w in warnings], ""]
    lines += [
        "## 耗时与多卡加速比",
        "",
        "单位秒。加速比仅使用相同模型、相同模式、已完成的单卡结果；单次无法估计波动，stdev=null。",
        "",
        "| 模型 | GPU | 模式 | 次数 / 状态 | E2E | distributed E2E | DiT max(sum) | DiT sum(max) | "
        "E2E 加速比 | DiT 加速比 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [fmt(row, field) for field in FIELDS]
        values += [f"{row[k]:.3f}×" if k in row else "—" for k in ["e2e_speedup", "dit_speedup"]]
        lines.append(
            f"| {row['model']} | {row['gpus']} | {row['mode']} | {row['count']} / "
            + ("完成" if row["complete"] else "待完成")
            + " | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "## Warmup 与双模式对照",
        "",
        "| 模型 | GPU | 模式 | warmup E2E / DiT | 正式 E2E / DiT |",
        "|---|---:|---|---|---|",
    ]
    comparisons = []
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['gpus']} | {row['mode']} | "
            f"{fmt(row['warmup'], FIELDS[0])} / {fmt(row['warmup'], FIELDS[2])} | "
            f"{fmt(row, FIELDS[0])} / {fmt(row, FIELDS[2])} |"
        )
        other = next(
            (
                r
                for r in rows
                if r["model"] == row["model"] and r["gpus"] == row["gpus"] and r["mode"] == "e2e" and r["complete"]
            ),
            None,
        )
        if row["mode"] == "dit" and row["complete"] and other:
            comparisons.append(
                dict(
                    model=row["model"],
                    gpus=row["gpus"],
                    mean_difference_pct=100 * (row[FIELDS[0]]["mean"] / other[FIELDS[0]]["mean"] - 1),
                    paired_difference_pct=[
                        100 * (a / b - 1)
                        for a, b in zip(
                            row[FIELDS[0]]["values"],
                            other[FIELDS[0]]["values"],
                            strict=False,
                        )
                    ],
                )
            )
    if comparisons:
        lines += [
            "",
            f"双模式 E2E 差异 (dit/e2e − 1)：`{comparisons}`；仅描述本次观测，不宣称统计显著性。",
        ]
    if baseline is not None:
        history = read_json(baseline)
        lines += [
            "",
            "## 历史对比",
            "",
            "按需比较同格式 summary.json。旧 exp baseline 的额外同步、阶段日志和策略不同，不能直接当作同口径结果。",
            "",
            "| 模型 | GPU | 模式 | E2E 差异 | DiT 差异 | 配置可比性 |",
            "|---|---:|---|---:|---:|---|",
        ]
        for row in rows:
            old = next(
                (r for r in history["results"] if all(r[k] == row[k] for k in ["model", "gpus", "mode"])),
                None,
            )
            if old is None:
                continue
            reasons = history_mismatches(plan, row, history, old)
            row["historical_mismatches"] = reasons
            delta = [
                f"{100 * (row[f]['mean'] / old[f]['mean'] - 1):+.3f}%"
                if not reasons and row["complete"] and old["complete"] and f in row and f in old
                else "—"
                for f in [FIELDS[0], FIELDS[2]]
            ]
            lines.append(
                f"| {row['model']} | {row['gpus']} | {row['mode']} | {' | '.join(delta)} | "
                + (", ".join(reasons) or "配置匹配；权重内容未完整核验")
                + " |"
            )
    findings = review_measurements(rows)
    lines += [
        "",
        "## 复查与状态",
        "",
        "多次正式测量 CV > 2%，或至少三次连续变快且首尾下降 > 2% 时提示；不自动增加请求次数。",
        "",
    ]
    lines += [f"- {finding}" for finding in findings] or ["- 暂无波动提示。"]
    lines += [
        "",
        f"组状态：`{status['completed']}`。",
        "",
        "原始逐 rank 数据见 `<模型>/gN/metrics.json`；失败摘要与日志路径见 [status.json](status.json)。",
        "",
    ]
    from sparse_attention import report_lines
    lines += report_lines(rows)
    (suite / "report.md").write_text("\n".join(lines))
    write_json(
        suite / "summary.json",
        dict(
            status=status["status"],
            plan=plan,
            results=rows,
            timing_comparisons=comparisons,
            review=findings,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    generate_report(args.suite, baseline=args.baseline)
