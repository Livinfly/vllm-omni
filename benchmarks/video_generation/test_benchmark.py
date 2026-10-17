"""Host-only regressions for native V2 admission, timing, RPC and matrix lifecycle."""

import copy
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run as launcher
from common import HERE, SCHEMA, aggregate, read_json, schedule, write_json
from config import METADATA, V1_METADATA, model_manifest, validate_metadata, validate_profile
from extension import attention_info
from report import collect, generate_report, history_mismatches
from runtime import engine_kwargs, launch_environment, native_vae_manifest, rpc, sampling_kwargs
from timing import DiTEvents


def rank_row(rank, mode="dit", seconds=10):
    return dict(
        rank=rank,
        response_e2e_s=seconds if rank == 0 else None,
        distributed_e2e_s=seconds + 1,
        dit_step_ms=[1] * 8 if mode == "dit" else None,
        peak_allocated_gib=1,
        peak_reserved_gib=2,
    )


def records():
    return [
        aggregate([rank_row(0)], mode="dit", tag="test", warmup=warmup, repeat=repeat)
        for mode, warmup, repeat in schedule(["dit"], 1, 1)
    ]


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.profile = read_json(HERE / "configs/fasth3_v2.json")

    def test_v2_fixed_sampling_and_native_defaults(self):
        validate_profile(self.profile)
        kwargs = engine_kwargs(self.profile, "/model", 8)
        self.assertEqual(kwargs["parallel_config"]["ulysses_degree"], 8)
        self.assertEqual(kwargs["parallel_config"]["tensor_parallel_size"], 1)
        for key in ["enforce_eager", "enable_cpu_offload", "use_hsdp", "enable_prompt_embed_cache"]:
            self.assertNotIn(key, kwargs)
        self.assertFalse(kwargs["enable_diffusion_pipeline_profiler"])
        self.assertTrue(kwargs["trust_remote_code"])
        self.assertEqual(sampling_kwargs(self.profile)["num_inference_steps"], 8)
        for mutate in [
            lambda p: p["sampling"].update(num_inference_steps=9),
            lambda p: p.update(model_id="V1"),
            lambda p: p["engine"].update(worker_extension_cls="other"),
            lambda p: p["attention"].update(checkpoint_sparsity=0.9),
        ]:
            p = copy.deepcopy(self.profile)
            mutate(p)
            with self.assertRaises(ValueError):
                validate_profile(p)

    def test_v1_uses_four_calls_and_version_specific_attention(self):
        profile = read_json(HERE / "configs/fasth3_v1_offload.json")
        validate_profile(profile)
        validate_metadata(V1_METADATA)
        self.assertEqual(sampling_kwargs(profile)["num_inference_steps"], 4)
        self.assertEqual(sampling_kwargs(profile)["extra_args"]["flow_shift"], 12.0)
        self.assertEqual(profile["contract"]["dmd_denoising_steps"], [999, 749, 500, 250])
        self.assertEqual(profile["attention"]["checkpoint_sparsity"], 0.9)
        for mutate in [
            lambda p: p["sampling"].update(num_inference_steps=5),
            lambda p: p["contract"].update(video_shift=10.0),
            lambda p: p["attention"].update(checkpoint_sparsity=0.8),
        ]:
            bad = copy.deepcopy(profile)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_profile(bad)
        impl = type("MiniMaxH3VSAImpl", (), {})()
        impl.provider, impl.precision, impl.topk, impl.fallback_on_error = "fastvideo", "bf16", 64, False
        impl.resolve_h3_topk = lambda blocks, sparsity: 66
        layer = SimpleNamespace(attn_backend=SimpleNamespace(get_name=lambda: "FASTVIDEO_VSA"), attention=impl)
        transformer = SimpleNamespace(
            blocks=[SimpleNamespace(attn=SimpleNamespace(to_gate_compress=object(), attention=layer, vsa_sparsity=0.9))]
        )
        resolved = attention_info(transformer, [768, 1344, 124, 37, 207], 0.9)
        self.assertEqual(resolved["sparsity_expected_topk"], 66)
        self.assertEqual(resolved["warnings"], [])

    def test_native_snapshot_requires_provenance_and_all_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / ".cache" / "snapshot"
            root.mkdir(parents=True)
            for leaf, data in [
                ("model_index.json", {}),
                ("fastvideo_inference.json", METADATA),
                ("transformer/config.json", {"num_attention_heads": 56}),
                ("text_encoder/config.json", {}),
                ("tokenizer/tokenizer_config.json", {}),
                ("processor/preprocessor_config.json", {}),
            ]:
                path = root / leaf
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, data)
            for component in ["transformer", "text_encoder"]:
                (root / component / "model.safetensors").write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "provenance"):
                model_manifest(root, self.profile, [1, 8])
            write_json(root / "provenance.json", {"base_model": "hf://MiniMaxAI/MiniMax-H3@" + "a" * 40})
            original = root / "transformer"
            target = Path(folder) / "linked_transformer"
            original.rename(target)
            original.symlink_to(target, target_is_directory=True)
            manifest = model_manifest(root, self.profile, [1, 8])
            self.assertEqual(manifest["weight_file_sizes"]["transformer/model.safetensors"], 7)
            self.assertIn("transformer/config.json", manifest["configs"])
            v1 = read_json(HERE / "configs/fasth3_v1.json")
            with self.assertRaisesRegex(ValueError, "model_id does not match"):
                model_manifest(root, v1, [1])
            write_json(root / "fastvideo_inference.json", V1_METADATA)
            for component, shift in [("scheduler", 12.0), ("audio_scheduler", 3.0)]:
                (root / component).mkdir()
                write_json(root / component / "scheduler_config.json", {"shift": shift})
            model_manifest(root, v1, [1, 8])
            write_json(root / "scheduler/scheduler_config.json", {"shift": 10.0})
            with self.assertRaisesRegex(ValueError, "shift must be 12"):
                model_manifest(root, v1, [1])
            write_json(root / "fastvideo_inference.json", METADATA)
            with self.assertRaisesRegex(ValueError, "attention heads"):
                model_manifest(root, self.profile, [3])
            write_json(root / "transformer/model.safetensors.index.json", {"weight_map": {"x": "missing.safetensors"}})
            with self.assertRaisesRegex(ValueError, "Missing shard"):
                model_manifest(root, self.profile, [1])

    def test_offload_profile_preserves_workload_and_explains_override(self):
        profile = read_json(HERE / "configs/fasth3_v2_offload.json")
        validate_profile(profile)
        self.assertEqual(
            {k: v for k, v in profile.items() if k not in {"name", "engine", "execution_note"}},
            {k: v for k, v in self.profile.items() if k not in {"name", "engine"}},
        )
        kwargs = engine_kwargs(profile, "/model", 8)
        self.assertEqual(kwargs["diffusion_offload_config"], {"mode": "layer", "components": ["dit", "text_encoder"]})
        self.assertNotIn("execution_note", kwargs)
        self.assertIn("OOM", profile["execution_note"])

    def test_attention_records_sparsity_topk_and_backend_default_separately(self):
        impl = type("MiniMaxH3VSAImpl", (), {})()
        impl.provider, impl.precision, impl.topk, impl.fallback_on_error = "fastvideo", "bf16", 64, True
        impl.resolve_h3_topk = lambda blocks, sparsity: 132
        layer = SimpleNamespace(attn_backend=SimpleNamespace(get_name=lambda: "FASTVIDEO_VSA"), attention=impl)
        transformer = SimpleNamespace(
            blocks=[SimpleNamespace(attn=SimpleNamespace(to_gate_compress=object(), attention=layer, vsa_sparsity=0.8))]
        )
        info = attention_info(transformer, [768, 1344, 124, 37, 207])
        self.assertEqual(info["video_blocks"], 660)
        self.assertEqual(info["sparsity_expected_topk"], 132)
        self.assertTrue(info["warnings"])
        self.assertEqual(info["attention_implementations"][0]["topk"], 132)
        self.assertEqual(info["attention_implementations"][0]["backend_default_topk"], 64)
        self.assertFalse(any("differs" in warning for warning in info["warnings"]))
        self.assertEqual(impl.topk, 64)

    def test_partial_offline_vae_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / ".cache" / "FL2VA"
            root.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "VAE config"):
                native_vae_manifest(root)
            for name in ["video_vae", "audio_vae"]:
                component = root / name
                component.mkdir()
                write_json(component / "config.json", {})
            with self.assertRaisesRegex(ValueError, "VAE weights"):
                native_vae_manifest(root)
            for name in ["video_vae", "audio_vae"]:
                (root / name / "model.safetensors").write_bytes(b"fixture")
                target = Path(folder) / name
                (root / name).rename(target)
                (root / name).symlink_to(target, target_is_directory=True)
            # A fully cached base snapshot can also contain unrelated DiT/TE files.
            (root / "transformer").mkdir()
            (root / "transformer/model.safetensors").write_bytes(b"unrelated")
            self.assertEqual(len(native_vae_manifest(root)["weight_file_sizes"]), 2)
            self.assertIn("video_vae/config.json", native_vae_manifest(root)["config_and_code_hashes"])
            write_json(
                root / "audio_vae/config.json", dict(source_path="source", source_safetensors_path="absent.safetensors")
            )
            with self.assertRaisesRegex(ValueError, "VAE source weights"):
                native_vae_manifest(root)

    def test_history_rejects_changed_assets_and_resolved_policies(self):
        plan = dict(
            timing_schema_version=SCHEMA,
            source_commit="commit",
            warmup=1,
            inference_source_modified=False,
            models={"fasth3_v2": dict(profile=self.profile)},
            model_manifest_sha256={"fasth3_v2": "manifest"},
        )
        row = dict(
            model="fasth3_v2",
            effective=dict(
                devices=[dict(name="GPU")],
                resolved_execution=[dict(enable_cpu_offload=False, master_port=123)],
                native_vae_manifest=dict(config_and_code_hashes={"video_vae/config.json": "hash"}),
            ),
        )
        history, old = dict(plan=copy.deepcopy(plan)), copy.deepcopy(row)
        old["effective"]["resolved_execution"][0]["master_port"] = 456
        self.assertEqual(history_mismatches(plan, row, history, old), [])
        old["effective"]["resolved_execution"][0]["enable_cpu_offload"] = True
        self.assertIn("resolved_execution", history_mismatches(plan, row, history, old))
        history["plan"]["model_manifest_sha256"]["fasth3_v2"] = "other"
        self.assertIn("model manifest", history_mismatches(plan, row, history, old))

    def test_rpc_validates_all_rank_results_and_native_failure(self):
        engine = SimpleNamespace(
            engine=SimpleNamespace(collective_rpc=Mock(return_value=[[[{"rank": 1}, {"rank": 0}]]]))
        )
        self.assertEqual([r["rank"] for r in rpc(engine, "prepare", 2)], [0, 1])
        engine.engine.collective_rpc.return_value = [{"supported": False, "error": "worker failed"}]
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            rpc(engine, "prepare", 2)
        engine.engine.collective_rpc.return_value = [[{"rank": 0}, {"rank": 0}]]
        with self.assertRaisesRegex(RuntimeError, "Missing or duplicate"):
            rpc(engine, "prepare", 2)

    def test_environment_disables_historical_plugin_and_uses_snapshot(self):
        with patch.dict(os.environ, {}, clear=True):
            env = launch_environment(dict(PATH="/bin", FASTH3_OMNI_BASELINE="1", FASTVIDEO_VSA_TK="1"), HERE)
        self.assertNotIn("FASTH3_OMNI_BASELINE", env)
        self.assertNotIn("FASTVIDEO_VSA_TK", env)
        self.assertEqual(env["VLLM_PLUGINS"], "")
        self.assertEqual(env["FASTVIDEO_VSA_TRITON"], "1")
        self.assertTrue(env["PYTHONPATH"].startswith(str(HERE)))

    def test_relocated_dry_run_needs_no_model_or_gpu(self):
        with tempfile.TemporaryDirectory(prefix="omni space ") as folder:
            destination = Path(folder) / "clone/benchmarks/video_generation"
            shutil.copytree(HERE, destination, ignore=shutil.ignore_patterns("__pycache__"))
            result = subprocess.run(
                [
                    sys.executable,
                    str(destination / "run.py"),
                    "--gpus",
                    "8",
                    "1",
                    "--model-path",
                    "fasth3_v2=relative model",
                    "--dry-run",
                ],
                cwd=folder,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("--plan"), 2)
            self.assertFalse((destination.parents[1] / "outputs").exists())

    def test_compat_library_environment_is_idempotent_for_preview_reexec(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "libcuda.so.1"):
                launch_environment({"BENCH_CUDA_COMPAT": folder}, HERE)
            (root / "libcuda.so.1").touch()
            env = launch_environment({"BENCH_CUDA_COMPAT": folder, "LD_LIBRARY_PATH": "/other"}, HERE)
            self.assertEqual(env["LD_LIBRARY_PATH"], folder + ":/other")
            self.assertEqual(launch_environment(env, HERE), env)

    def test_full_root_event_order_and_e2e_detach(self):
        import torch

        trace = []

        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self):
                trace.append("event")

            def elapsed_time(self, other):
                return 1.0

        class Model(torch.nn.Module):
            def forward(self, x):
                trace.append("forward")
                return x

        model = Model()
        model.register_forward_pre_hook(lambda *a: trace.append("pre"))
        model.register_forward_hook(lambda *a: trace.append("post"))
        timer = DiTEvents(
            SimpleNamespace(
                cuda=SimpleNamespace(Event=Event), accelerator=SimpleNamespace(synchronize=lambda: trace.append("sync"))
            ),
            1,
        )
        self.assertEqual(trace.count("sync"), 1)
        timer.attach(model)
        trace.clear()
        model(1)
        self.assertEqual(trace, ["event", "pre", "forward", "post", "event"])
        self.assertEqual(timer.elapsed_ms(), [1.0])
        timer.detach()
        trace.clear()
        model(1)
        self.assertEqual(trace, ["pre", "forward", "post"])

    def test_aggregation_rank_imbalance_and_forward_count(self):
        a, b = rank_row(0), rank_row(1)
        a["dit_step_ms"], b["dit_step_ms"] = [1, 3], [3, 1]
        row = aggregate([a, b], mode="dit", tag="test", warmup=False, repeat=1)
        self.assertEqual(row["dit_forward_sum_max_rank_s"], 0.004)
        self.assertEqual(row["dit_forward_step_max_sum_s"], 0.006)
        b["dit_step_ms"] = [1]
        with self.assertRaises(ValueError):
            aggregate([a, b], mode="dit", tag="test", warmup=False, repeat=1)

    def test_report_single_sample_and_schema_separation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            groups = [dict(model="fasth3_v2", gpus=n, directory=f"fasth3_v2/g{n}") for n in [1, 2]]
            plan = dict(
                groups=groups,
                modes=["dit"],
                warmup=1,
                repeats=1,
                models={"fasth3_v2": dict(path="/model", profile=read_json(HERE / "configs/fasth3_v2_offload.json"))},
                source_branch="exp-fixed-version",
                source_commit="test",
                inference_source_modified=False,
                validate_video=False,
                timing_schema_version=SCHEMA,
            )
            write_json(root / "plan.json", plan)
            write_json(
                root / "status.json",
                dict(status="complete", total=2, completed=[dict(g, status="complete") for g in groups]),
            )
            for group in groups:
                d = root / group["directory"]
                d.mkdir(parents=True)
                write_json(d / "metrics.json", dict(status="complete", records=records()))
            generate_report(root)
            self.assertIn("单次", (root / "report.md").read_text())
            self.assertIn("RPC", (root / "report.md").read_text())
            self.assertIn("非默认执行配置", (root / "report.md").read_text())
            self.assertIn("94.90 GiB", (root / "report.md").read_text())
            rows = collect(root, plan)
            self.assertEqual(rows[1]["e2e_speedup"], 1)
            self.assertIsNone(rows[0]["response_e2e_s"]["stdev"])


class LauncherTests(unittest.TestCase):
    def exercise(self, failure=None, phase="load"):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            suite = Path(folder) / "suite"
            stack.enter_context(
                patch.object(
                    sys,
                    "argv",
                    ["run.py", "--gpus", "2", "1", "--output", str(suite), "--model-path", "fasth3_v2=" + folder],
                )
            )
            stack.enter_context(patch.object(launcher, "model_manifest", return_value={}))
            stack.enter_context(
                patch.object(
                    launcher, "inventory", return_value=[dict(index=str(i), uuid=f"GPU-{i}") for i in range(2)]
                )
            )
            stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,0"}))
            stack.enter_context(
                patch.object(launcher.subprocess, "check_output", side_effect=["exp-fixed-version\n", "commit\n", ""])
            )
            stack.enter_context(patch.object(launcher, "wait_for_suite"))
            stack.enter_context(patch.object(launcher.os, "killpg"))
            signals = stack.enter_context(patch.object(launcher.signal, "signal"))
            handles, launched = [], []

            def wait(selected, notify):
                self.assertEqual(selected, ["GPU-1", "GPU-0"][: len(selected)])
                handle = (Path(folder) / f"lock{len(handles)}").open("w")
                handles.append(handle)
                return [handle]

            def popen(command, **kwargs):
                self.assertFalse(handles[-1].closed)
                d = Path(command[command.index("--output") + 1])
                launched.append(d)
                if failure == "interrupt" and len(launched) == 1:

                    def interrupted_wait(**kwargs):
                        signals.call_args.args[1](launcher.signal.SIGTERM, None)

                    return SimpleNamespace(pid=123456789, wait=interrupted_wait)
                if failure and len(launched) == 1:
                    write_json(
                        d / "error_rank0.json", dict(status=failure, phase=phase, traceback="RuntimeError: test")
                    )
                    return SimpleNamespace(pid=123456789, wait=lambda **kwargs: 1)
                write_json(d / "metrics.json", dict(status="complete", records=records()))
                return SimpleNamespace(pid=123456789, wait=lambda **kwargs: 0)

            stack.enter_context(patch.object(launcher, "wait_for_idle", side_effect=wait))
            stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=popen))
            with redirect_stdout(io.StringIO()):
                if failure == "interrupt":
                    with self.assertRaises(KeyboardInterrupt):
                        launcher.main()
                elif failure == "memory_insufficient" and phase in {"load", "warmup"}:
                    with self.assertRaises(SystemExit):
                        launcher.main()
                elif failure:
                    with self.assertRaisesRegex(RuntimeError, "Group failed"):
                        launcher.main()
                else:
                    launcher.main()
            self.assertTrue(all(h.closed for h in handles))
            self.assertEqual(read_json(suite / "status.json")["has_failures"], bool(failure))
            stop = failure and not (failure == "memory_insufficient" and phase in {"load", "warmup"})
            self.assertEqual(len(launched), 1 if stop else 2)
            self.assertTrue((suite / "scripts/configs/fasth3_v2.json").exists())

    def test_sigterm_updates_status_and_releases_locks(self):
        self.exercise("interrupt")

    def test_success(self):
        self.exercise()

    def test_failure_stops(self):
        self.exercise("failed")

    def test_load_oom_continues(self):
        self.exercise("memory_insufficient")

    def test_measurement_oom_stops(self):
        self.exercise("memory_insufficient", "repeat")


if __name__ == "__main__":
    unittest.main()
