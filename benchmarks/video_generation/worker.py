"""One Omni client; native workers own multi-GPU execution. Media save is timed."""

import argparse
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

from common import ROOT, SCHEMA, aggregate, read_json, schedule, sha256, write_json
from runtime import engine_kwargs, extract_media, resolve_config, rpc, sampling_kwargs, video_metadata


def package_versions():
    """Source checkouts may import successfully without distribution metadata."""
    versions = {}
    for name in ["torch", "vllm", "vllm-omni", "transformers", "diffusers", "fastvideo-kernel", "triton", "av"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    item = plan["models"][args.model]
    profile = item["profile"]
    is_wan = profile["adapter"] == "fastwan21_14b"
    os.environ.update(
        VLLM_OMNI_BENCH_OUTPUT=str(args.output),
        VLLM_OMNI_BENCH_SELECTED_UUIDS=json.dumps(plan["selected_gpu_uuids"][: args.gpus]),
    )
    engine, phase, records = None, "load", []

    def dump(name, value):
        write_json(args.output / name, value)

    try:
        import vllm_omni

        if Path(vllm_omni.__file__).resolve().parent != ROOT / "vllm_omni":
            raise ValueError("Wrong vllm_omni checkout imported")
        import torch

        from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        dump("progress.json", dict(status="preflight"))
        effective = resolve_config(profile, item["path"], args.gpus)
        packages = package_versions()
        effective.update(
            packages=packages,
            package_metadata_missing=[name for name, version in packages.items() if version is None],
        )
        dump(
            "effective_config.json",
            dict(
                effective,
                devices=[],
                python=sys.version,
                cuda=torch.version.cuda,
                resolved_execution=effective["pipeline_config"],
            ),
        )
        dump("progress.json", dict(status="loading"))
        begin = time.perf_counter()
        engine = Omni(**engine_kwargs(profile, item["path"], args.gpus))
        devices = rpc(engine, "benchmark_prepare", args.gpus, plan["modes"], effective["native_shape"])
        effective.update(
            devices=devices,
            python=sys.version,
            cuda=torch.version.cuda,
            timing_schema_version=SCHEMA,
            load_seconds_excluded=time.perf_counter() - begin,
            resolved_execution=[row["pipeline_config"] for row in devices],
            attention_implementations=[row["attention_implementations"] for row in devices],
            warnings=sorted({w for row in devices for w in row["warnings"]}),
            environment={
                k: v
                for k, v in os.environ.items()
                if k.startswith(("VLLM_", "FASTVIDEO_", "NCCL_"))
                or k in {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LD_LIBRARY_PATH", "OMP_NUM_THREADS"}
            },
        )
        dump("effective_config.json", effective)
        for warning in effective["warnings"]:
            print("CONFIG WARNING: " + warning, flush=True)
        for mode, warmup, repeat in schedule(plan["modes"], plan["warmup"], plan["repeats"]):
            phase = "warmup" if warmup else "repeat"
            tag = f"{mode}_{phase}_{repeat}"
            output = args.output / f"{tag}.mp4"
            sample = OmniDiffusionSamplingParams(**sampling_kwargs(profile))
            prompt = dict(prompt=profile["prompt"], negative_prompt=profile["negative_prompt"])
            dump("progress.json", dict(status="running", tag=tag))
            rpc(engine, "benchmark_arm", args.gpus, mode, phase, profile["sampling"]["seed"])
            # E2E START: native generate, response transport, media conversion, native mux and file write.
            started = time.perf_counter()
            result = engine.generate(prompt, sampling_params_list=[sample], use_tqdm=False)
            api_only_s = time.perf_counter() - started
            frames, audio, rate = extract_media(result, require_audio=False) if is_wan else extract_media(result)
            encoded = (
                frames
                if isinstance(frames, bytes)
                else mux_video_audio_bytes(frames, audio, fps=profile["sampling"]["fps"], audio_sample_rate=rate)
            )
            output.write_bytes(encoded)
            response_s = time.perf_counter() - started  # E2E END: saved MP4 returned.
            # No added pipeline-end/step CUDA sync, no timing RPC inside response E2E.
            per_rank = rpc(engine, "benchmark_collect", args.gpus, started, response_s)
            record = aggregate(per_rank, mode=mode, tag=tag, warmup=warmup, repeat=repeat)
            media = video_metadata(output, require_audio=False) if is_wan else video_metadata(output)
            if any(
                media[key] != profile["sampling"][field]
                for key, field in [("width", "width"), ("height", "height"), ("frames", "num_frames"), ("fps", "fps")]
            ):
                raise ValueError(f"Unexpected saved media: {media}")
            record.update(api_only_s=api_only_s, video_file=output.name, video_metadata=media)
            if plan["validate_video"]:
                record["video_sha256"] = sha256(output)
            records.append(record)
            dump("metrics.json", dict(status="running", timing_schema_version=SCHEMA, records=records))
            print(f"{tag}: E2E={response_s:.3f}s, DiT={record.get('dit_forward_sum_max_rank_s')}", flush=True)
            del result, frames, audio, encoded
        dump("metrics.json", dict(status="complete", timing_schema_version=SCHEMA, records=records))
        dump("progress.json", dict(status="complete"))
    except BaseException:
        dump("error_client.json", dict(status="failed", phase=phase, traceback=traceback.format_exc()))
        raise
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()
