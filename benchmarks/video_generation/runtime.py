"""Native API configuration and media I/O; no inference replacements."""

import json
import os
from dataclasses import asdict
from pathlib import Path

from common import ROOT, read_json, sha256


def serializable(value):
    return json.loads(json.dumps(value, default=lambda v: getattr(v, "__qualname__", str(v))))


def launch_environment(env, scripts):
    # The historical experimental plugin must remain inert in this native suite.
    env = {k: v for k, v in env.items() if not k.startswith(("FASTH3_", "FASTVIDEO_", "VLLM_OMNI_"))}
    env.update(
        VLLM_OMNI_BENCH_SOURCE_ROOT=str(ROOT),
        PYTHONPATH=os.pathsep.join([str(scripts), str(ROOT)]),
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_PLUGINS="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        FASTVIDEO_VSA_TRITON="1",
    )
    compat = env.get("BENCH_CUDA_COMPAT")
    if compat:
        compat = str(Path(compat).expanduser().resolve())
        if not (Path(compat) / "libcuda.so.1").is_file():
            raise ValueError("BENCH_CUDA_COMPAT must contain libcuda.so.1")
        existing = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p and p != compat]
        env["LD_LIBRARY_PATH"] = os.pathsep.join([compat, *existing])
    return env


def engine_kwargs(profile, model_path, world):
    if profile["adapter"] == "fastwan21_14b":
        from fastwan import engine_kwargs as wan_kwargs

        return wan_kwargs(profile, model_path, world)
    return dict(
        model=str(model_path),
        model_class_name="MiniMaxH3ModularPipeline",
        task_type="fl2va",
        trust_remote_code=True,  # Native pinned H3 VAEs load their checkpoint modeling code.
        dtype="bfloat16",
        num_gpus=world,
        parallel_config=dict(
            tensor_parallel_size=1, ulysses_degree=world, ring_degree=1, allgather_degree=1, cfg_parallel_size=1
        ),
        diffusion_attention_config={
            "default": {
                "backend": "FASTVIDEO_VSA",
                "fastvideo_vsa_provider": "fastvideo",
                "fastvideo_vsa_precision": "bf16",
            },
            "per_role": {"minimax_h3.token_refiner": {"backend": "TORCH_SDPA"}},
        },
        worker_extension_cls="extension.TimingExtension",
        enable_diffusion_pipeline_profiler=False,
        init_timeout=3600,
        stage_init_timeout=3600,
        **profile["engine"],
    )


def sampling_kwargs(profile):
    if profile["adapter"] == "fastwan21_14b":
        from fastwan import sampling_kwargs as wan_sampling

        return wan_sampling(profile)
    s = profile["sampling"]
    return dict(
        **s,
        quality="lossless",
        extra_args=dict(
            task="t2va",
            duration=s["num_frames"] / s["fps"],
            aspect_ratio="16:9",
            flow_shift=profile["contract"]["video_shift"],
            audio_flow_shift=profile["contract"]["audio_shift"],
            sampler="euler",
        ),
    )


def native_vae_manifest(root):
    """Offline snapshot resolution alone does not guarantee cached file coverage."""
    root = Path(root)
    configs, weights = {}, {}
    for name in ["video_vae", "audio_vae"]:
        component = root / name
        config = component / "config.json"
        if not config.is_file():
            raise ValueError(f"Missing native VAE config: {config}")
        data = read_json(config)
        if data.get("source_safetensors_path"):
            source = component / data.get("source_path", ".") / data["source_safetensors_path"]
            if not source.is_file():
                raise ValueError(f"Missing native VAE source weights: {source}")
        if not any(p.is_file() and p.suffix in {".safetensors", ".bin", ".pt", ".pth"} for p in component.rglob("*")):
            raise ValueError(f"Missing native VAE weights: {component}")
        for index in component.rglob("*.safetensors.index.json"):
            for shard in set(read_json(index)["weight_map"].values()):
                if not (index.parent / shard).is_file():
                    raise ValueError(f"Missing native VAE shard: {index.parent / shard}")
        for item in component.rglob("*"):
            if item.is_file() and ".cache" not in item.relative_to(root).parts:
                leaf = str(item.relative_to(root))
                if item.suffix in {".json", ".py", ".yaml", ".yml", ".txt"}:
                    configs[leaf] = sha256(item)
                elif item.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
                    weights[leaf] = item.stat().st_size
    return dict(config_and_code_hashes=configs, weight_file_sizes=weights)


def resolve_config(profile, model_path, world):
    if profile["adapter"] == "fastwan21_14b":
        from fastwan import resolve_config as resolve_wan

        return resolve_wan(profile, model_path, world)
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import FastH3CheckpointSpec
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import resolve_minimax_h3_shape

    kwargs = engine_kwargs(profile, model_path, world)
    config = OmniDiffusionConfig.from_kwargs(
        **{k: v for k, v in kwargs.items() if k not in {"init_timeout", "stage_init_timeout"}}
    )
    spec = FastH3CheckpointSpec.from_metadata(read_json(Path(model_path) / "fastvideo_inference.json"))
    spec.check_serving_contract(partition="fl2va", od_config=config)
    sample = OmniDiffusionSamplingParams(**sampling_kwargs(profile))
    spec.check_request(sample)
    shape = resolve_minimax_h3_shape("t2va", sample, None)
    s = profile["sampling"]
    if tuple(shape[:3]) != (s["height"], s["width"], s["num_frames"]):
        raise ValueError(f"Native resolver changes workload shape: {shape}")
    # Offline-only: fail before any model load if the pinned VAE snapshot is absent.
    vae_path = spec.resolve_native_vaes(Path(model_path))
    return dict(
        engine_kwargs=kwargs,
        pipeline_config=serializable(asdict(config)),
        sampling=serializable(asdict(sample)),
        native_shape=list(shape),
        native_vae_path=str(vae_path),
        native_vae_manifest=native_vae_manifest(vae_path),
        expected_dit_forwards=spec.num_inference_steps,
    )


def rpc(engine, method, world, *args):
    result = engine.engine.collective_rpc(method=method, args=args, timeout=7200, stage_ids=[0])

    def flatten(value):
        if isinstance(value, list):
            return [row for part in value for row in flatten(part)]
        if isinstance(value, dict) and "rank" in value:
            return [value]
        raise RuntimeError(f"Unexpected {method} response: {value}")

    rows = sorted(flatten(result), key=lambda row: row["rank"])
    if [row["rank"] for row in rows] != list(range(world)):
        raise RuntimeError(f"Missing or duplicate worker ranks in {method}")
    return rows


def extract_media(result, require_audio=True):
    import numpy as np
    import torch

    if not isinstance(result, list) or len(result) != 1:
        raise ValueError("Expected one Omni output")
    output = result[0]
    mm = output.multimodal_output or {}
    frames, audio, rate = output.images, mm.get("audio"), mm.get("audio_sample_rate", 32000)
    if isinstance(frames, list) and len(frames) == 1:
        frames = frames[0]
    if isinstance(frames, tuple) and len(frames) == 2:
        frames, audio = frames
    if isinstance(frames, dict):
        audio, rate = frames.get("audio", audio), frames.get("audio_sample_rate", rate)
        frames = frames.get("frames") if "frames" in frames else frames.get("video")
    if isinstance(frames, list) and len(frames) == 1:
        frames = frames[0]
    if isinstance(frames, (bytes, bytearray)):
        return bytes(frames), None, rate
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    if audio is not None:
        audio = np.asarray(audio)
        if audio.ndim == 3 and audio.shape[0] == 1:
            audio = audio[0]
    if not require_audio and np.issubdtype(frames.dtype, np.floating):
        if not np.isfinite(frames).all():
            raise ValueError("Nonfinite video output")
        # Native Wan postprocessing emits [0,1] THWC float frames.
        if frames.min() < -1e-4 or frames.max() > 1.0001:
            raise ValueError("Unexpected native Wan output range")
        frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or (require_audio and audio is None):
        raise ValueError("Expected uint8 video [T,H,W,3] and generated audio")
    return frames, audio, rate


def video_metadata(path, require_audio=True):
    import av
    import numpy as np

    with av.open(str(path)) as media:
        v = media.streams.video[0]
        result = dict(
            width=v.width, height=v.height, fps=float(v.average_rate), frames=sum(1 for _ in media.decode(video=0))
        )
    with av.open(str(path)) as media:
        if not media.streams.audio:
            if not require_audio:
                return dict(result, audio_samples=0)
            raise ValueError("Missing audio stream")
        samples = 0
        for frame in media.decode(audio=0):
            if not np.isfinite(frame.to_ndarray()).all():
                raise ValueError("Nonfinite audio")
            samples += frame.samples
        if not samples:
            raise ValueError("Empty audio")
    return dict(result, audio_samples=samples)
