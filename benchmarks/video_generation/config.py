"""CPU admission for the native FastH3 V2 full-checkpoint path."""

import re
from pathlib import Path

from common import read_json, sha256

MODEL_ID = "FastVideo/FastVideo-FastH3-8-Step-V2"
LADDER = [999, 874, 749, 624, 500, 375, 250, 125]
METADATA = dict(
    model_id=MODEL_ID,
    schema_version="fasth3-inference-contract-v1",
    dmd_denoising_steps=LADDER,
    transformer_forwards=8,
    video_scheduler_shift=10.0,
    audio_scheduler_shift=3.0,
    guidance_scale=1.0,
    attention_backend="VIDEO_SPARSE_ATTN_H3",
    vsa_sparsity=0.8,
    vsa_tile_size=64,
    task="t2av",
)
V1_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
V1_METADATA = dict(
    METADATA,
    model_id=V1_MODEL_ID,
    dmd_denoising_steps=[999, 749, 500, 250],
    transformer_forwards=4,
    vsa_sparsity=0.9,
    num_inference_steps=5,
)
V1_METADATA.pop("video_scheduler_shift")
V1_METADATA.pop("audio_scheduler_shift")


def contract_metadata(model_id):
    if model_id == MODEL_ID:
        return METADATA
    if model_id == V1_MODEL_ID:
        return V1_METADATA
    raise ValueError(f"Unsupported model_id: {model_id}")


# Topology and observation are owned by the harness. Other policies are explicit overrides.
ENGINE_KEYS = {
    "enable_cpu_offload",
    "enable_layerwise_offload",
    "enable_distributed_layerwise_offload",
    "pin_cpu_memory",
    "enforce_eager",
    "vae_use_tiling",
    "vae_use_slicing",
    "cache_backend",
    "cache_config",
    "enable_prompt_embed_cache",
    "diffusion_offload_config",
    "diffusion_compile_granularity",
    "diffusion_compile_dynamic",
}


def validate_metadata(data):
    for key, expected in contract_metadata(data.get("model_id")).items():
        if data.get(key) != expected:
            raise ValueError(f"Unsupported checkpoint metadata: {key}={data.get(key)!r}; expected {expected!r}")


def validate_profile(p):
    if set(p) - {"execution_note"} != {
        "name",
        "adapter",
        "model_id",
        "sampling",
        "prompt",
        "negative_prompt",
        "contract",
        "engine",
        "attention",
    }:
        raise ValueError("Use the keys in configs/fasth3_v2.json")
    if "execution_note" in p and (not isinstance(p["execution_note"], str) or not p["execution_note"].strip()):
        raise ValueError("execution_note must explain the nondefault execution policy")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", p["name"]):
        raise ValueError("Unsafe profile name")
    if p["adapter"] == "fastwan21_14b":
        from fastwan import validate_profile as validate_wan

        validate_wan(p)
        if set(p["engine"]) - (ENGINE_KEYS | {"engine_backend"}):
            raise ValueError("Unsupported engine overrides")
        return
    metadata = contract_metadata(p["model_id"])
    version = "v1" if p["model_id"] == V1_MODEL_ID else "v2"
    shift = 12.0 if version == "v1" else 10.0
    if p["adapter"] != "minimax_h3_" + version:
        raise ValueError("Adapter must match checkpoint version")
    s = p["sampling"]
    if set(s) != {"height", "width", "num_frames", "fps", "seed", "guidance_scale", "num_inference_steps"}:
        raise ValueError("Use the sampling keys in configs/fasth3_v2.json")
    for key in ["height", "width", "num_frames", "fps", "num_inference_steps"]:
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if s["num_inference_steps"] != metadata["transformer_forwards"] or s["guidance_scale"] != 1:
        raise ValueError("Require checkpoint transformer forwards and CFG=1")
    if type(s["seed"]) is not int or not 0 <= s["seed"] < 2**32:
        raise ValueError("Invalid seed")
    if p["contract"] != dict(dmd_denoising_steps=metadata["dmd_denoising_steps"], video_shift=shift, audio_shift=3.0):
        raise ValueError("Checkpoint schedule/shifts are fixed")
    if p["attention"] != dict(
        backend="FASTVIDEO_VSA",
        provider="fastvideo",
        precision="bf16",
        checkpoint_sparsity=metadata["vsa_sparsity"],
        tile_size=64,
    ):
        raise ValueError("Use native checkpoint VSA-H3 BF16/tile64; fixed topk overrides are unsupported")
    if set(p["engine"]) - ENGINE_KEYS:
        raise ValueError("Unsupported engine overrides: " + str(set(p["engine"]) - ENGINE_KEYS))
    if not isinstance(p["prompt"], str) or not p["prompt"].strip() or p["negative_prompt"] != "":
        raise ValueError("Require nonempty prompt and empty negative prompt for distilled T2VA")


def model_manifest(path, profile, gpu_counts):
    validate_profile(profile)
    if profile["adapter"] == "fastwan21_14b":
        from fastwan import model_manifest as wan_manifest

        return wan_manifest(path, profile, gpu_counts)
    path = Path(path)
    required = [
        "model_index.json",
        "fastvideo_inference.json",
        "provenance.json",
        "transformer/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "processor/preprocessor_config.json",
    ]
    for leaf in required:
        if not (path / leaf).is_file():
            raise ValueError(f"Missing native model asset: {path / leaf}; see README.md")
    metadata = read_json(path / "fastvideo_inference.json")
    validate_metadata(metadata)
    if metadata["model_id"] != profile["model_id"]:
        raise ValueError("Checkpoint model_id does not match profile")
    if metadata["model_id"] == V1_MODEL_ID:
        for component, expected in [("scheduler", 12.0), ("audio_scheduler", 3.0)]:
            if read_json(path / component / "scheduler_config.json").get("shift") != expected:
                raise ValueError(f"V1 {component} shift must be {expected}")
    base = read_json(path / "provenance.json").get("base_model", "")
    if not re.fullmatch(r"hf://MiniMaxAI/MiniMax-H3@[0-9a-fA-F]{40}", base):
        raise ValueError("provenance.json must pin the MiniMaxAI/MiniMax-H3 base commit")
    heads = read_json(path / "transformer/config.json")["num_attention_heads"]
    if any(heads % n for n in gpu_counts):
        raise ValueError(f"GPU counts must divide {heads} attention heads")
    for component in ["transformer", "text_encoder"]:
        shards = list((path / component).glob("*.safetensors"))
        if not shards:
            raise ValueError(f"Missing {component} weights")
        for index in (path / component).glob("*.safetensors.index.json"):
            for shard in set(read_json(index)["weight_map"].values()):
                if not (index.parent / shard).is_file():
                    raise ValueError(f"Missing shard: {index.parent / shard}")
    configs, weights = {}, {}
    # Start traversal at each component so linked checkpoint directories are
    # included too (Path.rglob does not recurse through directory symlinks).
    items = (
        item
        for root in path.iterdir()
        if root.name != ".cache"
        for item in (root.rglob("*") if root.is_dir() else [root])
    )
    for item in items:
        if item.is_file() and ".cache" not in item.relative_to(path).parts:
            name = str(item.relative_to(path))
            if item.suffix == ".json":
                configs[name] = dict(sha256=sha256(item), config=read_json(item))
            elif item.suffix in {".safetensors", ".bin", ".pt"}:
                weights[name] = item.stat().st_size
    return dict(
        configs=configs,
        weight_file_sizes=weights,
        native_vae_base=base,
        note="Config hashes / weight sizes only. Native VAEs are resolved from the pinned HF cache in preview/worker.",
    )
