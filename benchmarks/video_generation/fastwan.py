"""Native Wan DMD benchmark with our local tile64/pure-Ulysses adaptation."""

import json
import math
import struct
from dataclasses import asdict
from pathlib import Path

from common import read_json, sha256
from prepare_fastwan import MODEL_ID, REVISION

ADAPTER = "fastwan21_14b"
# Local adaptation: opt into our tile64 route to match FastVideo/SGLang.
TILE = (4, 4, 4)
MULTICARD = (
    "使用我们自行适配的 strict Ulysses VSA，显式开启 fastvideo_vsa_allow_ulysses；"
    "仅支持 G1/G2/G4/G8、TP/PP/CFG/Ring/AllGather=1、无 SP padding；self-attention 禁止 dense 回退。"
)
COMPARABILITY = (
    "使用我们自行适配的 Wan tile64 路由，显式设置 (4,4,4)，覆盖 Omni 原生 (4,8,8) 默认值，"
    "以对齐 FastVideo/SGLang 的分块与 topk；offload、缓存等执行策略仍须结合实际配置比较。"
)


def geometry(sampling):
    grid = [(sampling["num_frames"] - 1) // 4 + 1, sampling["height"] // 16, sampling["width"] // 16]
    blocks = math.prod(math.ceil(n / tile) for n, tile in zip(grid, TILE))
    return grid, blocks, max(1, math.ceil((1 - 0.9) * blocks))


def validate_world(profile, world):
    if type(world) is not int or world not in (1, 2, 4, 8):
        raise ValueError(MULTICARD)
    if math.prod(geometry(profile["sampling"])[0]) % world or 40 % world:
        raise ValueError("Local FastWan Ulysses requires sequence and heads divisible by GPU count")


def validate_profile(profile):
    if profile["model_id"] != MODEL_ID or profile["adapter"] != ADAPTER:
        raise ValueError("Require the selected FastWan 2.1 14B checkpoint")
    if (
        profile["engine"].get("engine_backend", "fastwan_engine.FastWanBenchmarkEngine")
        != "fastwan_engine.FastWanBenchmarkEngine"
    ):
        raise ValueError("Only our reviewed FastWan benchmark engine override is supported")
    s = profile["sampling"]
    if set(s) != {"height", "width", "num_frames", "fps", "seed", "guidance_scale", "num_inference_steps"}:
        raise ValueError("Unexpected FastWan sampling keys")
    if any(type(s[k]) is not int or s[k] < 1 for k in ("height", "width", "num_frames", "fps")):
        raise ValueError("Require positive video dimensions")
    if s["height"] % 16 or s["width"] % 16 or s["num_frames"] % 4 != 1:
        raise ValueError("Wan dimensions must preserve VAE/patch shape")
    if s["guidance_scale"] != 1 or s["num_inference_steps"] != 3:
        raise ValueError("Require distilled CFG1 / three DiT forwards")
    if type(s["seed"]) is not int or not 0 <= s["seed"] < 2**32:
        raise ValueError("Invalid seed")
    if profile["contract"] != dict(dmd_denoising_steps=[1000, 757, 522], flow_shift=8.0):
        raise ValueError("Require native FastWan DMD schedule")
    if profile["attention"] != dict(
        backend="FASTVIDEO_VSA",
        provider="fastvideo",
        precision="bf16",
        checkpoint_sparsity=0.9,
        tile_size=64,
        topk=geometry(s)[2],
        allow_ulysses=True,
    ):
        raise ValueError("Require locally adapted Wan tile64 and sparsity-derived fixed topk")
    if not isinstance(profile["prompt"], str) or not profile["prompt"].strip() or profile["negative_prompt"] != "":
        raise ValueError("Require a nonempty prompt; CFG1 does not use a negative prompt")


def tensor_headers(path):
    tensors = {}
    for shard in sorted((path / "transformer").glob("*.safetensors")):
        with shard.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 64 * 1024**2:
                raise ValueError("Invalid safetensors header size")
            header = json.loads(stream.read(size))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate transformer tensor: {name}")
            if entry["data_offsets"][1] + size + 8 > shard.stat().st_size:
                raise ValueError(f"Truncated transformer shard: {shard}")
            tensors[name] = entry
    return tensors


def model_manifest(path, profile, gpu_counts):
    validate_profile(profile)
    for world in gpu_counts:
        validate_world(profile, world)
    path = Path(path)
    verification = read_json(path / "download_verification.json")
    if (verification.get("model_id"), verification.get("revision"), verification.get("status")) != (
        MODEL_ID,
        REVISION,
        "complete",
    ):
        raise ValueError("Require a complete, verified pinned FastWan snapshot")
    configs, weights = {}, {}
    for entry in verification["files"]:
        file = path / entry["file"]
        if not file.is_file() or file.stat().st_size != entry["size"]:
            raise ValueError(f"Missing/truncated pinned asset: {file}")
        if file.suffix == ".json":
            configs[entry["file"]] = dict(sha256=sha256(file), config=read_json(file))
            if configs[entry["file"]]["sha256"] != entry["sha256"]:
                raise ValueError(f"Modified pinned config: {file}")
        elif file.suffix == ".safetensors":
            weights[entry["file"]] = file.stat().st_size
    if read_json(path / "model_index.json")["_class_name"] != "WanDMDPipeline":
        raise ValueError("Expected WanDMDPipeline")
    config = read_json(path / "transformer/config.json")
    if (config["num_layers"], config["num_attention_heads"], config["attention_head_dim"]) != (40, 40, 128):
        raise ValueError("Unexpected 14B transformer geometry")
    tensors = tensor_headers(path)
    gates = {name: entry for name, entry in tensors.items() if ".to_gate_compress." in name}
    expected = {f"blocks.{i}.to_gate_compress.{kind}" for i in range(40) for kind in ("weight", "bias")}
    if set(gates) != expected:
        raise ValueError("Incomplete learned VSA gates")
    for name, entry in gates.items():
        if entry["dtype"] != "BF16" or entry["shape"] != ([5120, 5120] if name.endswith("weight") else [5120]):
            raise ValueError(f"Unexpected gate: {name}")
    return dict(
        model_id=MODEL_ID,
        revision=REVISION,
        configs=configs,
        weight_file_sizes=weights,
        verified_download=verification,
        transformer_tensors=len(tensors),
        trained_gate_tensors=len(gates),
        multicard_support=MULTICARD,
        comparability=COMPARABILITY,
    )


def engine_kwargs(profile, path, world):
    validate_world(profile, world)
    kwargs = dict(
        model=str(path),
        model_class_name="WanDMDPipeline",
        dtype="bfloat16",
        num_gpus=world,
        parallel_config=dict(
            tensor_parallel_size=1,
            ulysses_degree=world,
            ring_degree=1,
            allgather_degree=1,
            cfg_parallel_size=1,
            ulysses_mode="strict",
        ),
        # Opt in only self-attention; dense cross-attention keeps platform defaults.
        diffusion_attention_config={
            "per_role": {
                "self": {
                    "backend": "FASTVIDEO_VSA",
                    "fastvideo_vsa_topk": profile["attention"]["topk"],
                    "fastvideo_vsa_block_size": TILE,
                    "fastvideo_vsa_allow_ulysses": True,
                }
            }
        },
        worker_extension_cls="extension.TimingExtension",
        enable_diffusion_pipeline_profiler=False,
        init_timeout=3600,
        stage_init_timeout=3600,
        **profile["engine"],
    )
    # Our Wan harness adaptation: this registered pipeline consumes stage-scoped
    # overrides. In particular a global nested parallel_config can be discarded.
    kwargs.update(
        {
            f"stage_0_{key}": value
            for key, value in list(kwargs.items())
            if key not in {"model", "init_timeout", "stage_init_timeout"}
        }
    )
    return kwargs


def sampling_kwargs(profile):
    return dict(**profile["sampling"], output_type="np", extra_args=dict(sample_solver="euler", flow_shift=8.0))


def resolve_config(profile, model_path, world):
    from runtime import serializable

    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.config.omni_config import extract_diffusion_stage_config_kwargs
    from vllm_omni.diffusion.attention.backends.fastvideo_vsa import validate_local_ulysses_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict_from_omni_stage_config
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    kwargs = engine_kwargs(profile, model_path, world)
    # Validate the same registry/stage projection used by Omni, rather than
    # checking terminal kwargs that may never reach the actual worker.
    stages = StageConfigFactory.create_from_model(
        str(model_path), trust_remote_code=False, cli_overrides=kwargs, deploy_config_path=None
    )
    if stages is None or len(stages.stage_configs) != 1:
        raise ValueError("Expected the registered single-stage Wan pipeline")
    projected = build_engine_args_dict_from_omni_stage_config(stages.stage_configs[0], str(model_path))
    config = OmniDiffusionConfig.from_kwargs(
        **extract_diffusion_stage_config_kwargs(projected, stage_id=0, include_engine_adapter_metadata=True)
    )
    if config.num_gpus != world or validate_local_ulysses_config(config.parallel_config) != world:
        raise ValueError("Actual Omni stage topology differs from the requested GPU count")
    if config.engine_backend != profile["engine"].get("engine_backend", "default"):
        raise ValueError("Actual Omni stage lost the explicitly selected benchmark engine")
    spec = config.diffusion_attention_config.per_role.get("self")
    if (
        spec is None
        or not spec.fastvideo_vsa_allow_ulysses
        or spec.fastvideo_vsa_topk != geometry(profile["sampling"])[2]
    ):
        raise ValueError("Actual Omni stage lost the requested VSA configuration")
    sample = OmniDiffusionSamplingParams(**sampling_kwargs(profile))
    s = profile["sampling"]
    return dict(
        engine_kwargs=kwargs,
        pipeline_config=serializable(asdict(config)),
        sampling=serializable(asdict(sample)),
        native_shape=[s["height"], s["width"], s["num_frames"]],
        expected_dit_forwards=3,
        video_grid=geometry(s)[0],
        video_blocks=geometry(s)[1],
        topk=geometry(s)[2],
        multicard_support=MULTICARD,
        comparability=COMPARABILITY,
    )


def prepare_worker(worker, modes, native_shape):
    import torch
    from runtime import serializable
    from timing import DiTEvents

    pipe = worker.model_runner.pipeline
    if not pipe.is_dmd or pipe.transformer_2 is not None or pipe._flow_shift != 8.0:
        raise ValueError("Expected single-transformer native FastWan DMD pipeline / shift8")
    from vllm_omni.diffusion.attention.backends.fastvideo_vsa import validate_local_ulysses_config

    degree = validate_local_ulysses_config(worker.od_config.parallel_config)
    if worker.od_config.num_gpus != degree or worker.od_config.enable_diffusion_pipeline_profiler:
        raise ValueError("FastWan benchmark requires pure Ulysses without synchronized stage profiling")
    if not pipe.has_gate_compress_weights:
        raise ValueError("Native loader did not find learned VSA gates")
    transformer = pipe.transformer
    routes = []
    s = dict(zip(("height", "width", "num_frames"), native_shape))
    validate_world({"sampling": s}, degree)
    grid, blocks, topk = geometry(s)
    for block in transformer.blocks:
        attn = block.attn1
        if attn.to_gate_compress is None:
            raise ValueError("Missing trained Wan VSA gate")
        impl = attn.attn.attention
        if type(impl).__name__ != "FastVideoVSAImpl" or impl.topk != topk or impl.block_size != TILE:
            raise ValueError("Unexpected native Wan VSA implementation / topk / tile")
        if not impl.allow_ulysses or impl.fallback_on_error:
            raise ValueError("Require our explicit Ulysses adaptation with dense fallback disabled")
        if not torch.isfinite(attn.to_gate_compress.weight).all() or not torch.count_nonzero(
            attn.to_gate_compress.weight
        ):
            raise ValueError("Unloaded/nonfinite learned gate")
        routes.append(
            dict(
                backend=attn.attn.attn_backend.get_name(),
                implementation=type(impl).__name__,
                provider=impl.provider,
                precision=impl.precision,
                topk=impl.topk,
                block_size=list(impl.block_size),
                video_blocks=blocks,
                fallback_on_error=impl.fallback_on_error,
                allow_ulysses=impl.allow_ulysses,
                local_heads=40 // degree,
                cross_backend=block.attn2.attn.attn_backend.get_name(),
                gate_nonzero_verified=True,
            )
        )
    if len(routes) != 40:
        raise ValueError("Expected 40 Wan transformer blocks")
    worker._bench_transformer = transformer
    worker._bench_timer = DiTEvents(torch, 3) if "dit" in modes else None
    props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    components = {}
    for name in ("transformer", "text_encoder", "vae"):
        module = getattr(pipe, name)
        components[name] = dict(
            devices=sorted({str(p.device) for p in module.parameters()}),
            dtypes=sorted({str(p.dtype) for p in module.parameters()}),
        )
    return dict(
        rank=worker.rank,
        name=props.name,
        uuid=str(props.uuid),
        total_memory_gib=props.total_memory / 2**30,
        capability=[props.major, props.minor],
        pipeline_class=type(pipe).__name__,
        pipeline_config=serializable(asdict(worker.od_config)),
        components=components,
        attention_implementations=routes,
        video_grid=grid,
        video_blocks=blocks,
        warnings=[MULTICARD, COMPARABILITY],
    )
