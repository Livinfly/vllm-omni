"""Opt-in native worker extension. Only observation and request-boundary control."""

import json
import math
import os
import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from common import write_json
from gpu_wait import check_device_identity
from runtime import serializable
from timing import DiTEvents


def attention_info(transformer, native_shape, expected_sparsity=0.8):
    # These are the native H3 root blocks, not its dense token refiner.
    blocks = list(transformer.blocks)
    if not blocks or any(block.attn.to_gate_compress is None for block in blocks):
        raise ValueError("Missing trained VSA gates")
    height, width, _, latent_t, _ = native_shape
    video_grid = [latent_t, height // 32, width // 32]
    video_blocks = math.prod(math.ceil(size / 4) for size in video_grid)
    routes = []
    for block in blocks:
        layer = block.attn.attention
        backend = layer.attn_backend.get_name()
        impl = layer.attention
        if backend != "FASTVIDEO_VSA" or type(impl).__name__ != "MiniMaxH3VSAImpl":
            raise ValueError(f"Unexpected DiT attention: {backend}/{type(impl).__name__}")
        routes.append(
            dict(
                backend=backend,
                implementation=type(impl).__name__,
                provider=impl.provider,
                precision=impl.precision,
                topk=impl.resolve_h3_topk(video_blocks, block.attn.vsa_sparsity),
                backend_default_topk=impl.topk,
                sparsity=block.attn.vsa_sparsity,
                topk_policy="checkpoint_sparsity",
                fallback_on_error=impl.fallback_on_error,
            )
        )
    # Native shape is H,W,frames,latent_t,audio_t; spatial VSA layout is patchified by 2.
    expected = math.ceil((1 - expected_sparsity) * video_blocks)
    warnings = []
    if any(row["topk"] != expected for row in routes):
        warnings.append(
            f"Native VSA topk differs from checkpoint sparsity={expected_sparsity}: "
            f"video_blocks={video_blocks}, expected topk={expected}, "
            f"configured={sorted({r['topk'] for r in routes})}. "
            "Not a matched FastVideo attention workload."
        )
    if any(row["fallback_on_error"] for row in routes):
        warnings.append(
            "Native VSA permits dense fallback. Inspect inference.log; "
            "configured backend is not proof of every executed route."
        )
    return dict(
        attention_implementations=routes,
        video_grid=video_grid,
        video_blocks=video_blocks,
        sparsity_expected_topk=expected,
        warnings=warnings,
    )


class TimingExtension:
    def __init__(self, local_rank, rank, od_config, *args, **kwargs):
        # Executed by Omni's supported mixin before native initialization/model load.
        self._bench_timer = None
        self._bench_phase = "load"
        self._bench_output = Path(os.environ["VLLM_OMNI_BENCH_OUTPUT"])
        selected = json.loads(os.environ["VLLM_OMNI_BENCH_SELECTED_UUIDS"])
        try:
            import torch

            check_device_identity(torch, local_rank, selected[rank])
            for key, leaf in [("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")]:
                os.environ[key] = str(self._bench_output / "cache" / f"rank{rank}" / leaf)
            super().__init__(local_rank, rank, od_config, *args, **kwargs)
        except BaseException as error:
            self._bench_error(rank, error)
            raise

    def _bench_error(self, rank, error):
        import torch

        oom = isinstance(error, torch.OutOfMemoryError) and self._bench_phase in {"load", "warmup"}
        write_json(
            self._bench_output / f"error_rank{rank}.json",
            dict(
                phase=self._bench_phase,
                status="memory_insufficient" if oom else "failed",
                traceback=traceback.format_exc(),
            ),
        )

    def execute_model(self, *args, **kwargs):
        # Observe native inference failures before its RPC layer rewrites them.
        # Preserve all arguments, outputs and exceptions; add no synchronization.
        try:
            return super().execute_model(*args, **kwargs)
        except BaseException as error:
            self._bench_error(self.rank, error)
            raise

    def _bench_gather(self, operation, function):
        from vllm_omni.diffusion.worker.diffusion_worker import _run_and_gather_rank_values

        def guarded():
            try:
                return function()
            except BaseException as error:
                self._bench_error(self.rank, error)
                raise

        return _run_and_gather_rank_values(operation, guarded)

    def benchmark_prepare(self, modes, native_shape):
        def prepare():
            import torch

            pipe = self.model_runner.pipeline
            if type(pipe).__name__ == "Wan22Pipeline":
                from fastwan import prepare_worker

                return prepare_worker(self, modes, native_shape)
            from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import FastH3CheckpointSpec

            if type(pipe._fasth3_checkpoint) is not FastH3CheckpointSpec:
                raise ValueError("Expected unmodified native FastH3 checkpoint contract")
            if self.od_config.enable_diffusion_pipeline_profiler:
                raise ValueError("Native stage profiler adds CUDA synchronizations")
            self._bench_transformer = pipe.transformer
            self._bench_timer = (
                DiTEvents(torch, pipe._fasth3_checkpoint.num_inference_steps) if "dit" in modes else None
            )
            props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
            components = {}
            for name in ["transformer", "text_encoder", "video_vae", "audio_vae"]:
                module = getattr(pipe, name, None)
                if module is not None and hasattr(module, "parameters"):
                    components[name] = dict(
                        devices=sorted({str(p.device) for p in module.parameters()}),
                        dtypes=sorted({str(p.dtype) for p in module.parameters()}),
                    )
            return dict(
                rank=self.rank,
                name=props.name,
                uuid=str(props.uuid),
                total_memory_gib=props.total_memory / 2**30,
                capability=[props.major, props.minor],
                pipeline_class=type(pipe).__name__,
                pipeline_config=serializable(asdict(self.od_config)),
                components=components,
                **attention_info(pipe.transformer, native_shape, pipe._fasth3_checkpoint.vsa_sparsity),
            )

        return self._bench_gather("benchmark_prepare", prepare)

    def benchmark_arm(self, mode, phase, seed):
        def arm():
            import numpy as np
            import torch

            self._bench_phase = phase
            if self._bench_timer:
                self._bench_timer.detach()
            if mode == "dit":
                self._bench_timer.attach(self._bench_transformer)
            self._bench_mode = mode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.accelerator.synchronize()
            torch.accelerator.reset_peak_memory_stats()
            return dict(rank=self.rank, ready=True)

        # Native control-plane gather is before E2E and provides all-rank admission.
        return self._bench_gather("benchmark_arm", arm)

    def benchmark_collect(self, started, response_s):
        def collect():
            import torch
            import torch.distributed as dist

            from vllm_omni.diffusion.distributed.parallel_state import get_world_group

            # DISTRIBUTED E2E END: post-save control RPC + independent CPU barrier.
            # It includes RPC overhead and is explicitly separate from response E2E.
            dist.barrier(group=get_world_group().cpu_group)
            distributed_s = time.perf_counter() - started
            if self._bench_timer:
                self._bench_timer.detach()
            torch.accelerator.synchronize()  # OUTSIDE response/distributed E2E.
            durations = self._bench_timer.elapsed_ms() if self._bench_mode == "dit" else None
            return dict(
                rank=self.rank,
                response_e2e_s=response_s if self.rank == 0 else None,
                distributed_e2e_s=distributed_s,
                dit_step_ms=durations,
                peak_allocated_gib=torch.accelerator.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )

        return self._bench_gather("benchmark_collect", collect)
