"""Our FastWan benchmark engine: use the harness's full-resolution warmup."""

from vllm.logger import init_logger

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

logger = init_logger(__name__)


class FastWanBenchmarkEngine(DiffusionEngine):
    def run_startup_warmup(self) -> None:
        # Our local adaptation, selected explicitly via native engine_backend.
        # Omni's 512x512/single-frame dummy has only 64 VSA blocks, below the
        # fixed topk144 workload. The harness instead runs one complete 720p
        # warmup before measurement; keep the native sampler/kernel unchanged.
        config = self.od_config
        spec = config.diffusion_attention_config.per_role.get("self")
        if (
            config.model_class_name != "WanDMDPipeline"
            or spec is None
            or spec.backend != "FASTVIDEO_VSA"
            or not spec.fastvideo_vsa_allow_ulysses
            or spec.fastvideo_vsa_block_size != (4, 4, 4)
            or spec.fastvideo_vsa_topk != 144
        ):
            raise ValueError("Our FastWan benchmark engine requires the reviewed tile64/topk144 DMD profile")
        logger.info(
            "Our FastWan benchmark adaptation skips the small startup dummy (64 blocks < topk144); "
            "the harness runs one full-resolution warmup followed by one measured request."
        )
