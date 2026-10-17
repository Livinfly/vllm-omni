"""Validate our explicit benchmark engine's startup hook without model loading."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine


@pytest.fixture
def benchmark_engine():
    path = Path(__file__).resolve().parents[2] / "benchmarks/video_generation/fastwan_engine.py"
    spec = importlib.util.spec_from_file_location("fastwan_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = object.__new__(module.FastWanBenchmarkEngine)
    engine.od_config = SimpleNamespace(
        model_class_name="WanDMDPipeline",
        diffusion_attention_config=SimpleNamespace(
            per_role={
                "self": SimpleNamespace(
                    backend="FASTVIDEO_VSA",
                    fastvideo_vsa_allow_ulysses=True,
                    fastvideo_vsa_block_size=(4, 4, 4),
                    fastvideo_vsa_topk=144,
                )
            }
        ),
    )
    return engine


def test_fastwan_uses_native_engine_backend_extension(benchmark_engine, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks/video_generation"))
    cls = DiffusionEngine.resolve_engine_class(
        OmniDiffusionConfig(model="", engine_backend="fastwan_engine.FastWanBenchmarkEngine")
    )
    assert issubclass(cls, DiffusionEngine)
    for name in ("__init__", "add_req_and_wait_for_response"):
        assert getattr(cls, name) is getattr(DiffusionEngine, name)


def test_fastwan_startup_submits_no_extra_request(benchmark_engine):
    benchmark_engine._dummy_run = Mock(side_effect=AssertionError("extra startup request submitted"))
    benchmark_engine.run_startup_warmup()
    benchmark_engine._dummy_run.assert_not_called()
    assert benchmark_engine.od_config.diffusion_attention_config.per_role["self"].fastvideo_vsa_topk == 144


@pytest.mark.parametrize("model,topk", [("WanPipeline", 144), ("WanDMDPipeline", 64)])
def test_fastwan_startup_rejects_unreviewed_workload(benchmark_engine, model, topk):
    benchmark_engine.od_config.model_class_name = model
    benchmark_engine.od_config.diffusion_attention_config.per_role["self"].fastvideo_vsa_topk = topk
    with pytest.raises(ValueError, match="reviewed"):
        benchmark_engine.run_startup_warmup()
