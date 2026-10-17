"""Exercise worker-extension controls with native API doubles, without CUDA."""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from common import read_json
from extension import TimingExtension


@dataclass
class NativeConfig:
    enable_diffusion_pipeline_profiler: bool = False


class ExtensionTests(unittest.TestCase):
    def test_uuid_check_runs_before_native_loader(self):
        class NativeWorker:
            def __init__(self, *args, **kwargs):
                raise AssertionError("Native model loading must not begin on wrong GPU")

        class Worker(TimingExtension, NativeWorker):
            pass

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(
                os.environ,
                {
                    "VLLM_OMNI_BENCH_OUTPUT": folder,
                    "VLLM_OMNI_BENCH_SELECTED_UUIDS": json.dumps(["GPU-a", "GPU-b"]),
                },
            ),
            patch("extension.check_device_identity", side_effect=ValueError("UUID mismatch")),
        ):
            with self.assertRaisesRegex(ValueError, "UUID mismatch"):
                Worker(local_rank=1, rank=1, od_config=object())
            self.assertEqual(read_json(Path(folder) / "error_rank1.json")["phase"], "load")

    def test_e2e_only_allocates_no_events_and_collect_sync_is_after_boundary(self):
        class NativeSpec:
            num_inference_steps = 8
            vsa_sparsity = 0.8

        trace = []
        cuda = SimpleNamespace(
            get_device_properties=lambda n: SimpleNamespace(
                name="TestGPU", uuid="GPU-test", total_memory=2**30, major=9, minor=0
            ),
            current_device=lambda: 0,
            synchronize=lambda: trace.append("sync"),
            reset_peak_memory_stats=lambda: trace.append("reset"),
            manual_seed_all=lambda seed: None,
            max_memory_allocated=lambda: 10,
            max_memory_reserved=lambda: 20,
            Event=Mock(side_effect=AssertionError("e2e must not allocate events")),
        )
        torch = SimpleNamespace(
            cuda=cuda,
            accelerator=SimpleNamespace(
                current_device_index=cuda.current_device,
                synchronize=cuda.synchronize,
                reset_peak_memory_stats=cuda.reset_peak_memory_stats,
                max_memory_allocated=cuda.max_memory_allocated,
            ),
            manual_seed=lambda seed: None,
        )
        dist = SimpleNamespace(barrier=lambda **kwargs: trace.append("barrier"))
        torch.distributed = dist
        impl = type("MiniMaxH3VSAImpl", (), {})()
        impl.provider, impl.precision, impl.topk, impl.fallback_on_error = "fastvideo", "bf16", 64, True
        impl.resolve_h3_topk = lambda blocks, sparsity: 132
        transformer = SimpleNamespace(
            blocks=[
                SimpleNamespace(
                    attn=SimpleNamespace(
                        to_gate_compress=object(),
                        vsa_sparsity=0.8,
                        attention=SimpleNamespace(
                            attn_backend=SimpleNamespace(get_name=lambda: "FASTVIDEO_VSA"), attention=impl
                        ),
                    )
                )
            ],
            parameters=lambda: [],
        )
        worker = TimingExtension.__new__(TimingExtension)
        worker.rank = 0
        worker.od_config = NativeConfig()
        worker.model_runner = SimpleNamespace(
            pipeline=SimpleNamespace(_fasth3_checkpoint=NativeSpec(), transformer=transformer)
        )
        worker._bench_gather = lambda operation, function: [function()]
        with (
            patch.dict(
                sys.modules,
                {
                    "torch": torch,
                    "torch.distributed": dist,
                    "vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint": SimpleNamespace(
                        FastH3CheckpointSpec=NativeSpec
                    ),
                    "vllm_omni.diffusion.distributed.parallel_state": SimpleNamespace(
                        get_world_group=lambda: SimpleNamespace(cpu_group="gloo")
                    ),
                },
            ),
            patch("extension.time.perf_counter", side_effect=lambda: trace.append("clock") or 11),
        ):
            worker.benchmark_prepare(["e2e"], [768, 1344, 124, 37, 207])
            self.assertIsNone(worker._bench_timer)
            worker.benchmark_arm("e2e", "repeat", 1000)
            trace.clear()
            row = worker.benchmark_collect(1, 8)[0]
            self.assertEqual(trace, ["barrier", "clock", "sync"])
            self.assertIsNone(row["dit_step_ms"])
            self.assertEqual(row["distributed_e2e_s"], 10)
            self.assertEqual(row["response_e2e_s"], 8)
            cuda.Event.assert_not_called()

    def test_native_inference_failure_preserves_oom_and_phase(self):
        import torch

        failure = torch.OutOfMemoryError("native warmup OOM")

        class NativeWorker:
            def execute_model(self, *args, **kwargs):
                raise failure

        class Worker(TimingExtension, NativeWorker):
            pass

        worker = Worker.__new__(Worker)
        worker.rank, worker._bench_phase = 1, "warmup"
        with tempfile.TemporaryDirectory() as folder:
            worker._bench_output = Path(folder)
            with self.assertRaises(torch.OutOfMemoryError) as context:
                worker.execute_model(object(), od_config=object())
            self.assertIs(context.exception, failure)
            error = read_json(Path(folder) / "error_rank1.json")
            self.assertEqual(error["status"], "memory_insufficient")
            self.assertEqual(error["phase"], "warmup")
            self.assertIn("native warmup OOM", error["traceback"])

    def test_rank_control_error_preserves_phase_and_traceback(self):
        worker = TimingExtension.__new__(TimingExtension)
        worker.rank, worker._bench_phase = 0, "warmup"
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(
                sys.modules,
                {
                    "vllm_omni.diffusion.worker.diffusion_worker": SimpleNamespace(
                        _run_and_gather_rank_values=lambda operation, function: function()
                    ),
                },
            ),
        ):
            worker._bench_output = Path(folder)

            def fail():
                raise ValueError("real control error")

            with self.assertRaisesRegex(ValueError, "real control error"):
                worker._bench_gather("test", fail)
            error = read_json(Path(folder) / "error_rank0.json")
            self.assertEqual(error["phase"], "warmup")
            self.assertEqual(error["status"], "failed")
            self.assertIn("ValueError: real control error", error["traceback"])


if __name__ == "__main__":
    unittest.main()
