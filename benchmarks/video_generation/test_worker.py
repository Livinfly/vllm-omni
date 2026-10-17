"""Client integration: the native API/save boundary, rank records and teardown."""

import io
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import worker
from common import HERE, ROOT, read_json, write_json


class WorkerTests(unittest.TestCase):
    def exercise(self, fail=False, missing_metadata=False):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            directory = Path(folder)
            p = read_json(HERE / "configs/fasth3_v2.json")
            plan = dict(
                models={"fasth3_v2": dict(profile=p, path="/model")},
                selected_gpu_uuids=["GPU-a", "GPU-b"],
                modes=["dit", "e2e"],
                warmup=1,
                repeats=1,
                validate_video=True,
            )
            write_json(directory / "plan.json", plan)
            stack.enter_context(
                patch.object(
                    sys,
                    "argv",
                    [
                        "worker.py",
                        "--plan",
                        str(directory / "plan.json"),
                        "--model",
                        "fasth3_v2",
                        "--gpus",
                        "2",
                        "--output",
                        str(directory),
                    ],
                )
            )
            generated, closed, controls = [], [], []

            class Omni:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def generate(self, prompt, **kwargs):
                    generated.append((prompt, kwargs))
                    return [object()]

                def close(self):
                    closed.append(True)

            fake_torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="test"))
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch,
                        "vllm_omni": SimpleNamespace(__file__=str(ROOT / "vllm_omni/__init__.py")),
                        "vllm_omni.entrypoints.omni": SimpleNamespace(Omni=Omni),
                        "vllm_omni.inputs.data": SimpleNamespace(OmniDiffusionSamplingParams=lambda **kwargs: kwargs),
                        "vllm_omni.diffusion.utils.media_utils": SimpleNamespace(mux_video_audio_bytes=None),
                    },
                )
            )
            def version(name):
                if missing_metadata and name == "vllm-omni":
                    raise worker.importlib.metadata.PackageNotFoundError(name)
                return "test"

            stack.enter_context(patch.object(worker.importlib.metadata, "version", side_effect=version))
            stack.enter_context(
                patch.object(
                    worker,
                    "resolve_config",
                    return_value=dict(
                        pipeline_config={}, native_shape=[768, 1344, 124, 37, 207], expected_dit_forwards=8
                    ),
                )
            )
            stack.enter_context(patch.object(worker, "extract_media", return_value=(b"encoded mp4", None, 32000)))

            def media(path):
                self.assertEqual(path.read_bytes(), b"encoded mp4")
                return dict(width=1344, height=768, frames=124, fps=24)

            stack.enter_context(patch.object(worker, "video_metadata", side_effect=media))
            current_mode = []

            def rpc(engine, method, world, *args):
                controls.append(method)
                self.assertEqual(world, 2)
                if method == "benchmark_prepare":
                    return [
                        dict(
                            rank=i,
                            name="GPU",
                            pipeline_config={},
                            attention_implementations=[],
                            warnings=[],
                            total_memory_gib=96,
                        )
                        for i in range(2)
                    ]
                if method == "benchmark_arm":
                    current_mode[:] = [args[0]]
                    return []
                # The MP4 must already exist before the collect RPC.
                self.assertEqual(len(list(directory.glob("*.mp4"))), len(generated))
                if fail:
                    raise RuntimeError("native rank failed")
                return [
                    dict(
                        rank=i,
                        response_e2e_s=args[1] if i == 0 else None,
                        distributed_e2e_s=args[1] + 0.01,
                        dit_step_ms=[1] * 8 if current_mode[0] == "dit" else None,
                        peak_allocated_gib=1,
                        peak_reserved_gib=2,
                    )
                    for i in range(2)
                ]

            stack.enter_context(patch.object(worker, "rpc", side_effect=rpc))
            with redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "native rank failed"):
                        worker.main()
                else:
                    worker.main()
            self.assertEqual(closed, [True])
            if fail:
                error = read_json(directory / "error_client.json")
                self.assertEqual(error["phase"], "warmup")
                self.assertIn("native rank failed", error["traceback"])
            else:
                effective = read_json(directory / "effective_config.json")
                self.assertEqual(effective["package_metadata_missing"], ["vllm-omni"] if missing_metadata else [])
                self.assertEqual(effective["packages"]["vllm-omni"], None if missing_metadata else "test")
                self.assertEqual(effective["packages"]["torch"], "test")
                result = read_json(directory / "metrics.json")
                self.assertEqual(result["status"], "complete")
                self.assertEqual(len(result["records"]), 4)
                for row in result["records"]:
                    self.assertEqual(len(row["per_rank"]), 2)
                    self.assertIn("video_sha256", row)
                    self.assertEqual("dit_forward_sum_max_rank_s" in row, row["mode"] == "dit")
                    self.assertGreater(row["response_e2e_s"], row["api_only_s"])
                self.assertEqual(controls.count("benchmark_arm"), 4)
                self.assertEqual(controls.count("benchmark_collect"), 4)

    def test_dual_mode_client_and_media_save(self):
        self.exercise()

    def test_native_failure_closes_engine_and_preserves_error(self):
        self.exercise(fail=True)

    def test_source_checkout_without_package_metadata_completes_measurements(self):
        self.exercise(missing_metadata=True)


if __name__ == "__main__":
    unittest.main()
