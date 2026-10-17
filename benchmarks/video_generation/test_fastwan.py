"""Admission checks for our local FastWan tile64/Ulysses adaptation."""

import unittest

from common import HERE, read_json
from config import validate_profile as validate_root_profile
from fastwan import COMPARABILITY, engine_kwargs, geometry, validate_profile, validate_world
from prepare_fastwan import MODEL_ID


class FastWanTests(unittest.TestCase):
    def setUp(self):
        self.profile = dict(
            model_id=MODEL_ID,
            adapter="fastwan21_14b",
            sampling=dict(
                height=720, width=1280, num_frames=81, fps=16, seed=1024, guidance_scale=1, num_inference_steps=3
            ),
            contract=dict(dmd_denoising_steps=[1000, 757, 522], flow_shift=8.0),
            attention=dict(
                backend="FASTVIDEO_VSA",
                provider="fastvideo",
                precision="bf16",
                checkpoint_sparsity=0.9,
                tile_size=64,
                topk=144,
                allow_ulysses=True,
            ),
            prompt="A cafe scene.",
            negative_prompt="",
            engine={},
        )

    def test_aligned_tile64_geometry_and_engine_configuration(self):
        validate_profile(self.profile)
        self.assertEqual(geometry(self.profile["sampling"]), ([21, 45, 80], 1440, 144))
        config = engine_kwargs(self.profile, "/model", 1)["diffusion_attention_config"]["per_role"]["self"]
        self.assertEqual(config["fastvideo_vsa_block_size"], (4, 4, 4))
        self.assertEqual(config["fastvideo_vsa_topk"], 144)
        self.assertTrue(config["fastvideo_vsa_allow_ulysses"])
        self.assertIn("自行适配", COMPARABILITY)

    def test_old_tile256_is_not_admitted(self):
        self.profile["attention"].update(tile_size=256, topk=36)
        with self.assertRaises(ValueError):
            validate_profile(self.profile)

    def test_reviewed_profile_selects_only_the_benchmark_startup_adapter(self):
        profile = read_json(HERE / "configs/fastwan21_14b.json")
        validate_root_profile(profile)
        self.assertEqual(profile["engine"]["engine_backend"], "fastwan_engine.FastWanBenchmarkEngine")
        profile["engine"]["engine_backend"] = "other.Engine"
        with self.assertRaises(ValueError):
            validate_root_profile(profile)

    def test_pure_ulysses_configuration_and_divisibility(self):
        for world in (1, 2, 4, 8):
            config = engine_kwargs(self.profile, "/model", world)
            self.assertEqual(config["num_gpus"], world)
            self.assertEqual(config["parallel_config"]["ulysses_degree"], world)
            self.assertEqual(config["parallel_config"]["ulysses_mode"], "strict")
            self.assertEqual(config["stage_0_parallel_config"], config["parallel_config"])
            self.assertEqual(config["stage_0_diffusion_attention_config"], config["diffusion_attention_config"])
            self.assertNotIn("default", config["diffusion_attention_config"])
        with self.assertRaises(ValueError):
            engine_kwargs(self.profile, "/model", 3)
        self.profile["sampling"].update(height=16, width=16, num_frames=1)
        with self.assertRaises(ValueError, msg="nondivisible sequences must not be padded"):
            validate_world(self.profile, 2)
