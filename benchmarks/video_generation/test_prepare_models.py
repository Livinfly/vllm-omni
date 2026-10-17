"""Verify VAE binding against independent small safetensors fixtures."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import read_json, write_json
from config import METADATA, V1_METADATA
from prepare_models import prepare, verify_component, verify_download_origin


def write_tensors(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    header, chunks, offset = {}, [], 0
    for name, (shape, numbers) in values.items():
        chunk = struct.pack("<" + "f" * len(numbers), *numbers)
        header[name] = dict(dtype="F32", shape=shape, data_offsets=[offset, offset + len(chunk)])
        offset += len(chunk)
        chunks.append(chunk)
    metadata = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(metadata)) + metadata + b"".join(chunks))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.base = self.root / "v2", self.root / "base"
        self.source.mkdir()
        write_json(self.source / "fastvideo_inference.json", METADATA)
        stats = dict(latent_channels=1, latents_mean=[0.0], latents_std=[1.0])
        configs = [
            (self.source / "vae/config.json", dict(stats, decoder_num_attention_heads=2, decoder_attention_head_dim=2)),
            (self.source / "audio_vae/config.json", stats),
            (
                self.base / "FL2VA/video_vae/config.json",
                dict(stats, source_path="source", source_safetensors_path="model.safetensors"),
            ),
            (self.base / "FL2VA/audio_vae/config.json", dict(stats, source_safetensors_path="model.safetensors")),
        ]
        for path, config in configs:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, config)
        prefix = "decoder.transformer_blocks.0."
        reference = {
            prefix + "attn.to_q.weight": ([4, 1], [1, 2, 3, 4]),
            prefix + "attn.to_k.weight": ([4, 1], [11, 12, 13, 14]),
            prefix + "attn.to_v.weight": ([4, 1], [21, 22, 23, 24]),
            prefix + "ff.net.0.proj.weight": ([4, 1], [1, 2, 3, 4]),
            "encoder.down_blocks.0.resnets.0.conv_shortcut.weight": ([1], [8]),
        }
        native = {
            prefix + "attn.to_qkv.weight": ([12, 1], [1, 2, 11, 12, 21, 22, 3, 4, 13, 14, 23, 24]),
            prefix + "ff.w1.weight": ([4, 1], [3, 4, 1, 2]),
            "encoder.down.0.block.0.nin_shortcut.weight": ([1], [8]),
            "decoder.mask_token": ([1, 1, 4], [0, 0, 0, 0]),
        }
        write_tensors(self.source / "vae/model.safetensors", reference)
        write_tensors(self.base / "FL2VA/video_vae/source/model.safetensors", native)
        for path in [self.source / "audio_vae/model.safetensors", self.base / "FL2VA/audio_vae/model.safetensors"]:
            write_tensors(path, {"audio.weight": ([2], [4, 5])})
        for path in (self.base / "FL2VA").rglob("*"):
            if path.is_file():
                metadata = self.base / ".cache/huggingface/download" / (str(path.relative_to(self.base)) + ".metadata")
                metadata.parent.mkdir(parents=True, exist_ok=True)
                data = path.read_bytes()
                etag = (
                    hashlib.sha256(data).hexdigest()
                    if path.suffix == ".safetensors"
                    else hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                )
                metadata.write_text("a" * 40 + "\n" + etag + "\n0\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_qkv_head_interleave_swiglu_order_and_encoder_names(self):
        result = verify_component(self.source, self.base / "FL2VA", "video_vae")
        self.assertTrue(result["tensors_equal"])
        self.assertEqual(result["native_tensors"], 4)
        self.assertEqual(result["exported_tensors"], 5)

    def test_changed_tensor_and_normalization_are_rejected(self):
        target = self.source / "audio_vae/model.safetensors"
        write_tensors(target, {"audio.weight": ([2], [4, 6])})
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            verify_component(self.source, self.base / "FL2VA", "audio_vae")
        config = read_json(self.source / "vae/config.json")
        config["latents_std"] = [2.0]
        write_json(self.source / "vae/config.json", config)
        with self.assertRaisesRegex(ValueError, "normalization"):
            verify_component(self.source, self.base / "FL2VA", "video_vae")

    def test_hf_metadata_proves_revision_and_detects_modified_files(self):
        self.assertEqual(verify_download_origin(self.base)["revision"], "a" * 40)
        target = self.base / "FL2VA/audio_vae/config.json"
        target.write_text(target.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "ETag mismatch"):
            verify_download_origin(self.base)

    def test_v1_preserves_publisher_provenance(self):
        write_json(self.source / "fastvideo_inference.json", V1_METADATA)
        publisher = dict(
            model_id=V1_METADATA["model_id"],
            base_model="hf://MiniMaxAI/MiniMax-H3@" + "a" * 40,
            training_run_id="publisher-run",
            known_recipe_deviation="historic recipe",
        )
        write_json(self.source / "provenance.json", publisher)
        output = self.root / "v1-prepared"
        with patch("prepare_models.model_manifest", return_value={}):
            prepare(self.source, self.base, output, self.root / "cache")
        self.assertEqual(read_json(output / "provenance.json"), publisher)
        self.assertEqual(read_json(self.source / "provenance.json"), publisher)
        self.assertTrue(read_json(output / "vae_verification.json")["vae_checks"][0]["tensors_equal"])

    def test_prepared_copy_marks_local_binding_and_preserves_source(self):
        output, cache = self.root / "prepared", self.root / "cache"
        with patch("prepare_models.model_manifest", return_value={}):
            prepare(self.source, self.base, output, cache)
        provenance = read_json(output / "provenance.json")
        self.assertFalse(provenance["publisher_provided"])
        self.assertTrue(provenance["base_model"].endswith("@" + "a" * 40))
        self.assertFalse((self.source / "provenance.json").exists())
        self.assertTrue((output / "vae").samefile(self.source / "vae"))
        cached = cache / "models--MiniMaxAI--MiniMax-H3/snapshots" / ("a" * 40) / "FL2VA/video_vae"
        self.assertTrue(cached.samefile(self.base / "FL2VA/video_vae"))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare(self.source, self.base, output, cache)


if __name__ == "__main__":
    unittest.main()
