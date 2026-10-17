"""Bind verified local base VAEs to a FastH3 checkpoint without changing source assets.

Verification streams safetensors on CPU, reversing the export's names, QKV
head interleave and SwiGLU half order. No weights or checkpoint code execute.
"""

import argparse
import hashlib
import json
import re
import shlex
import struct
from pathlib import Path

from common import ROOT, read_json, sha256, write_json
from config import V1_MODEL_ID, model_manifest, validate_metadata
from runtime import native_vae_manifest


def tensors(paths):
    result = {}
    for path in paths:
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in result:
                raise ValueError(f"Duplicate tensor {name}")
            start, end = entry["data_offsets"]
            if not 0 <= start <= end <= path.stat().st_size - 8 - size:
                raise ValueError(f"Invalid tensor offsets: {path}/{name}")
            result[name] = dict(entry, path=path, offset=8 + size + start, length=end - start)
    if not result:
        raise ValueError("Missing safetensors weights")
    return result


def digest_parts(parts, *, require_zero=False):
    digest = hashlib.sha256()
    for tensor, offset, length in parts:
        with tensor["path"].open("rb") as stream:
            stream.seek(tensor["offset"] + offset)
            while length:
                chunk = stream.read(min(length, 8 * 1024**2))
                if not chunk:
                    raise EOFError(str(tensor["path"]))
                if require_zero and any(chunk):
                    raise ValueError("Dropped decoder.mask_token must contain only zeros")
                digest.update(chunk)
                length -= len(chunk)
    return digest.hexdigest()


def source_parts(name, native, reference, heads, dim):
    """Return export byte ranges in the native tensor's exact row order."""
    key = name.replace("decoder.x_embedder.", "decoder.proj_in.")
    key = key.replace(".attn.to_out.", ".attn.to_out.0.").replace(".ff.w2.", ".ff.net.2.")
    key = re.sub(r"^encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2.", key)
    key = re.sub(r"^encoder\.down\.(\d+)\.downsample\.", r"encoder.down_blocks.\1.downsamplers.0.", key)
    key = key.replace(".nin_shortcut.", ".conv_shortcut.")
    if ".attn.to_qkv." in key:
        keys = [key.replace(".attn.to_qkv.", f".attn.to_{part}.") for part in ("q", "k", "v")]
        parts = [reference[k] for k in keys]
        expected_shape = [heads * dim, *native["shape"][1:]]
        if native["shape"][0] != 3 * heads * dim or any(
            p["shape"] != expected_shape or p["dtype"] != native["dtype"] for p in parts
        ):
            raise ValueError(f"QKV shape/dtype mismatch: {name}")
        if any(p["length"] % heads for p in parts):
            raise ValueError(f"QKV head alignment mismatch: {name}")
        ranges = [(p, head * (p["length"] // heads), p["length"] // heads) for head in range(heads) for p in parts]
        return ranges, keys
    swap = ".ff.w1." in key
    key = key.replace(".ff.w1.", ".ff.net.0.proj.")
    tensor = reference[key]
    if tensor["shape"] != native["shape"] or tensor["dtype"] != native["dtype"]:
        raise ValueError(f"Tensor shape/dtype mismatch: {name}")
    if swap:
        if tensor["shape"][0] % 2 or tensor["length"] % 2:
            raise ValueError(f"SwiGLU half alignment mismatch: {name}")
        half = tensor["length"] // 2
        return [(tensor, half, half), (tensor, 0, half)], [key]
    return [(tensor, 0, tensor["length"])], [key]


def verify_component(source, base, component):
    video = component == "video_vae"
    exported = source / ("vae" if video else "audio_vae")
    config, native_config = read_json(exported / "config.json"), read_json(base / component / "config.json")
    for key in ["latent_channels", "latents_mean", "latents_std"]:
        if config[key] != native_config[key]:
            raise ValueError(f"VAE normalization mismatch: {component}/{key}")
    native_file = base / component / native_config.get("source_path", ".") / native_config["source_safetensors_path"]
    native, reference = tensors([native_file]), tensors(sorted(exported.glob("*.safetensors")))
    heads = config["decoder_num_attention_heads"] if video else 0
    dim = config["decoder_attention_head_dim"] if video else 0
    used, proof = set(), hashlib.sha256()
    for name, tensor in sorted(native.items()):
        actual = digest_parts([(tensor, 0, tensor["length"])], require_zero=video and name == "decoder.mask_token")
        if video and name == "decoder.mask_token":
            if tensor["dtype"] != "F32" or tensor["shape"] != [1, 1, heads * dim]:
                raise ValueError("Invalid dropped decoder.mask_token")
        else:
            if video:
                parts, keys = source_parts(name, tensor, reference, heads, dim)
            else:
                ref = reference[name]
                if (ref["shape"], ref["dtype"]) != (tensor["shape"], tensor["dtype"]):
                    raise ValueError(f"Audio VAE shape/dtype mismatch: {name}")
                parts, keys = [(ref, 0, ref["length"])], [name]
            if sum(p[2] for p in parts) != tensor["length"] or digest_parts(parts) != actual:
                raise ValueError(f"VAE tensor content mismatch: {component}/{name}")
            used.update(keys)
        proof.update(json.dumps([name, tensor["dtype"], tensor["shape"], actual]).encode())
    if used != set(reference):
        raise ValueError(f"Unmatched exported VAE tensors: {sorted(set(reference) - used)}")
    return dict(
        component=component,
        native_tensors=len(native),
        exported_tensors=len(reference),
        canonical_tensor_sha256=proof.hexdigest(),
        normalization_equal=True,
        tensors_equal=True,
        omitted_zero_mask_token=video,
    )


def verify_download_origin(base):
    """Authenticate local files using the HF local-dir commit and ETag records."""
    revisions, files = set(), {}
    for component in ["video_vae", "audio_vae"]:
        for path in sorted((base / "FL2VA" / component).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(base)
            metadata = base / ".cache/huggingface/download" / (str(relative) + ".metadata")
            lines = metadata.read_text().splitlines()
            revision, etag = lines[:2]
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError(f"Invalid HF revision: {metadata}")
            revisions.add(revision)
            if len(etag) == 64:
                digest = sha256(path)
            elif len(etag) == 40:
                blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                        blob.update(chunk)
                digest = blob.hexdigest()
            else:
                raise ValueError(f"Unsupported HF ETag: {metadata}")
            if digest != etag:
                raise ValueError(f"HF ETag mismatch: {path}")
            files[str(relative)] = dict(revision=revision, etag=etag, verified=True)
    if len(revisions) != 1 or not files:
        raise ValueError(f"Native VAE files must have one verified HF revision, got {revisions}")
    return dict(repo="MiniMaxAI/MiniMax-H3", revision=revisions.pop(), files=files)


def prepare(source, base, output, cache):
    source, base, output, cache = (Path(p).resolve() for p in (source, base, output, cache))
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    if source in output.parents:
        raise ValueError("Output must be outside the source checkpoint")
    metadata = read_json(source / "fastvideo_inference.json")
    validate_metadata(metadata)
    publisher = read_json(source / "provenance.json") if (source / "provenance.json").is_file() else None
    native_vae_manifest(base / "FL2VA")
    print("Verifying local HF commit/ETags for native VAE assets", flush=True)
    origin = verify_download_origin(base)
    checks = []
    for name in ["video_vae", "audio_vae"]:
        print(f"Comparing every {name} tensor with checkpoint (CPU streaming)", flush=True)
        checks.append(verify_component(source, base / "FL2VA", name))
    snapshot = cache / "models--MiniMaxAI--MiniMax-H3/snapshots" / origin["revision"] / "FL2VA"
    for name in ["video_vae", "audio_vae"]:
        target = snapshot / name
        if target.exists() and target.resolve() != (base / "FL2VA" / name).resolve():
            raise ValueError(f"Conflicting HF snapshot: {target}")
    output.mkdir(parents=True)
    for item in source.iterdir():
        if item.name not in {".cache", "provenance.json", "vae_verification.json", "environment.sh"}:
            (output / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    if not (output / "model_index.json").exists():
        write_json(output / "model_index.json", {"_class_name": "MiniMaxH3ModularPipeline"})
    evidence = dict(
        origin=origin,
        vae_checks=checks,
        source=str(source),
        native_base=str(base),
        scope="Local inference VAE binding verified against checkpoint; not publisher or training provenance.",
    )
    write_json(output / "vae_verification.json", evidence)
    write_json(
        output / "provenance.json",
        dict(
            schema_version="omni-local-vae-binding-v1",
            model_id=metadata["model_id"],
            base_model="hf://MiniMaxAI/MiniMax-H3@" + origin["revision"],
            origin="locally_verified_vae_binding",
            publisher_provided=False,
            evidence="vae_verification.json",
            evidence_sha256=sha256(output / "vae_verification.json"),
        ),
    )
    if publisher is not None:
        if (
            publisher.get("model_id") != metadata["model_id"]
            or publisher.get("base_model") != "hf://MiniMaxAI/MiniMax-H3@" + origin["revision"]
        ):
            raise ValueError("Publisher provenance disagrees with checkpoint or verified base revision")
        write_json(output / "provenance.json", publisher)
    version = "v1" if metadata["model_id"] == V1_MODEL_ID else "v2"
    profile = read_json(Path(__file__).parent / f"configs/fasth3_{version}.json")
    model_manifest(output, profile, [1])
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ["video_vae", "audio_vae"]:
        target = snapshot / name
        if not target.exists():
            target.symlink_to(base / "FL2VA" / name, target_is_directory=True)
    (output / "environment.sh").write_text("export HF_HUB_CACHE=" + shlex.quote(str(cache)) + "\n")
    print(f"Prepared: {output}\nsource {output / 'environment.sh'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Complete FastVideo FastH3 snapshot")
    parser.add_argument("--base-model", type=Path, required=True, help="Native MiniMax-H3 with HF download metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-hub-cache", type=Path, default=ROOT / ".cache/benchmark/huggingface/hub")
    args = parser.parse_args()
    prepare(args.source, args.base_model, args.output, args.hf_hub_cache)


if __name__ == "__main__":
    main()
