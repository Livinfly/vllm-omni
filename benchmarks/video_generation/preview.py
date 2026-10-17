"""Resolve native V2 metadata, shape, VAE cache and engine config without loading weights."""

import argparse
import json
import os
import sys
from pathlib import Path

from common import HERE, ROOT, read_json, write_json
from config import model_manifest
from runtime import launch_environment, resolve_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/fasth3_v2.json")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.gpus) < 1:
        parser.error("GPU counts must be positive")
    profile = read_json(args.config)
    manifest = model_manifest(args.model_path, profile, args.gpus)
    environment = launch_environment(dict(os.environ), HERE)
    if environment.get("LD_LIBRARY_PATH") != os.environ.get("LD_LIBRARY_PATH"):
        # glibc reads its library search path at process startup, before CUDA dlopen.
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    os.environ.clear()
    os.environ.update(environment)
    sys.path.insert(0, str(ROOT))
    result = dict(
        model_manifest=manifest,
        profile=profile,
        groups=[dict(gpus=n, **resolve_config(profile, args.model_path.resolve(), n)) for n in args.gpus],
    )
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
