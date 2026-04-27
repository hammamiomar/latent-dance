#!/usr/bin/env python3
"""Download or materialize public SAE weights from Hugging Face."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from hambajuba2ba.artifacts import (
    find_sae_block_dir,
    has_sae_weights,
    resolve_sae_weights_dir,
)
from hambajuba2ba.config import load_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that weights are resolvable.",
    )
    parser.add_argument(
        "--materialize",
        type=Path,
        help="Copy resolved weights into this directory for tools that need real files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_from_env().sae
    resolved = resolve_sae_weights_dir(config)

    if not has_sae_weights(resolved, config.blocks):
        raise SystemExit(f"Resolved path is incomplete: {resolved}")

    print(f"SAE weights resolved: {resolved}")

    if args.materialize:
        target = args.materialize
        target.mkdir(parents=True, exist_ok=True)
        for block in config.blocks:
            source = find_sae_block_dir(resolved, block)
            if source is None:
                raise SystemExit(f"Missing SAE weights for block: {block}")
            destination = target / block
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        print(f"Materialized SAE weights into: {target}")

    if args.check:
        return


if __name__ == "__main__":
    main()
