"""Stage 1: Generate images + capture sparse SAE activations.

Local orchestrator that filters prompts from a text-to-image dataset and
submits generation batches to the optional Modal ImageGenerator. Handles
resume by checking how many activations already exist on the output volume.

This script documents and reproduces the method used to create the public
Hugging Face label dataset. It is not required for normal runtime usage.

Prompt sources (tried in order):
    1. laion/laion-coco (gated, needs HF access approval)
    2. google/conceptual-captions (CC-BY-4.0, 3.3M captions, ungated)

Usage:
    uv sync --extra labeling-modal
    uv run python -m hambajuba2ba.labeling.stage1_generate [--n-images 50000] [--chunk-size 1000]

Download results:
    modal volume get hambajuba-sae-labeling-output /generated ./data/labeling/sae_images/generated/
    modal volume get hambajuba-sae-labeling-output /activations.jsonl ./data/labeling/sae_images/
"""

from __future__ import annotations

import argparse
from typing import Iterator


# ─── Prompt filtering ────────────────────────────────────────────


def _is_ascii_english(text: str, threshold: float = 0.9) -> bool:
    """Check if a string is predominantly ASCII (proxy for English)."""
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) >= threshold


def filter_prompt(text: str) -> str | None:
    """Apply quality filters to a single prompt.

    Filters: strip, drop < 5 words, drop > 200 chars, drop non-ASCII.
    Returns cleaned prompt or None if filtered.
    """
    text = text.strip()
    if len(text.split()) < 5:
        return None
    if len(text) > 200:
        return None
    if not _is_ascii_english(text, threshold=0.9):
        return None
    return text


def stream_filtered_prompts(n_total: int) -> Iterator[str]:
    """Stream deduplicated, filtered prompts from a caption dataset.

    Tries LAION-COCO first (richer captions); falls back to
    Conceptual Captions if LAION-COCO is gated/unavailable.
    Uses streaming mode so we never load full datasets into memory.
    """
    from datasets import load_dataset

    # Try LAION-COCO first, fall back to Conceptual Captions
    sources = [
        ("laion/laion-coco", "train", "top_caption"),
        ("google/conceptual-captions", "train", "caption"),
    ]

    for dataset_id, split, caption_field in sources:
        try:
            print(f"Trying {dataset_id}...")
            ds = load_dataset(dataset_id, split=split, streaming=True)
            # Probe first row to verify access
            first = next(iter(ds))
            if caption_field not in first:
                print(f"  Field '{caption_field}' not found, skipping")
                continue
            print(f"  Using {dataset_id} (field: {caption_field})")
            break
        except Exception as e:
            print(f"  {dataset_id} unavailable: {e}")
            continue
    else:
        raise RuntimeError("No prompt dataset available. Accept LAION-COCO terms "
                           "at https://huggingface.co/datasets/laion/laion-coco")

    # Re-open stream (we consumed one row in the probe)
    ds = load_dataset(dataset_id, split=split, streaming=True)

    seen: set[str] = set()
    yielded = 0
    scanned = 0

    for row in ds:
        scanned += 1
        raw = row.get(caption_field, "")
        if not raw:
            continue

        cleaned = filter_prompt(raw)
        if cleaned is None:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)

        yielded += 1
        if yielded % 10_000 == 0:
            print(f"  Filtered {yielded}/{n_total} prompts "
                  f"(scanned {scanned} rows)")

        yield cleaned

        if yielded >= n_total:
            break

    print(f"  Done: {yielded} prompts from {scanned} rows "
          f"({yielded / max(scanned, 1) * 100:.1f}% pass rate)")


# ─── CLI entrypoint ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: Generate images + capture SAE activations on Modal",
    )
    parser.add_argument(
        "--n-images", type=int, default=50_000,
        help="Total number of images to generate (default: 50000)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1_000,
        help="Batch size per Modal call (default: 1000)",
    )
    args = parser.parse_args()

    n_images: int = args.n_images
    chunk_size: int = args.chunk_size

    # Collect prompts BEFORE starting Modal (no GPU cost during filtering)
    print(f"Collecting {n_images} prompts...")
    all_prompts: list[str] = list(stream_filtered_prompts(n_images))

    # Now start Modal and run generation.
    try:
        import modal
        from hambajuba2ba.labeling.modal_app import (
            OUTPUT_VOLUME_NAME,
            ImageGenerator,
            app,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Stage 1 reproduction requires the optional Modal dependencies. "
            "Install them with `uv sync --extra labeling-modal` and run "
            "`modal setup` before retrying."
        ) from exc

    with modal.enable_output(), app.run():
        # Check resume state on Modal volume
        print("\nChecking for existing progress on Modal volume...")
        existing_count = ImageGenerator().count_existing.remote()
        print(f"  Found {existing_count} existing activations")

        remaining = n_images - existing_count
        if remaining <= 0:
            print(f"Already have {existing_count}/{n_images} images. Nothing to do.")
            return

        print(f"  Need {remaining} more images")

        # Skip prompts that were already processed
        prompts = all_prompts[existing_count:]
        print(f"  {len(prompts)} prompts ready for generation")

        if not prompts:
            print("No new prompts to process.")
            return

        # Submit chunks to Modal
        generator = ImageGenerator()
        total_chunks = (len(prompts) + chunk_size - 1) // chunk_size
        print(f"\nSubmitting {total_chunks} chunks of up to {chunk_size} images...")

        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i : i + chunk_size]
            start_idx = existing_count + i
            chunk_num = i // chunk_size + 1

            print(f"\n  Chunk {chunk_num}/{total_chunks}: "
                  f"images {start_idx}-{start_idx + len(chunk) - 1}")

            n_generated = generator.generate_batch.remote(chunk, start_idx)
            print(f"  -> Generated {n_generated} images")

        final_count = generator.count_existing.remote()
        print(f"\nStage 1 complete! {final_count} total images on volume.")

    print("\nDownload results:")
    print(
        f"  modal volume get {OUTPUT_VOLUME_NAME} "
        "/generated ./data/labeling/sae_images/generated/"
    )
    print(
        f"  modal volume get {OUTPUT_VOLUME_NAME} "
        "/activations.jsonl ./data/labeling/sae_images/"
    )


if __name__ == "__main__":
    main()
