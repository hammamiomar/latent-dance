"""Runtime artifact discovery and Hugging Face download helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from hambajuba2ba.config.sae import SAEConfig

logger = logging.getLogger(__name__)

UPSTREAM_SDXL_SAE_DIRS: dict[str, str] = {
    "down.2.1": "unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001",
    "mid.0": "unet.mid_block.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001",
    "up.0.0": "unet.up_blocks.0.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001",
    "up.0.1": "unet.up_blocks.0.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001",
}

RUNTIME_SAE_FILES: tuple[str, ...] = ("config.json", "state_dict.pth", "mean.pt")


def has_sae_weights(weights_dir: str | Path, blocks: list[str] | tuple[str, ...]) -> bool:
    """Return True when every requested block has a usable SAE checkpoint."""
    root = Path(weights_dir)
    for block in blocks:
        if find_sae_block_dir(root, block) is None:
            return False
    return True


def find_sae_block_dir(weights_dir: str | Path, block: str) -> Path | None:
    """Find a block checkpoint directory across supported public layouts."""
    for candidate in _candidate_block_dirs(Path(weights_dir), block):
        if (candidate / "config.json").exists() and (
            candidate / "state_dict.pth"
        ).exists():
            return candidate
    return None


def resolve_sae_weights_dir(config: SAEConfig) -> Path:
    """Resolve SAE weights, downloading public artifacts if local files are absent."""
    weights_dir = Path(config.weights_dir)
    if has_sae_weights(weights_dir, config.blocks):
        return weights_dir

    if not config.auto_download_weights:
        raise FileNotFoundError(
            f"SAE weights not found at {weights_dir}; automatic download is disabled"
        )

    logger.info(
        "SAE weights missing at %s; downloading %s/%s from Hugging Face",
        weights_dir,
        config.artifact_repo_id,
        config.artifact_weights_subdir,
    )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "huggingface_hub is required to download public SAE artifacts"
        ) from exc

    snapshot_root = Path(
        snapshot_download(
            repo_id=config.artifact_repo_id,
            repo_type=config.artifact_repo_type,
            allow_patterns=_allow_patterns(config),
            cache_dir=config.artifact_cache_dir,
        )
    )
    downloaded_dir = _downloaded_weights_root(snapshot_root, config.artifact_weights_subdir)

    if not has_sae_weights(downloaded_dir, config.blocks):
        raise FileNotFoundError(
            "Downloaded artifact repo does not contain complete SAE weights under "
            f"{config.artifact_weights_subdir or '<repo root>'!r}. "
            f"Expected blocks: {config.blocks}"
        )

    return downloaded_dir


def _allow_patterns(config: SAEConfig) -> list[str]:
    """Return exact runtime snapshot files for supported checkpoint layouts."""
    prefix = config.artifact_weights_subdir.strip("/")
    patterns: list[str] = []
    for block in config.blocks:
        candidates = [f"{block}/final", block]
        if upstream := UPSTREAM_SDXL_SAE_DIRS.get(block):
            candidates.append(upstream)

        for candidate in candidates:
            path = f"{prefix}/{candidate}" if prefix else candidate
            patterns.extend(f"{path}/{filename}" for filename in RUNTIME_SAE_FILES)
    return patterns


def _downloaded_weights_root(snapshot_root: Path, subdir: str) -> Path:
    prefix = subdir.strip("/")
    if not prefix:
        return snapshot_root
    return snapshot_root / prefix


def _candidate_block_dirs(root: Path, block: str) -> list[Path]:
    """Return checkpoint directories for short-name and upstream HF layouts."""
    candidates = [
        root / block / "final",
        root / block,
    ]
    if upstream := UPSTREAM_SDXL_SAE_DIRS.get(block):
        candidates.extend(
            [
                root / upstream,
                root / upstream / "final",
            ]
        )
    return candidates
