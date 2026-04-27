"""Audio upload and processing endpoints.

Handles audio file upload, stem separation via Demucs,
and feature extraction for SAE steering.

Also serves stem audio files for frontend playback with Web Audio API.

Cache directory structure:
  ~/.cache/hambajuba2ba/audio/
    ├── {content_hash}/           # Hash of audio content for deduplication
    │   ├── mix.wav               # Original audio
    │   ├── bass.wav              # Separated stems
    │   ├── drums.wav
    │   ├── vocals.wav
    │   └── other.wav
    └── ...
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, Depends, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.caching import CacheManager
from app.dependencies import get_audio_cache, get_config
from hambajuba2ba.config import PipelineConfig
from hambajuba2ba.audio import (
    StemSeparator,
    extract_all_features,
    extract_virtual_stems,
)
from hambajuba2ba.audio.beat_onset_detection import detect_beats
from hambajuba2ba.audio.coupling import CrossStemFeatures
from hambajuba2ba.audio.features import FEATURE_CACHE_VERSION, StemFeatures
from hambajuba2ba.audio.separator import get_audio_duration

logger = logging.getLogger("app.routers.audio")

router = APIRouter(prefix="/api/audio", tags=["audio"])

# Also keep a temp dir for session-specific files (cleaned on exit)
AUDIO_TEMP_DIR = Path(tempfile.mkdtemp(prefix="hambajuba_audio_"))
logger.info(f"Audio temp directory: {AUDIO_TEMP_DIR}")


def _cleanup_temp_storage():
    """Cleanup temp storage on process exit (cache persists)."""
    if AUDIO_TEMP_DIR.exists():
        shutil.rmtree(AUDIO_TEMP_DIR, ignore_errors=True)
        logger.info(f"Cleaned up temp storage: {AUDIO_TEMP_DIR}")


def _get_cache_root(config: PipelineConfig) -> Path:
    """Resolve and ensure the configured cache root."""
    cache_root = Path(config.audio.stems_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _hash_audio_content(content: bytes) -> str:
    """Generate a short hash of audio content for cache key."""
    return hashlib.sha256(content).hexdigest()[:16]


def _get_cached_stems(
    content_hash: str,
    cache_root: Path,
) -> Optional[Tuple[Dict[str, str], str]]:
    """Check if stems are cached for this audio content.

    Returns:
        Tuple of (stem_files dict, mix_file path) if cached, None otherwise
    """
    cache_dir = cache_root / content_hash
    if not cache_dir.exists():
        return None

    # Check all required stems exist
    stem_names = ["bass", "drums", "vocals", "other"]
    stem_files = {}
    for stem in stem_names:
        stem_path = cache_dir / f"{stem}.wav"
        if not stem_path.exists():
            logger.warning(f"Cache incomplete for {content_hash}: missing {stem}.wav")
            return None
        stem_files[stem] = str(stem_path)

    # Check mix exists (any extension)
    mix_files = list(cache_dir.glob("mix.*"))
    if not mix_files:
        logger.warning(f"Cache incomplete for {content_hash}: missing mix file")
        return None

    logger.info(f"Cache hit for audio {content_hash}")
    return stem_files, str(mix_files[0])


atexit.register(_cleanup_temp_storage)


def _get_device() -> str:
    """Get the best available device for audio processing."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        logger.warning("CUDA not available, using CPU for Demucs (slower)")
        return "cpu"


def _save_stems_to_cache(
    content_hash: str,
    stems: Dict[str, np.ndarray],
    cache_root: Path,
    sr: int = 44100,
) -> Dict[str, str]:
    """Save separated stems as WAV files to cache.

    Args:
        content_hash: Hash of audio content for cache key
        stems: Dictionary of stem name to audio array
        cache_root: Root cache directory
        sr: Sample rate

    Returns:
        Dictionary of stem name to file path
    """
    cache_dir = cache_root / content_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    stem_files = {}
    for stem_name, audio_data in stems.items():
        stem_path = cache_dir / f"{stem_name}.wav"
        sf.write(str(stem_path), audio_data, sr)
        stem_files[stem_name] = str(stem_path)
        logger.debug(f"Cached stem {stem_name} to {stem_path}")

    return stem_files


def _save_mix_to_cache(
    content_hash: str,
    source_path: str,
    cache_root: Path,
) -> str:
    """Copy the original mix file to cache.

    Args:
        content_hash: Hash of audio content for cache key
        source_path: Path to the uploaded/downloaded audio file
        cache_root: Root cache directory

    Returns:
        Path to the cached mix file
    """
    cache_dir = cache_root / content_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine extension from source
    ext = Path(source_path).suffix or ".wav"
    mix_path = cache_dir / f"mix{ext}"
    shutil.copy2(source_path, mix_path)
    logger.debug(f"Cached mix to {mix_path}")
    return str(mix_path)


# ============================================================================
# Feature Cache Helpers
# ============================================================================


def _feature_cache_dir(content_hash: str, cache_root: Path) -> Path:
    return cache_root / content_hash / "features"


def _load_feature_cache(
    content_hash: str,
    cache_root: Path,
    feature_level: str,
    coupling_stems: str,
    sample_rate: int,
) -> Optional[Tuple[Dict[str, StemFeatures], Optional[CrossStemFeatures], float]]:
    """Load cached features and cross-stem data if available and valid."""
    cache_dir = _feature_cache_dir(content_hash, cache_root)
    meta_path = cache_dir / "features_meta.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None

    if meta.get("version") != FEATURE_CACHE_VERSION:
        return None
    if meta.get("feature_level") != feature_level:
        return None
    if meta.get("coupling_stems") != coupling_stems:
        return None
    if meta.get("sample_rate") != sample_rate:
        return None

    stems = meta.get("stems", [])
    if not stems:
        return None

    features: Dict[str, StemFeatures] = {}
    for stem in stems:
        stem_path = cache_dir / f"{stem}.npz"
        if not stem_path.exists():
            return None
        features[stem] = StemFeatures.from_npz(stem_path)

    cross_path = cache_dir / "cross_stem.npz"
    cross_stem = CrossStemFeatures.from_npz(cross_path) if cross_path.exists() else None

    bpm = float(meta.get("bpm", 120.0))
    return features, cross_stem, bpm


def _save_feature_cache(
    content_hash: str,
    features: Dict[str, StemFeatures],
    cross_stem: Optional[CrossStemFeatures],
    bpm: float,
    cache_root: Path,
    feature_level: str,
    coupling_stems: str,
    sample_rate: int,
) -> None:
    """Persist extracted features to disk for reuse."""
    cache_dir = _feature_cache_dir(content_hash, cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save per-stem features
    for stem, feat in features.items():
        feat_path = cache_dir / f"{stem}.npz"
        feat.to_npz(feat_path)

    # Save cross-stem features if present
    if cross_stem is not None:
        cross_path = cache_dir / "cross_stem.npz"
        cross_stem.to_npz(cross_path)

    meta = {
        "version": FEATURE_CACHE_VERSION,
        "bpm": bpm,
        "feature_level": feature_level,
        "coupling_stems": coupling_stems,
        "sample_rate": sample_rate,
        "stems": list(features.keys()),
    }
    meta_path = cache_dir / "features_meta.json"
    meta_path.write_text(json.dumps(meta))



class AudioUploadResponse(BaseModel):
    """Response from audio upload."""

    audio_id: str
    stems: List[str]
    duration: float
    bpm: float = 120.0  # Detected tempo


class AudioStatusResponse(BaseModel):
    """Audio processing status."""

    audio_id: str
    status: str
    progress: float = 0.0
    stems: List[str] = []
    duration: float = 0.0
    bpm: float = 0.0


# Track upload/processing status (thread-safe)
_processing_status: Dict[str, dict] = {}
_status_lock = asyncio.Lock()


async def _set_status(audio_id: str, status: dict) -> None:
    """Thread-safe status update."""
    async with _status_lock:
        _processing_status[audio_id] = status


async def _get_status(audio_id: str) -> dict | None:
    """Thread-safe status read."""
    async with _status_lock:
        return _processing_status.get(audio_id)


async def _delete_status(audio_id: str) -> None:
    """Thread-safe status deletion."""
    async with _status_lock:
        _processing_status.pop(audio_id, None)


async def _process_audio_pipeline(
    *,
    audio_id: str,
    tmp_path: str,
    content_hash: str,
    config: PipelineConfig,
    audio_cache: CacheManager,
    cache_root: Path,
    filename: str | None = None,
) -> AudioUploadResponse:
    """Shared processing pipeline for audio files (upload + YouTube).

    Handles cache lookup, stem separation, feature extraction, and caching.
    Updates processing status throughout. Always cleans up temp files.
    """
    try:
        # Check cache first
        cached = _get_cached_stems(content_hash, cache_root)
        if cached:
            stem_files, mix_file = cached
            logger.info(f"Audio {audio_id}: Using cached stems from {content_hash}")
            await _set_status(audio_id, {"status": "loading_cache", "progress": 0.5})

            # Load stems from cache for feature extraction
            import librosa
            stems = {}
            for stem_name, stem_path in stem_files.items():
                audio_data, _ = librosa.load(stem_path, sr=config.audio.sample_rate, mono=True)
                stems[stem_name] = audio_data.astype(np.float32)

            duration = get_audio_duration(mix_file)
        else:
            await _set_status(audio_id, {"status": "separating", "progress": 0.1})

            # Get duration
            duration = get_audio_duration(tmp_path)
            logger.info(f"Audio {audio_id}: Duration={duration:.2f}s, starting separation...")

            # Separate stems (auto-detect device, use configured cache dir)
            device = _get_device()
            logger.info(f"Audio {audio_id}: Using device={device} for separation")
            separator = StemSeparator(
                device=device,
                output_dir=config.audio.stems_cache_dir,
            )
            stems = await separator.separate(tmp_path, sample_rate=config.audio.sample_rate)
            logger.info(f"Audio {audio_id}: Separation complete, stems={list(stems.keys())}")

            await _set_status(audio_id, {"status": "caching", "progress": 0.6})
            logger.info(f"Audio {audio_id}: Caching stems to {content_hash}...")

            # Save stems and mix to cache for reuse
            stem_files = _save_stems_to_cache(content_hash, stems, cache_root, sr=config.audio.sample_rate)
            mix_file = _save_mix_to_cache(content_hash, tmp_path, cache_root)
            logger.info(f"Audio {audio_id}: Cached {len(stem_files)} stems + mix")

        # Check feature cache (persists across restarts)
        feature_level = config.audio.feature_level
        coupling_stems = config.audio.coupling_stems
        feature_cache = None

        if config.audio.enable_feature_cache:
            feature_cache = _load_feature_cache(
                content_hash,
                cache_root,
                feature_level,
                coupling_stems,
                config.audio.sample_rate,
            )

        if feature_cache:
            # Load features from disk cache
            features, cross_stem_features, bpm = feature_cache
            all_stem_names = list(features.keys())
            # Derive duration from first stem
            if all_stem_names:
                duration = features[all_stem_names[0]].duration
            logger.info(f"Audio {audio_id}: Loaded features from cache (v{FEATURE_CACHE_VERSION})")
        else:
            await _set_status(audio_id, {"status": "extracting", "progress": 0.7})
            logger.info(f"Audio {audio_id}: Detecting BPM and extracting virtual stems...")

            # Detect BPM using Beat This! (ISMIR 2024 state-of-the-art)
            # Use drums stem for best accuracy, fallback to mix
            drums_audio = stems.get("drums")
            if drums_audio is None:
                drums_audio = sum(stems.values())  # Mix fallback
            beats, bpm = detect_beats(drums_audio, sr=config.audio.sample_rate)
            logger.info(f"Audio {audio_id}: Beat detection complete: BPM={bpm:.1f}, beats={len(beats)}")

            # Extract virtual stems (bandpass filtering)
            all_stems = extract_virtual_stems(stems, sr=config.audio.sample_rate)
            all_stem_names = list(all_stems.keys())
            logger.info(f"Audio {audio_id}: Virtual stems extracted: {all_stem_names}")

            await _set_status(audio_id, {"status": "extracting_features", "progress": 0.8})
            logger.info(f"Audio {audio_id}: Extracting perceptual features...")

            # Extract perceptual features + cross-stem coupling in one call
            sr = config.audio_sample_rate
            fps = int(config.fps)
            features, cross_stem_features = extract_all_features(
                all_stems,
                sr=sr,
                fps=fps,
                bpm=bpm,
                feature_level=feature_level,
                feature_backend=config.audio.feature_backend,
                feature_device=config.audio.feature_device,
                chroma_mode=config.audio.chroma_mode,
                hpss_backend=config.audio.hpss_backend,
                coupling_stems=coupling_stems,
            )
            logger.info(f"Audio {audio_id}: Feature extraction + cross-stem coupling complete")

            # Save features to disk cache for future restarts
            if config.audio.enable_feature_cache:
                _save_feature_cache(
                    content_hash=content_hash,
                    features=features,
                    cross_stem=cross_stem_features,
                    bpm=bpm,
                    cache_root=cache_root,
                    feature_level=feature_level,
                    coupling_stems=coupling_stems,
                    sample_rate=config.audio.sample_rate,
                )
                logger.info(f"Audio {audio_id}: Saved features to disk cache")

        # Cache for streaming session (includes file paths for serving)
        cache_payload = {
            "features": features,
            "cross_stem_features": cross_stem_features,
            "duration": duration,
            "bpm": bpm,
            "sr": config.audio.sample_rate,
            "stems": all_stem_names,
            "stem_files": stem_files,
            "mix_file": mix_file,
        }
        if filename is not None:
            cache_payload["filename"] = filename

        audio_cache.set(
            audio_id,
            cache_payload,
            ttl=config.audio.cache_ttl_seconds,
        )
        logger.info(f"Audio {audio_id}: Cached with TTL=3600s")

        await _set_status(
            audio_id,
            {
                "status": "complete",
                "progress": 1.0,
                "stems": all_stem_names,
                "duration": duration,
                "bpm": bpm,
            },
        )
        logger.info(f"Audio {audio_id}: ✓ Complete! duration={duration:.2f}s, bpm={bpm:.1f}, stems={all_stem_names}")

        # Clean up processing status — data is now in audio_cache,
        # so get_audio_status falls through to cache lookup
        await _delete_status(audio_id)

        return AudioUploadResponse(
            audio_id=audio_id,
            stems=all_stem_names,
            duration=duration,
            bpm=bpm,
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        await _set_status(audio_id, {"status": "error", "progress": 0.0, "error": "validation_failed"})
        raise

    except (ValueError, OSError) as e:
        # Known error types from audio processing
        await _set_status(audio_id, {"status": "error", "progress": 0.0, "error": str(e)})
        logger.error(f"Audio {audio_id}: Processing error - {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {e}")

    except Exception as e:
        # Unexpected errors - log full traceback
        await _set_status(audio_id, {"status": "error", "progress": 0.0, "error": "internal_error"})
        logger.exception(f"Audio {audio_id}: Unexpected error - {e}")
        raise HTTPException(status_code=500, detail="Audio processing failed unexpectedly")

    finally:
        # Always cleanup temp file
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning(f"Failed to cleanup temp file {tmp_path}: {e}")


@router.post("/upload", response_model=AudioUploadResponse)
async def upload_and_separate(
    file: UploadFile,
    audio_cache: CacheManager = Depends(get_audio_cache),
    config: PipelineConfig = Depends(get_config),
    async_mode: bool = Query(False, alias="async"),
) -> AudioUploadResponse:
    """Upload audio file, separate stems, extract features.

    This endpoint:
    1. Validates file type and size
    2. Saves the uploaded file temporarily
    3. Separates into stems using Demucs
    4. Extracts per-stem features (envelope, onsets, centroid)
    5. Caches results with TTL

    Args:
        file: Audio file (mp3, wav, flac, ogg, m4a, aac, wma)

    Returns:
        AudioUploadResponse with audio_id for referencing

    Raises:
        HTTPException 400: Invalid file type or missing filename
        HTTPException 413: File too large
        HTTPException 500: Processing error
    """
    # Validate filename
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate file extension (use config)
    allowed_exts = set(config.audio.allowed_extensions)
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"File type .{ext} not supported. Use: {', '.join(sorted(allowed_exts))}",
        )

    # Validate file size (use config)
    max_size = config.audio.max_upload_mb * 1024 * 1024
    if file.size is not None and file.size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file.size / 1e6:.1f}MB). Max: {config.audio.max_upload_mb}MB",
        )

    # Generate unique ID
    audio_id = str(uuid.uuid4())
    await _set_status(audio_id, {"status": "uploading", "progress": 0.0})

    cache_root = _get_cache_root(config)
    tmp_path: str | None = None
    # Save uploaded file
    suffix = f".{ext}" if ext else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()

        # Double-check size after reading (in case header was wrong)
        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(content) / 1e6:.1f}MB). Max: {config.audio.max_upload_mb}MB",
            )

        tmp.write(content)
        tmp_path = tmp.name

    file_size_mb = len(content) / 1e6
    content_hash = _hash_audio_content(content)
    logger.info(f"Audio {audio_id}: Uploaded {file_size_mb:.1f}MB, hash={content_hash}")

    if async_mode:
        async def _run():
            try:
                await _process_audio_pipeline(
                    audio_id=audio_id,
                    tmp_path=tmp_path,
                    content_hash=content_hash,
                    config=config,
                    audio_cache=audio_cache,
                    cache_root=cache_root,
                )
            except Exception:
                logger.exception(f"Audio {audio_id}: Async processing failed")

        asyncio.create_task(_run())
        return AudioUploadResponse(
            audio_id=audio_id,
            stems=[],
            duration=0.0,
            bpm=0.0,
        )

    return await _process_audio_pipeline(
        audio_id=audio_id,
        tmp_path=tmp_path,
        content_hash=content_hash,
        config=config,
        audio_cache=audio_cache,
        cache_root=cache_root,
    )


@router.get("/status/{audio_id}", response_model=AudioStatusResponse)
async def get_audio_status(
    audio_id: str,
    audio_cache: CacheManager = Depends(get_audio_cache),
) -> AudioStatusResponse:
    """Get processing status for an audio upload.

    Args:
        audio_id: ID returned from upload

    Returns:
        Current processing status
    """
    # Check processing status (thread-safe)
    status = await _get_status(audio_id)
    if status is not None:
        return AudioStatusResponse(
            audio_id=audio_id,
            status=status.get("status", "unknown"),
            progress=status.get("progress", 0.0),
            stems=status.get("stems", []),
            duration=status.get("duration", 0.0),
            bpm=status.get("bpm", 0.0),
        )

    # Check cache
    cached = audio_cache.get(audio_id)
    if cached:
        return AudioStatusResponse(
            audio_id=audio_id,
            status="complete",
            progress=1.0,
            stems=cached.get("stems", []),
            duration=cached.get("duration", 0.0),
            bpm=cached.get("bpm", 120.0),
        )

    raise HTTPException(status_code=404, detail="Audio not found")


@router.delete("/{audio_id}")
async def delete_audio(
    audio_id: str,
    audio_cache: CacheManager = Depends(get_audio_cache),
) -> dict:
    """Delete cached audio data.

    Args:
        audio_id: ID to delete

    Returns:
        Success message
    """
    deleted = audio_cache.delete(audio_id)
    await _delete_status(audio_id)

    if deleted:
        return {"message": f"Audio {audio_id} deleted"}
    raise HTTPException(status_code=404, detail="Audio not found")


class YouTubeDownloadRequest(BaseModel):
    """Request to download audio from YouTube."""

    url: str


class YouTubeDownloadResponse(BaseModel):
    """Response from YouTube download."""

    audio_id: str
    filename: str
    duration: float
    stems: List[str]
    bpm: float = 120.0


@router.post("/youtube", response_model=YouTubeDownloadResponse)
async def download_from_youtube(
    request: YouTubeDownloadRequest,
    audio_cache: CacheManager = Depends(get_audio_cache),
    config: PipelineConfig = Depends(get_config),
    async_mode: bool = Query(False, alias="async"),
) -> YouTubeDownloadResponse:
    """Download audio from YouTube and process it.

    Downloads audio via yt-dlp, then runs the same processing
    pipeline as file uploads (stem separation + feature extraction).

    Args:
        request: YouTube URL and quality settings

    Returns:
        AudioID and metadata for the processed audio

    Raises:
        HTTPException 400: Invalid URL or download failed
        HTTPException 500: Processing error
        HTTPException 501: yt-dlp not installed
    """
    try:
        from hambajuba2ba.audio import download_audio
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="yt-dlp not installed. Install with: uv sync --extra audio",
        )

    audio_id = str(uuid.uuid4())
    await _set_status(audio_id, {"status": "downloading", "progress": 0.0})
    cache_root = _get_cache_root(config)

    async def _run_youtube_pipeline() -> YouTubeDownloadResponse:
        try:
            # Download to temp directory
            loop = asyncio.get_running_loop()

            def _download():
                return download_audio(
                    request.url,
                    output_dir=tempfile.gettempdir(),
                )

            logger.info(f"Audio {audio_id}: Downloading from YouTube: {request.url}")
            tmp_path = await loop.run_in_executor(None, _download)
            filename = os.path.basename(tmp_path)
            logger.info(f"Audio {audio_id}: Downloaded to {tmp_path}")

            # Hash the downloaded content for caching
            with open(tmp_path, "rb") as f:
                content = f.read()
            content_hash = _hash_audio_content(content)
            logger.info(f"Audio {audio_id}: Content hash={content_hash}")

            result = await _process_audio_pipeline(
                audio_id=audio_id,
                tmp_path=tmp_path,
                content_hash=content_hash,
                config=config,
                audio_cache=audio_cache,
                cache_root=cache_root,
                filename=filename,
            )

            return YouTubeDownloadResponse(
                audio_id=result.audio_id,
                filename=filename,
                duration=result.duration,
                stems=result.stems,
                bpm=result.bpm,
            )

        except HTTPException:
            await _set_status(audio_id, {"status": "error", "error": "download_failed"})
            raise
        except Exception:
            await _set_status(audio_id, {"status": "error", "error": "internal_error"})
            raise HTTPException(status_code=500, detail="Processing failed unexpectedly")

    if async_mode:
        async def _run():
            try:
                await _run_youtube_pipeline()
            except Exception:
                logger.exception(f"Audio {audio_id}: Async YouTube processing failed")

        asyncio.create_task(_run())
        return YouTubeDownloadResponse(
            audio_id=audio_id,
            filename=request.url,
            duration=0.0,
            stems=[],
            bpm=0.0,
        )

    return await _run_youtube_pipeline()

# ============================================================================
# Audio File Serving Endpoints
# ============================================================================


@router.get("/{audio_id}/stem/{stem_name}")
async def get_stem_audio(
    audio_id: str,
    stem_name: str,
    audio_cache: CacheManager = Depends(get_audio_cache),
) -> FileResponse:
    """Serve an individual stem audio file.

    Used by the frontend Web Audio API to load stems for mixing.

    Args:
        audio_id: ID from upload/youtube endpoint
        stem_name: One of: bass, drums, vocals, other

    Returns:
        WAV audio file for the requested stem

    Raises:
        HTTPException 404: Audio or stem not found
    """
    cached = audio_cache.get(audio_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Audio not found")

    stem_files = cached.get("stem_files", {})
    if stem_name not in stem_files:
        available = list(stem_files.keys()) if stem_files else cached.get("stems", [])
        raise HTTPException(
            status_code=404,
            detail=f"Stem '{stem_name}' not found. Available: {available}",
        )

    stem_path = stem_files[stem_name]
    if not Path(stem_path).exists():
        raise HTTPException(status_code=404, detail="Stem file no longer exists")

    return FileResponse(
        stem_path,
        media_type="audio/wav",
        filename=f"{stem_name}.wav",
    )


@router.get("/{audio_id}/mix")
async def get_mix_audio(
    audio_id: str,
    audio_cache: CacheManager = Depends(get_audio_cache),
) -> FileResponse:
    """Serve the original mixed audio file.

    Args:
        audio_id: ID from upload/youtube endpoint

    Returns:
        Audio file (original format)

    Raises:
        HTTPException 404: Audio not found
    """
    cached = audio_cache.get(audio_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Audio not found")

    mix_file = cached.get("mix_file")
    if not mix_file:
        raise HTTPException(status_code=404, detail="Mix file not available")

    mix_path = Path(mix_file)
    if not mix_path.exists():
        raise HTTPException(status_code=404, detail="Mix file no longer exists")

    # Determine media type from extension
    ext = mix_path.suffix.lower()
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
    }
    media_type = media_types.get(ext, "audio/mpeg")

    return FileResponse(
        str(mix_path),
        media_type=media_type,
        filename=f"mix{ext}",
    )
