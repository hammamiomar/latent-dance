"""YouTube audio download utility.

Downloads best quality audio from YouTube using yt-dlp.
No transcoding — keeps the original codec (opus/aac) to avoid
ffmpeg issues and quality loss. Demucs/librosa handle these natively.
"""

from __future__ import annotations

import glob
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def download_audio(
    url: str,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Download best quality audio from YouTube URL.

    Skips transcoding entirely — saves in the native container format
    (typically .webm/opus or .m4a/aac). This avoids ffmpeg codec issues
    on minimal containers and preserves maximum quality for Demucs.

    Args:
        url: YouTube URL (video or playlist item)
        output_dir: Directory to save file (default: ./inputs)
        filename: Output filename without extension (default: video title)
        quality: Unused (kept for API compat). Always downloads best quality.

    Returns:
        Path to downloaded audio file

    Raises:
        ImportError: If yt-dlp is not installed
        RuntimeError: If download fails
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp not installed. Install with: uv sync --extra audio"
        )

    # Default to inputs directory
    if output_dir is None:
        output_dir = "./inputs"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build output template
    stem = filename or "%(title)s"
    outtmpl = str(output_path / stem) + ".%(ext)s"

    # Download best audio stream, no postprocessing (no ffmpeg needed)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "noplaylist": True,  # Single video only, even from playlist URLs
    }

    logger.info(f"Downloading audio from: {url}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "audio")

            prepared_path = Path(ydl.prepare_filename(info))
            if prepared_path.exists():
                logger.info(f"Downloaded: {prepared_path}")
                return str(prepared_path)

            for download in info.get("requested_downloads") or []:
                filepath = download.get("filepath")
                if filepath and Path(filepath).exists():
                    logger.info(f"Downloaded: {filepath}")
                    return str(filepath)

            # Find the actual downloaded file (extension varies: webm, m4a, opus, etc.)
            if filename:
                safe_name = filename
            else:
                safe_name = yt_dlp.utils.sanitize_filename(title)

            # Glob for the file with any extension
            pattern = str(output_path / safe_name) + ".*"
            matches = glob.glob(pattern)

            if not matches:
                raise RuntimeError(
                    f"Download succeeded but file not found: {pattern}"
                )

            final_path = max(matches, key=os.path.getmtime)
            logger.info(f"Downloaded: {final_path}")
            return final_path

    except Exception as e:
        raise RuntimeError(f"Failed to download audio: {e}") from e


def download_audio_to_temp(url: str) -> str:
    """Download audio to a temporary directory at best quality.

    Args:
        url: YouTube URL

    Returns:
        Path to temporary audio file (caller should delete when done)
    """
    temp_dir = tempfile.mkdtemp(prefix="hambajuba_")
    return download_audio(url, output_dir=temp_dir)


# CLI entrypoint for quick downloads
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m hambajuba2ba.audio.youtube <url> [output_dir]")
        sys.exit(1)

    url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    logging.basicConfig(level=logging.INFO)
    path = download_audio(url, output_dir)
    print(f"Downloaded: {path}")
