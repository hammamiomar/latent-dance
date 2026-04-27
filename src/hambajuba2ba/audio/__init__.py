"""Audio processing: stem separation, feature extraction, perceptual DSP.

All heavy DSP runs at upload time. Runtime is O(1) array lookup.

Submodules:
    features    - StemAnalyzer (5-phase extraction), StemFeatures dataclass
    perceptual  - Asymmetric envelope, onset strength, spectral flux, brightness
    harmonic    - Tension, tonal distance, roughness, spectral entropy
    pitch       - PESTO/CREPE pitch tracking, normalization
    hpss        - Harmonic-percussive separation (GPU-accelerated)
    virtual_stems - Sub-band stems (drums_low/mid/high, other_mid/high)
    coupling    - Cross-stem: PLV, spectral overlap, call-response
    prominence  - Dancer Ensemble prominence engine (ranking + surprise)
    structure   - Multi-timescale novelty, layer detection
    classification - Component classification for physics/spatial preset selection
    focus_config - BlockLinkConfig, DANCE_MODEL_DEFAULTS
    sampler     - Runtime O(1) sampling from pre-computed arrays
    separator   - Demucs stem separation

Example:
    from hambajuba2ba.audio import StemSeparator, extract_all_features

    stems = await StemSeparator().separate("track.mp3")
    features, cross_stem = extract_all_features(
        stems, sr=44100, fps=60, coupling_stems="physical",
    )
    energy = features["drums"].sample_at_time(t=1.5, channel="energy_smooth")
"""

from .separator import StemSeparator
from .features import StemFeatures, StemAnalyzer, extract_all_features
from .perceptual import (
    EnvelopeConfig,
    DualLayerConfig,
    asymmetric_envelope_follow,
    compute_dual_layer,
    compute_onset_strength,
    detect_peaks,
    normalize_feature,
)
from .virtual_stems import (
    BandConfig,
    VIRTUAL_STEM_BANDS,
    apply_bandpass,
    extract_virtual_stems,
    get_virtual_stem_names,
    get_all_stem_names,
)
from .hpss import (
    compute_normalized_flux,
    compute_energy_db,
    compute_hpss_components,
    HPSSComponents,
)
from .harmonic import (
    compute_roughness_curve,
    compute_spectral_entropy,
    compute_tension,
)
from .pitch import (
    extract_pitch_pesto,
    extract_pitch_polyphonic,
    resample_to_fps,
    normalize_pitch,
)
from .classification import ComponentClassification, classify_component
from .focus_config import (
    FocusConfig,
    BlockLinkConfig,
    derive_focus_config,
    DEFAULT_BLOCK_CONFIGS,
)
from .coupling import (
    compute_plv,
    compute_lock_index,
    compute_onset_synchrony,
    sliding_correlation,
    detect_call_response,
    extract_cross_stem_features,
    CrossStemFeatures,
)
from .prominence import (
    ProminenceEngine,
    SurpriseState,
    RANK_TO_BASE,
    compute_all_prominences,
)
from .structure import (
    compute_multi_timescale_novelty,
    detect_layer_entries,
    estimate_layer_count,
    compute_novelty_derivative,
    compute_structure_features,
)
from .util import align_1d

# Optional: YouTube downloads (requires audio extra)
try:
    from .youtube import download_audio, download_audio_to_temp
except ImportError:
    def download_audio(*args, **kwargs):
        raise ImportError("yt-dlp not installed. Run: uv sync --extra audio")

    def download_audio_to_temp(*args, **kwargs):
        raise ImportError("yt-dlp not installed. Run: uv sync --extra audio")


__all__ = [
    # Stem separation
    "StemSeparator",
    # Feature extraction
    "StemFeatures",
    "StemAnalyzer",
    "extract_all_features",
    # Perceptual DSP
    "EnvelopeConfig",
    "DualLayerConfig",
    "asymmetric_envelope_follow",
    "compute_dual_layer",
    "compute_onset_strength",
    "detect_peaks",
    "normalize_feature",
    # Virtual stems
    "BandConfig",
    "VIRTUAL_STEM_BANDS",
    "apply_bandpass",
    "extract_virtual_stems",
    "get_virtual_stem_names",
    "get_all_stem_names",
    # YouTube
    "download_audio",
    "download_audio_to_temp",
    # HPSS
    "compute_normalized_flux",
    "compute_energy_db",
    "compute_hpss_components",
    "HPSSComponents",
    # Harmonic features
    "compute_roughness_curve",
    "compute_spectral_entropy",
    "compute_tension",
    # Pitch tracking
    "extract_pitch_pesto",
    "extract_pitch_polyphonic",
    "resample_to_fps",
    "normalize_pitch",
    # Classification
    "ComponentClassification",
    "classify_component",
    # Focus config
    "FocusConfig",
    "BlockLinkConfig",
    "derive_focus_config",
    "DEFAULT_BLOCK_CONFIGS",
    # Cross-stem coupling
    "compute_plv",
    "compute_lock_index",
    "compute_onset_synchrony",
    "sliding_correlation",
    "detect_call_response",
    "extract_cross_stem_features",
    "CrossStemFeatures",
    # Prominence engine
    "ProminenceEngine",
    "SurpriseState",
    "RANK_TO_BASE",
    "compute_all_prominences",
    # Structural features
    "compute_multi_timescale_novelty",
    "detect_layer_entries",
    "estimate_layer_count",
    "compute_novelty_derivative",
    "compute_structure_features",
    # Utilities
    "align_1d",
]
