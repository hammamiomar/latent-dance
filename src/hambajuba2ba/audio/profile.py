"""Descriptive song intelligence built from cached audio features.

This module runs outside the live frame path. It summarizes already-computed
audio features and packs selected curves for one-time transfer to the frontend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import struct
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import find_peaks

from .classification import ComponentClassification, classify_component

if TYPE_CHECKING:
    from .coupling import CrossStemFeatures
    from .features import StemFeatures


PHYSICAL_STEMS = ("bass", "drums", "vocals", "other")
HPSS_SUFFIXES = ("harmonic", "percussive")
HPSS_TARGETS = tuple(f"{stem}_{suffix}" for stem in PHYSICAL_STEMS for suffix in HPSS_SUFFIXES)
SUBBAND_TARGETS = ("drums_low", "drums_mid", "drums_high", "other_mid", "other_high")
DERIVED_TARGETS = ("tension", "tonal_distance", "global")
LINK_TARGETS = PHYSICAL_STEMS + HPSS_TARGETS + SUBBAND_TARGETS + DERIVED_TARGETS
TARGET_CURVE_CHANNELS = (
    "envelope",
    "energy_smooth",
    "transient",
    "flux",
    "brightness",
    "sustain",
    "pitch_confidence",
    "pitch_normalized",
    "chroma_centroid",
    "tension",
    "tonal_distance",
    "novelty_long",
)
SECTION_TARGET_CATEGORIES = {
    "primary_energy": "energy_smooth",
    "rhythmic_hits": "transient",
    "texture_motion": "flux",
    "bright_air": "brightness",
    "sustain_body": "sustain",
}

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class StemProfile:
    """Descriptive summary of one audio stem. No recommendations."""

    name: str
    role: str
    texture: str
    frequency_range: str
    mean_energy: float
    hpss_ratio: float
    has_pitch: bool
    has_tension: bool
    onset_density: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SongProfile:
    """Descriptive profile of a full track. No visual recommendations."""

    bpm: float
    estimated_key: str | None
    key_confidence: float
    duration: float
    stems: dict[str, StemProfile]
    coupling: dict[str, float]
    sections: list[float]
    tension_arc: list[dict]
    overall_character: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["stems"] = {name: profile.to_dict() for name, profile in self.stems.items()}
        return payload


@dataclass(frozen=True)
class LinkTargetSignal:
    """Internal normalized signal bundle for one manual link target."""

    name: str
    group: str
    source: str
    curve: np.ndarray
    timestamps: np.ndarray
    duration: float
    channels: dict[str, np.ndarray]
    onsets: np.ndarray


def _finite_mean(values: np.ndarray | None, default: float = 0.0) -> float:
    if values is None or len(values) == 0:
        return default
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return default
    return float(np.clip(np.mean(finite), 0.0, 1.0))


def _finite_variance(values: np.ndarray | None) -> float:
    if values is None or len(values) == 0:
        return 0.0
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.var(finite))


def _safe_float(value: float | int | None, default: float = 0.0) -> float:
    if value is None:
        return default
    result = float(value)
    if not np.isfinite(result):
        return default
    return result


def _classification_for(features: "StemFeatures") -> ComponentClassification:
    if features.hpss_ratio is None:
        return ComponentClassification(
            percussive_confidence=0.5,
            harmonic_confidence=0.5,
            rhythm_confidence=0.0,
            melody_confidence=0.0,
            harmony_confidence=0.0,
            texture_confidence=0.0,
            is_bass=False,
            is_mid=True,
            is_high=False,
        )
    return classify_component(features)


def _profile_role(features: "StemFeatures", classification: ComponentClassification) -> str:
    hpss_ratio = _safe_float(features.hpss_ratio, 0.5)
    onset_density = len(features.onsets) / max(features.duration, 0.1)
    has_pitch = _finite_mean(features.pitch_confidence, 0.0) > 0.3

    if hpss_ratio > 0.65 and onset_density >= 0.5:
        return "percussive"
    if has_pitch or classification.melody_confidence >= 0.3:
        return "melodic"
    if classification.harmony_confidence >= classification.texture_confidence:
        return "harmonic"
    return "textural"


def _profile_texture(features: "StemFeatures") -> str:
    hpss_ratio = _safe_float(features.hpss_ratio, 0.5)
    flatness = _finite_mean(features.flatness, 0.0)

    if flatness > 0.55:
        return "noise"
    if hpss_ratio > 0.65:
        return "percussive"
    if hpss_ratio < 0.35:
        return "harmonic"
    return "mixed"


def _profile_frequency_range(classification: ComponentClassification) -> str:
    if classification.is_bass:
        return "bass"
    if classification.is_high:
        return "high"
    return "mid"


def build_stem_profile(
    name: str,
    features: "StemFeatures",
    classification: ComponentClassification | None = None,
) -> StemProfile:
    """Build a descriptive profile for one physical stem."""

    resolved = classification or _classification_for(features)
    hpss_ratio = float(np.clip(_safe_float(features.hpss_ratio, 0.5), 0.0, 1.0))
    onset_density = len(features.onsets) / max(features.duration, 0.1)

    return StemProfile(
        name=name,
        role=_profile_role(features, resolved),
        texture=_profile_texture(features),
        frequency_range=_profile_frequency_range(resolved),
        mean_energy=_finite_mean(features.energy_smooth, 0.0),
        hpss_ratio=hpss_ratio,
        has_pitch=_finite_mean(features.pitch_confidence, 0.0) > 0.3,
        has_tension=features.tension is not None and _finite_variance(features.tension) > 1e-4,
        onset_density=float(max(0.0, onset_density)),
    )


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def estimate_key(features: dict[str, "StemFeatures"]) -> tuple[str | None, float]:
    """Estimate track key from average harmonic-stem chroma."""

    chroma_columns: list[np.ndarray] = []
    for stem_features in features.values():
        if stem_features.chroma is None:
            continue
        if _safe_float(stem_features.hpss_ratio, 0.5) >= 0.7:
            continue
        chroma = np.asarray(stem_features.chroma, dtype=np.float32)
        if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] == 0:
            continue
        mean_chroma = np.mean(chroma, axis=1)
        if np.isfinite(mean_chroma).all() and float(np.sum(mean_chroma)) > 1e-6:
            chroma_columns.append(mean_chroma)

    if not chroma_columns:
        return None, 0.0

    pooled = np.mean(np.stack(chroma_columns, axis=0), axis=0)
    if not np.isfinite(pooled).all() or float(np.sum(pooled)) <= 1e-6:
        return None, 0.0

    best_key: str | None = None
    best_corr = -1.0
    for root, note in enumerate(NOTE_NAMES):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            corr = _correlation(pooled, np.roll(profile, root))
            if corr > best_corr:
                best_corr = corr
                best_key = f"{note} {mode}"

    return best_key, float(np.clip(best_corr, 0.0, 1.0))


def detect_sections(novelty_long: np.ndarray, timestamps: np.ndarray) -> list[float]:
    """Find section boundaries from a long-timescale novelty curve."""

    novelty = np.nan_to_num(
        np.asarray(novelty_long, dtype=np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    times = np.asarray(timestamps, dtype=np.float32)
    if novelty.size == 0 or times.size == 0:
        return [0.0]

    n = min(novelty.size, times.size)
    novelty = novelty[:n]
    times = times[:n]

    if n < 3 or float(np.nanstd(novelty)) <= 1e-6:
        return [0.0]

    frame_dt = float(np.median(np.diff(times))) if n > 1 else 1.0
    if not np.isfinite(frame_dt) or frame_dt <= 0:
        frame_dt = 1.0
    distance = max(1, int(round(8.0 / frame_dt)))
    height = float(np.nanmean(novelty) + 0.5 * np.nanstd(novelty))
    peaks, _ = find_peaks(novelty, height=height, distance=distance, prominence=0.1)

    boundaries = [0.0]
    for peak in peaks:
        t = float(times[int(peak)])
        if t > 0.25:
            boundaries.append(round(t, 3))
    return sorted(set(boundaries))


def summarize_tension_arc(
    tension: np.ndarray,
    timestamps: np.ndarray,
    n_points: int = 10,
) -> list[dict]:
    """Sample tension at sparse timestamps and annotate coarse trend."""

    tension_arr = np.asarray(tension, dtype=np.float32)
    times = np.asarray(timestamps, dtype=np.float32)
    if tension_arr.size == 0 or times.size == 0:
        return []

    n = min(tension_arr.size, times.size)
    tension_arr = tension_arr[:n]
    times = times[:n]
    if n == 0:
        return []

    count = max(1, min(n_points, n))
    sample_times = np.linspace(float(times[0]), float(times[-1]), count)
    sample_values = np.interp(sample_times, times, tension_arr)
    result: list[dict] = []

    previous = float(sample_values[0])
    for idx, (t, value) in enumerate(zip(sample_times, sample_values, strict=True)):
        current = float(np.clip(value, 0.0, 1.0))
        if idx == 0:
            trend = "stable"
        else:
            delta = current - previous
            if delta > 0.05:
                trend = "rising"
            elif delta < -0.05:
                trend = "falling"
            else:
                trend = "stable"
        result.append({"time": round(float(t), 3), "tension": current, "trend": trend})
        previous = current
    return result


def _reference_timestamps(features: dict[str, "StemFeatures"]) -> np.ndarray:
    for stem in PHYSICAL_STEMS:
        stem_features = features.get(stem)
        if stem_features is not None and len(stem_features.timestamps) > 0:
            return np.asarray(stem_features.timestamps, dtype=np.float32)
    for stem_features in features.values():
        if len(stem_features.timestamps) > 0:
            return np.asarray(stem_features.timestamps, dtype=np.float32)
    return np.zeros(0, dtype=np.float32)


def _resample_curve(
    values: np.ndarray | None,
    source_timestamps: np.ndarray,
    reference_timestamps: np.ndarray,
) -> np.ndarray | None:
    if values is None or len(values) == 0 or reference_timestamps.size == 0:
        return None

    source = np.asarray(source_timestamps, dtype=np.float32)
    curve = np.asarray(values, dtype=np.float32)
    n = min(curve.size, source.size)
    if n == 0:
        return None
    source = source[:n]
    curve = curve[:n]

    if n == reference_timestamps.size and np.allclose(source, reference_timestamps[:n]):
        return np.nan_to_num(curve, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    resampled = np.interp(reference_timestamps, source, curve, left=curve[0], right=curve[-1])
    return np.nan_to_num(resampled, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def _aggregate_feature_curve(
    features: dict[str, "StemFeatures"],
    channel: str,
    reference_timestamps: np.ndarray,
) -> np.ndarray:
    curves: list[np.ndarray] = []
    for stem_features in features.values():
        values = getattr(stem_features, channel, None)
        resampled = _resample_curve(values, stem_features.timestamps, reference_timestamps)
        if resampled is not None:
            curves.append(resampled)

    if not curves:
        return np.zeros_like(reference_timestamps, dtype=np.float32)

    return np.clip(np.mean(np.stack(curves, axis=0), axis=0), 0.0, 1.0).astype(np.float32)


def _target_curve_name(target: str, channel: str) -> str:
    return f"target:{target}:{channel}"


def _parse_target_curve_name(name: str) -> tuple[str, str] | None:
    if not name.startswith("target:"):
        return None
    parts = name.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def _target_curve_catalog(curves: dict[str, np.ndarray]) -> dict[str, list[str]]:
    catalog: dict[str, list[str]] = {}
    for name in curves:
        parsed = _parse_target_curve_name(name)
        if parsed is None:
            continue
        target, channel = parsed
        catalog.setdefault(target, []).append(channel)
    return {target: sorted(channels) for target, channels in sorted(catalog.items())}


def _pair_curve_timestamps(
    features: dict[str, "StemFeatures"],
    pair: tuple[str, str],
    length: int,
) -> np.ndarray:
    first = features.get(pair[0])
    if first is not None and len(first.timestamps) == length:
        return np.asarray(first.timestamps, dtype=np.float32)
    second = features.get(pair[1])
    if second is not None and len(second.timestamps) == length:
        return np.asarray(second.timestamps, dtype=np.float32)
    return _reference_timestamps(features)


def build_song_curves(
    features: dict[str, "StemFeatures"],
    cross_stem_features: "CrossStemFeatures | None" = None,
) -> dict[str, np.ndarray]:
    """Build full frontend-local song intelligence curves."""

    reference = _reference_timestamps(features)
    curves: dict[str, np.ndarray] = {"timestamps": reference.astype(np.float32)}
    curves["tension"] = _aggregate_feature_curve(features, "tension", reference)
    curves["tonal_distance"] = _aggregate_feature_curve(features, "tonal_distance", reference)
    curves["novelty_long"] = _aggregate_feature_curve(features, "novelty_long", reference)

    duration = float(reference[-1]) if reference.size else 0.0
    signals = _collect_link_target_signals(features, curves, duration=duration)
    for target, signal in sorted(signals.items()):
        for channel in TARGET_CURVE_CHANNELS:
            values = signal.channels.get(channel)
            resampled = _resample_curve(values, signal.timestamps, reference)
            if resampled is not None:
                curves[_target_curve_name(target, channel)] = np.clip(resampled, 0.0, 1.0)

    if cross_stem_features is not None:
        for pair, values in sorted(cross_stem_features.lock_index.items()):
            if pair[0] not in PHYSICAL_STEMS or pair[1] not in PHYSICAL_STEMS:
                continue
            source_timestamps = _pair_curve_timestamps(features, pair, len(values))
            resampled = _resample_curve(values, source_timestamps, reference)
            if resampled is not None:
                curves[f"lock_index:{pair[0]}-{pair[1]}"] = np.clip(resampled, 0.0, 1.0)

    return curves


def _mean_lock_index(cross_stem_features: "CrossStemFeatures | None") -> dict[str, float]:
    if cross_stem_features is None:
        return {}

    coupling: dict[str, float] = {}
    for pair, values in sorted(cross_stem_features.lock_index.items()):
        if pair[0] not in PHYSICAL_STEMS or pair[1] not in PHYSICAL_STEMS:
            continue
        coupling[f"{pair[0]}-{pair[1]}"] = _finite_mean(values, 0.0)
    return coupling


def classify_overall_character(stems: dict[str, StemProfile]) -> str:
    """Return a compact descriptive whole-song character label."""

    if not stems:
        return "unknown"

    profiles = list(stems.values())
    mean_hpss = sum(p.hpss_ratio for p in profiles) / len(profiles)
    mean_onsets = sum(p.onset_density for p in profiles) / len(profiles)
    pitched = sum(1 for p in profiles if p.has_pitch)
    tense = sum(1 for p in profiles if p.has_tension)

    if mean_hpss > 0.6 and mean_onsets > 1.5:
        return "percussive_rhythmic"
    if pitched >= 2:
        return "melodic_harmonic"
    if tense >= 2:
        return "tense_harmonic"
    if mean_onsets < 0.5:
        return "sparse_textural"
    return "mixed"


def build_song_profile(
    features: dict[str, "StemFeatures"],
    cross_stem_features: "CrossStemFeatures | None",
    *,
    bpm: float,
    duration: float,
    classifications: dict[str, ComponentClassification | None] | None = None,
    curves: dict[str, np.ndarray] | None = None,
) -> SongProfile:
    """Build a descriptive song profile from cached audio features."""

    stem_profiles: dict[str, StemProfile] = {}
    for stem in PHYSICAL_STEMS:
        stem_features = features.get(stem)
        if stem_features is None:
            continue
        classification = classifications.get(stem) if classifications else None
        stem_profiles[stem] = build_stem_profile(stem, stem_features, classification)

    resolved_curves = curves if curves is not None else build_song_curves(features, cross_stem_features)
    timestamps = resolved_curves.get("timestamps", np.zeros(0, dtype=np.float32))
    novelty_long = resolved_curves.get("novelty_long", np.zeros_like(timestamps))
    tension = resolved_curves.get("tension", np.zeros_like(timestamps))
    estimated_key, key_confidence = estimate_key(features)

    return SongProfile(
        bpm=_safe_float(bpm, 120.0),
        estimated_key=estimated_key,
        key_confidence=key_confidence,
        duration=_safe_float(duration, 0.0),
        stems=stem_profiles,
        coupling=_mean_lock_index(cross_stem_features),
        sections=detect_sections(novelty_long, timestamps),
        tension_arc=summarize_tension_arc(tension, timestamps),
        overall_character=classify_overall_character(stem_profiles),
    )


def _clean_1d(values: np.ndarray | None) -> np.ndarray:
    if values is None or len(values) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.clip(
        np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float32)


def _curve_delta(values: np.ndarray) -> np.ndarray:
    curve = _clean_1d(values)
    if curve.size == 0:
        return curve
    return np.clip(np.abs(np.diff(curve, prepend=curve[0])), 0.0, 1.0).astype(np.float32)


def _percentile(values: np.ndarray, q: float) -> float:
    clean = _clean_1d(values)
    if clean.size == 0:
        return 0.0
    return float(np.percentile(clean, q))


def _dynamic_range(values: np.ndarray) -> float:
    clean = _clean_1d(values)
    if clean.size == 0:
        return 0.0
    return float(np.clip(np.percentile(clean, 90) - np.percentile(clean, 10), 0.0, 1.0))


def _curve_score(values: np.ndarray | None) -> float:
    clean = _clean_1d(values)
    if clean.size == 0:
        return 0.0
    return float(np.clip(0.5 * np.mean(clean) + 0.5 * np.percentile(clean, 90), 0.0, 1.0))


def _round(value: float | int | None, digits: int = 6) -> float:
    return round(_safe_float(value, 0.0), digits)


def _feature_channels(stem_features: "StemFeatures") -> dict[str, np.ndarray]:
    channels: dict[str, np.ndarray] = {}
    for channel in (
        "envelope",
        "energy_smooth",
        "transient",
        "flux",
        "brightness",
        "flatness",
        "flash",
        "sustain",
        "pitch_confidence",
        "pitch_normalized",
        "chroma_centroid",
        "tension",
        "tonal_distance",
        "novelty_long",
    ):
        values = getattr(stem_features, channel, None)
        if values is not None:
            channels[channel] = _clean_1d(values)
    return channels


def _stem_signal(name: str, stem_features: "StemFeatures", group: str) -> LinkTargetSignal:
    curve = _clean_1d(stem_features.energy_smooth)
    return LinkTargetSignal(
        name=name,
        group=group,
        source=name,
        curve=curve,
        timestamps=np.asarray(stem_features.timestamps, dtype=np.float32),
        duration=_safe_float(stem_features.duration, 0.0),
        channels=_feature_channels(stem_features),
        onsets=np.asarray(stem_features.onsets, dtype=np.float32),
    )


def _hpss_signal(name: str, features: dict[str, "StemFeatures"]) -> LinkTargetSignal | None:
    parent_name, suffix = name.rsplit("_", 1)
    parent = features.get(parent_name)
    if parent is None:
        return None

    channel_name = f"{suffix}_energy"
    primary = getattr(parent, channel_name, None)
    if primary is None:
        ratio = float(np.clip(_safe_float(parent.hpss_ratio, 0.5), 0.0, 1.0))
        weight = 1.0 - ratio if suffix == "harmonic" else ratio
        primary = _clean_1d(parent.energy_smooth) * weight

    curve = _clean_1d(primary)
    channels = _feature_channels(parent)
    channels["envelope"] = curve
    channels["energy_smooth"] = curve
    if suffix == "harmonic":
        channels["transient"] = _clean_1d(parent.transient) * 0.35
        channels["flux"] = _clean_1d(parent.flux) * 0.6
        channels["sustain"] = np.maximum(_clean_1d(parent.sustain), curve)
        onsets = np.zeros(0, dtype=np.float32)
    else:
        channels["transient"] = np.maximum(_clean_1d(parent.transient), _clean_1d(parent.flash))
        channels["flux"] = np.maximum(_clean_1d(parent.flux), _curve_delta(curve))
        channels["sustain"] = _clean_1d(parent.sustain) * 0.5
        onsets = np.asarray(parent.onsets, dtype=np.float32)

    return LinkTargetSignal(
        name=name,
        group="hpss",
        source=f"{parent_name}.{suffix}",
        curve=curve,
        timestamps=np.asarray(parent.timestamps, dtype=np.float32),
        duration=_safe_float(parent.duration, 0.0),
        channels=channels,
        onsets=onsets,
    )


def _derived_signal(
    name: str,
    curve: np.ndarray,
    timestamps: np.ndarray,
    *,
    duration: float,
) -> LinkTargetSignal:
    clean = _clean_1d(curve)
    delta = _curve_delta(clean)
    return LinkTargetSignal(
        name=name,
        group="derived",
        source=name,
        curve=clean,
        timestamps=np.asarray(timestamps, dtype=np.float32),
        duration=duration,
        channels={
            "envelope": clean,
            "energy_smooth": clean,
            "transient": delta,
            "flux": delta,
            "brightness": clean if name == "tonal_distance" else np.zeros_like(clean),
            "sustain": clean,
            "tension": clean if name == "tension" else np.zeros_like(clean),
            "tonal_distance": clean if name == "tonal_distance" else np.zeros_like(clean),
            "novelty_long": clean,
        },
        onsets=np.zeros(0, dtype=np.float32),
    )


def _collect_link_target_signals(
    features: dict[str, "StemFeatures"],
    curves: dict[str, np.ndarray],
    *,
    duration: float,
) -> dict[str, LinkTargetSignal]:
    signals: dict[str, LinkTargetSignal] = {}

    for name in PHYSICAL_STEMS:
        stem_features = features.get(name)
        if stem_features is not None:
            signals[name] = _stem_signal(name, stem_features, "physical")

    for name in HPSS_TARGETS:
        signal = _hpss_signal(name, features)
        if signal is not None:
            signals[name] = signal

    for name in SUBBAND_TARGETS:
        stem_features = features.get(name)
        if stem_features is not None:
            signals[name] = _stem_signal(name, stem_features, "subband")

    timestamps = np.asarray(curves.get("timestamps", _reference_timestamps(features)), dtype=np.float32)
    if timestamps.size == 0:
        return signals

    for name in ("tension", "tonal_distance"):
        curve = curves.get(name)
        if curve is not None:
            signals[name] = _derived_signal(name, curve, timestamps, duration=duration)

    physical = {name: features[name] for name in PHYSICAL_STEMS if name in features}
    if physical:
        global_curve = _aggregate_feature_curve(physical, "energy_smooth", timestamps)
        signals["global"] = _derived_signal("global", global_curve, timestamps, duration=duration)

    return signals


def _correlation_metadata(
    signals: dict[str, LinkTargetSignal],
) -> dict[str, tuple[float, list[dict[str, float | str]]]]:
    reference = next(
        (signal.timestamps for signal in signals.values() if signal.timestamps.size > 0),
        np.zeros(0, dtype=np.float32),
    )
    if reference.size == 0:
        return {name: (0.0, []) for name in signals}

    resampled: dict[str, np.ndarray] = {}
    for name, signal in signals.items():
        curve = _resample_curve(signal.curve, signal.timestamps, reference)
        if curve is not None:
            resampled[name] = curve

    metadata: dict[str, tuple[float, list[dict[str, float | str]]]] = {}
    for name, curve in resampled.items():
        pairs: list[tuple[str, float]] = []
        for other_name, other_curve in resampled.items():
            if name == other_name:
                continue
            corr = _correlation(curve, other_curve)
            if abs(corr) >= 0.45:
                pairs.append((other_name, corr))
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        max_abs = abs(pairs[0][1]) if pairs else 0.0
        metadata[name] = (
            float(np.clip(1.0 - max_abs, 0.0, 1.0)),
            [
                {"target": other_name, "correlation": _round(corr, 4)}
                for other_name, corr in pairs[:3]
            ],
        )

    return metadata


def _movement_words(
    *,
    active_ratio: float,
    dynamic_range: float,
    transient_score: float,
    flux_score: float,
    brightness_score: float,
    sustain_score: float,
    onset_density: float,
    pitch_confidence: float,
    chroma_motion: float,
    uniqueness_score: float,
) -> list[str]:
    words: list[str] = []
    if transient_score >= 0.45:
        words.append("punchy")
    if onset_density >= 1.4:
        words.append("busy")
    if active_ratio <= 0.25:
        words.append("sparse")
    if sustain_score >= 0.55:
        words.append("sustained")
    if dynamic_range >= 0.35:
        words.append("wide_swing")
    if flux_score >= 0.45:
        words.append("changing")
    if brightness_score >= 0.6:
        words.append("bright")
    if pitch_confidence >= 0.3:
        words.append("pitched")
    if chroma_motion >= 0.2:
        words.append("pitch_moving")
    if uniqueness_score >= 0.6:
        words.append("distinct")
    return words or ["steady"]


def _position_affordances(
    *,
    target: str,
    pitch_confidence: float,
    chroma_motion: float,
    brightness_dynamic: float,
    tension_variance: float,
) -> list[str]:
    sources: list[str] = []
    if pitch_confidence >= 0.3:
        sources.append("pitch")
    if chroma_motion >= 0.18:
        sources.append("chroma")
    if brightness_dynamic >= 0.18:
        sources.append("brightness")
    if tension_variance >= 0.01 or target == "tension":
        sources.append("tension")
    if target == "tension":
        sources.append("tension_global")
    return sources or ["auto"]


def _good_for(
    *,
    target: str,
    transient_score: float,
    flux_score: float,
    brightness_score: float,
    sustain_score: float,
    dynamic_range: float,
    uniqueness_score: float,
    pitch_confidence: float,
    chroma_motion: float,
    tension_variance: float,
) -> list[str]:
    uses: list[str] = []
    if transient_score >= 0.4:
        uses.extend(["rhythmic_hits", "detail_punctuation"])
    if sustain_score >= 0.5:
        uses.append("continuous_body")
    if flux_score >= 0.4:
        uses.append("texture_motion")
    if brightness_score >= 0.55 or target.endswith("_high"):
        uses.append("bright_air")
    if dynamic_range >= 0.3 and uniqueness_score >= 0.45:
        uses.append("primary_driver")
    if pitch_confidence >= 0.3 or chroma_motion >= 0.18:
        uses.append("prompt_position")
    if tension_variance >= 0.01 or target in {"tension", "tonal_distance"}:
        uses.append("section_or_harmony_arc")
    return uses or ["supporting_driver"]


def _preferred_intensity_source(
    *,
    transient_score: float,
    flux_score: float,
    dynamic_range: float,
) -> str:
    if transient_score >= 0.45:
        return "transient"
    if flux_score >= 0.4:
        return "flux"
    if dynamic_range >= 0.2:
        return "energy_smooth"
    return "envelope"


def _target_profile(
    signal: LinkTargetSignal,
    *,
    uniqueness_score: float,
    coupled_targets: list[dict[str, float | str]],
) -> dict:
    curve = _clean_1d(signal.curve)
    active_threshold = max(0.08, _percentile(curve, 50) + 0.1 * float(np.std(curve)))
    active_ratio = float(np.mean(curve > active_threshold)) if curve.size else 0.0
    onset_density = len(signal.onsets) / max(signal.duration, 0.1)

    mean_energy = float(np.mean(curve)) if curve.size else 0.0
    peak_energy = float(np.max(curve)) if curve.size else 0.0
    dynamic_range = _dynamic_range(curve)
    if peak_energy < 0.05 and dynamic_range < 0.03:
        uniqueness_score = 0.0
    elif dynamic_range < 0.03:
        uniqueness_score *= 0.3
    transient_score = max(_curve_score(signal.channels.get("transient")), min(onset_density / 3.0, 1.0))
    flux_score = _curve_score(signal.channels.get("flux"))
    brightness_score = _curve_score(signal.channels.get("brightness"))
    sustain_score = _curve_score(signal.channels.get("sustain"))
    pitch_confidence = _finite_mean(signal.channels.get("pitch_confidence"), 0.0)
    chroma_motion = _dynamic_range(signal.channels.get("chroma_centroid", np.zeros(0, dtype=np.float32)))
    brightness_dynamic = _dynamic_range(signal.channels.get("brightness", np.zeros(0, dtype=np.float32)))
    tension_variance = _finite_variance(signal.channels.get("tension"))

    words = _movement_words(
        active_ratio=active_ratio,
        dynamic_range=dynamic_range,
        transient_score=transient_score,
        flux_score=flux_score,
        brightness_score=brightness_score,
        sustain_score=sustain_score,
        onset_density=onset_density,
        pitch_confidence=pitch_confidence,
        chroma_motion=chroma_motion,
        uniqueness_score=uniqueness_score,
    )
    good_for = _good_for(
        target=signal.name,
        transient_score=transient_score,
        flux_score=flux_score,
        brightness_score=brightness_score,
        sustain_score=sustain_score,
        dynamic_range=dynamic_range,
        uniqueness_score=uniqueness_score,
        pitch_confidence=pitch_confidence,
        chroma_motion=chroma_motion,
        tension_variance=tension_variance,
    )

    return {
        "target": signal.name,
        "group": signal.group,
        "source": signal.source,
        "stats": {
            "mean_energy": _round(mean_energy),
            "peak_energy": _round(peak_energy),
            "dynamic_range": _round(dynamic_range),
            "active_ratio": _round(active_ratio),
            "onset_density": _round(onset_density),
            "transient_score": _round(transient_score),
            "flux_score": _round(flux_score),
            "brightness_score": _round(brightness_score),
            "sustain_score": _round(sustain_score),
            "pitch_confidence": _round(pitch_confidence),
            "chroma_motion": _round(chroma_motion),
            "tension_variance": _round(tension_variance),
            "uniqueness_score": _round(uniqueness_score),
        },
        "movement_words": words,
        "good_for": good_for,
        "preferred_intensity_source": _preferred_intensity_source(
            transient_score=transient_score,
            flux_score=flux_score,
            dynamic_range=dynamic_range,
        ),
        "position_source_affordances": _position_affordances(
            target=signal.name,
            pitch_confidence=pitch_confidence,
            chroma_motion=chroma_motion,
            brightness_dynamic=brightness_dynamic,
            tension_variance=tension_variance,
        ),
        "coupled_targets": coupled_targets,
    }


def _ranked_drivers(profiles: dict[str, dict]) -> dict[str, list[dict[str, float | str | list[str]]]]:
    categories = {
        "primary_driver": lambda s, high: (
            0.25 * s["mean_energy"]
            + 0.25 * s["dynamic_range"]
            + 0.25 * s["uniqueness_score"]
            + 0.15 * s["sustain_score"]
            + 0.10 * s["active_ratio"]
        ),
        "rhythmic_hits": lambda s, high: (
            0.55 * s["transient_score"]
            + 0.20 * s["dynamic_range"]
            + 0.15 * s["uniqueness_score"]
            + 0.10 * s["flux_score"]
        ),
        "texture_motion": lambda s, high: (
            0.45 * s["flux_score"]
            + 0.25 * s["brightness_score"]
            + 0.15 * s["dynamic_range"]
            + 0.15 * s["uniqueness_score"]
        ),
        "prompt_position": lambda s, high: (
            0.35 * s["pitch_confidence"]
            + 0.25 * s["chroma_motion"]
            + 0.20 * s["brightness_score"]
            + 0.15 * min(s["tension_variance"] * 12.0, 1.0)
            + 0.05 * s["uniqueness_score"]
        ),
        "slow_arc": lambda s, high: (
            0.35 * min(s["tension_variance"] * 12.0, 1.0)
            + 0.25 * s["sustain_score"]
            + 0.25 * s["dynamic_range"]
            + 0.15 * s["uniqueness_score"]
        ),
        "bright_air": lambda s, high: (
            0.40 * s["brightness_score"]
            + 0.30 * s["flux_score"]
            + 0.20 * s["transient_score"]
            + 0.10 * high
        ),
    }

    ranked: dict[str, list[dict[str, float | str | list[str]]]] = {}
    for category, score_fn in categories.items():
        scored: list[dict[str, float | str | list[str]]] = []
        for target, profile in profiles.items():
            stats = profile["stats"]
            high = 1.0 if target.endswith("_high") or stats["brightness_score"] >= 0.6 else 0.0
            score = float(np.clip(score_fn(stats, high), 0.0, 1.0))
            if score <= 0.03:
                continue
            scored.append(
                {
                    "target": target,
                    "score": _round(score),
                    "reasons": list(profile["movement_words"][:3]),
                }
            )
        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        ranked[category] = scored[:6]
    return ranked


def _window_curve_stats(
    values: np.ndarray | None,
    timestamps: np.ndarray,
    start: float,
    end: float,
) -> dict[str, float]:
    clean = _clean_1d(values)
    if clean.size == 0 or timestamps.size == 0:
        return {"mean": 0.0, "max": 0.0, "delta": 0.0}

    n = min(clean.size, timestamps.size)
    curve = clean[:n]
    times = timestamps[:n]
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        value = float(np.interp((start + end) * 0.5, times, curve, left=curve[0], right=curve[-1]))
        return {"mean": value, "max": value, "delta": 0.0}

    window = curve[mask]
    first = float(window[0])
    last = float(window[-1])
    return {
        "mean": float(np.mean(window)),
        "max": float(np.max(window)),
        "delta": last - first,
    }


def _rank_section_targets(
    curves: dict[str, np.ndarray],
    timestamps: np.ndarray,
    start: float,
    end: float,
    *,
    limit: int = 5,
) -> dict[str, list[dict[str, float | str]]]:
    catalog = _target_curve_catalog(curves)
    ranked: dict[str, list[dict[str, float | str]]] = {}
    for category, channel in SECTION_TARGET_CATEGORIES.items():
        scored: list[dict[str, float | str]] = []
        for target, channels in catalog.items():
            if channel not in channels:
                continue
            stats = _window_curve_stats(
                curves.get(_target_curve_name(target, channel)),
                timestamps,
                start,
                end,
            )
            score = float(np.clip(0.65 * stats["max"] + 0.35 * stats["mean"], 0.0, 1.0))
            if score <= 0.03:
                continue
            scored.append(
                {
                    "target": target,
                    "channel": channel,
                    "score": _round(score),
                    "mean": _round(stats["mean"]),
                    "max": _round(stats["max"]),
                    "delta": _round(stats["delta"]),
                }
            )
        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        ranked[category] = scored[:limit]
    return ranked


def _section_target_summary(
    curves: dict[str, np.ndarray],
    section_boundaries: list[float],
    duration: float,
) -> list[dict]:
    timestamps = np.asarray(curves.get("timestamps", np.zeros(0, dtype=np.float32)), dtype=np.float32)
    if timestamps.size == 0 or not section_boundaries:
        return []

    boundaries = sorted({float(max(0.0, boundary)) for boundary in section_boundaries})
    if not boundaries or boundaries[0] != 0.0:
        boundaries.insert(0, 0.0)
    if duration > boundaries[-1]:
        boundaries.append(float(duration))

    summaries: list[dict] = []
    for index, start in enumerate(boundaries[:-1]):
        end = boundaries[index + 1]
        if end <= start:
            continue
        summaries.append(
            {
                "index": index,
                "start": _round(start, 3),
                "end": _round(end, 3),
                "ranked_targets": _rank_section_targets(curves, timestamps, start, end),
            }
        )
    return summaries


def build_song_analysis(
    features: dict[str, "StemFeatures"],
    cross_stem_features: "CrossStemFeatures | None",
    *,
    bpm: float,
    duration: float,
    curves: dict[str, np.ndarray] | None = None,
    metadata: dict[str, str] | None = None,
) -> dict:
    """Build whole-song DSP evidence for agent entry planning.

    Filename metadata is included when available. DSP remains the primary truth
    for visual control decisions.
    """

    resolved_curves = curves if curves is not None else build_song_curves(features, cross_stem_features)
    signals = _collect_link_target_signals(
        features,
        resolved_curves,
        duration=_safe_float(duration, 0.0),
    )
    correlation_metadata = _correlation_metadata(signals)
    profiles = {
        target: _target_profile(
            signal,
            uniqueness_score=correlation_metadata.get(target, (0.0, []))[0],
            coupled_targets=correlation_metadata.get(target, (0.0, []))[1],
        )
        for target, signal in signals.items()
    }

    timestamps = resolved_curves.get("timestamps", np.zeros(0, dtype=np.float32))
    novelty_long = resolved_curves.get("novelty_long", np.zeros_like(timestamps))
    tension = resolved_curves.get("tension", np.zeros_like(timestamps))
    section_boundaries = detect_sections(novelty_long, timestamps)
    curve_catalog = _target_curve_catalog(resolved_curves)

    clean_metadata = {
        key: value
        for key, value in (metadata or {}).items()
        if isinstance(key, str) and isinstance(value, str) and value.strip()
    }
    has_metadata = bool(clean_metadata)

    return {
        "version": "hamba-song-analysis/v1",
        "anonymous": not has_metadata,
        "metadata_policy": (
            "filename included when available; reason primarily from DSP and user directives"
            if has_metadata
            else "metadata unavailable; reason from DSP and user directives only"
        ),
        "metadata": clean_metadata,
        "duration": _round(duration),
        "bpm": _round(bpm),
        "target_count": len(profiles),
        "link_targets": profiles,
        "curve_catalog": {
            "format": "target:<link_target>:<channel>",
            "target_count": len(curve_catalog),
            "targets": curve_catalog,
        },
        "ranked_drivers": _ranked_drivers(profiles),
        "structure": {
            "section_boundaries": section_boundaries,
            "tension_arc": summarize_tension_arc(tension, timestamps),
            "section_target_summary": _section_target_summary(
                resolved_curves,
                section_boundaries,
                duration,
            ),
        },
        "entry_planning": {
            "read_this_before_first_plan": [
                "Start from prompts and composition/noise for the global story and scene movement.",
                "Use ranked_drivers to choose which link targets deserve prompt, latent, and SAE control.",
                "Use curve_catalog and section_target_summary to avoid over-indexing on tension when stems or subbands carry the local motion.",
                "Use sae_rank 1 for Hermes-authored enabled SAE blocks; make support/subtlety with strength ranges, masks, curves, and link targets instead of lower ranks.",
                "Treat stage bounds as gain staging: narrower ranges reduce a layer without changing its target.",
            ],
            "stable_tool_order": [
                "hamba_get_state",
                "hamba_get_control_surface",
                "hamba_get_song_analysis",
                "hamba_get_music_window",
                "hamba_search_features",
                "hamba_apply_visual_plan",
            ],
        },
    }


def _curve_sort_key(name: str) -> tuple[int, str]:
    if name == "timestamps":
        return (0, name)
    if name in {"tension", "tonal_distance", "novelty_long"}:
        return (1, name)
    return (2, name)


def pack_song_curves_binary(curves: dict[str, np.ndarray]) -> bytes:
    """Pack named float32 curves into the frontend transfer format."""

    chunks = [struct.pack("<I", len(curves))]
    for name in sorted(curves, key=_curve_sort_key):
        encoded = name.encode("utf-8")
        values = np.asarray(curves[name], dtype="<f4")
        values = np.ascontiguousarray(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0))
        chunks.append(struct.pack("<I", len(encoded)))
        chunks.append(encoded)
        chunks.append(struct.pack("<I", values.size))
        chunks.append(values.tobytes(order="C"))
    return b"".join(chunks)
