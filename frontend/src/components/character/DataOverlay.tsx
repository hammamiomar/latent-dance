/**
 * DataOverlay — "Phosphor Archive"
 *
 * The DATA mode of the belly screen. The belly bezel IS the chassis;
 * this overlay renders as pure phosphor terminal content on the CRT —
 * no metal, no chassis, just engraved-green text and archive rows.
 *
 * Lists the persistent song library and lets the user activate a song.
 * The currently-loaded song is marked with a '>' cursor + bright phosphor.
 */

import {
  useCallback,
  useEffect,
  useState,
  type CSSProperties,
} from "react";
import { useAudioStore } from "../../stores/useAudioStore";
import {
  useSongIntelligenceStore,
  type SongAnalysis,
  type SongProfile,
} from "../../stores/useSongIntelligenceStore";
import { notify } from "../../stores/useNotificationStore";
import type { PhysicalStem, Stem } from "../../types/sae";

interface SongLibraryItem {
  song_id: string;
  filename: string | null;
  source_type: string;
  duration: number;
  bpm: number;
  stems: string[];
  status: string;
  unavailable_reason: string | null;
  updated_at: number;
}

interface SongLibraryResponse {
  songs: SongLibraryItem[];
}

interface SongActivationResponse {
  audio_id: string;
  song_id: string;
  filename: string | null;
  stems: string[];
  duration: number;
  bpm: number;
  song_profile?: SongProfile | null;
  song_analysis?: SongAnalysis | null;
  song_sections?: number[] | null;
}

const PHYSICAL_STEM_ORDER: PhysicalStem[] = ["bass", "drums", "vocals", "other"];
const ROW_FADE_STAGGER_MS = 40;
const ROW_STAGGER_CAP = 12;
let pendingLibraryRequest: Promise<SongLibraryResponse> | null = null;

const ASCII_ART = `    _______________                        |*\\_/*|________
  |  ___________  |     .-.     .-.      ||_/-\\_|______  |
  | |           | |    .****. .****.     | |           | |
  | |   0   0   | |    .*****.*****.     | |   0   0   | |
  | |     -     | |     .*********.      | |     -     | |
  | |   \\___/   | |      .*******.       | |   \\___/   | |
  | |___     ___| |       .*****.        | |___________| |
  |_____|\\_/|_____|        .***.         |_______________|
    _|__|/ \\|_|_.............*.............._|________|_
   / ********** \\                          / ********** \\
 /  ************  \\                      /  ************  \\
--------------------                    --------------------`;

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "—:—";
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}

function formatClock(d: Date): string {
  return d.toTimeString().slice(0, 5);
}

function formatAge(epochSec: number): string {
  if (!Number.isFinite(epochSec) || epochSec <= 0) return "—";
  const diff = Date.now() / 1000 - epochSec;
  if (diff < 45) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86_400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 86_400 * 30) return `${Math.floor(diff / 86_400)}d ago`;
  if (diff < 86_400 * 365) return `${Math.floor(diff / 86_400 / 30)}mo ago`;
  return `${Math.floor(diff / 86_400 / 365)}y ago`;
}

function songLabel(song: SongLibraryItem): string {
  return song.filename || song.song_id.slice(0, 8);
}

async function requestSongLibrary(): Promise<SongLibraryResponse> {
  pendingLibraryRequest ??= fetch("/api/audio/library")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Library request failed: ${response.status}`);
      }
      return response.json() as Promise<SongLibraryResponse>;
    })
    .finally(() => {
      pendingLibraryRequest = null;
    });
  return pendingLibraryRequest;
}

function StemList({ stems }: { stems: string[] }) {
  const present = new Set(stems);
  const label = PHYSICAL_STEM_ORDER
    .filter((stem) => present.has(stem))
    .join("/");
  return (
    <span className="data-stems" aria-label={`stems: ${label || "none"}`}>
      {label || "no stems"}
    </span>
  );
}

function AsciiArt() {
  return (
    <div className="data-ascii" aria-hidden>
      <pre className="data-ascii__art">{ASCII_ART}</pre>
    </div>
  );
}

export function DataOverlay() {
  const [songs, setSongs] = useState<SongLibraryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [activatingId, setActivatingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastSync, setLastSync] = useState<Date | null>(null);

  const activeLibrarySongId = useAudioStore((s) => s.librarySongId);

  const loadSongs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await requestSongLibrary();
      setSongs(data.songs);
      setLastSync(new Date());
    } catch (err) {
      console.error("[DataOverlay] Failed to load song library:", err);
      setError("archive offline");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSongs();
  }, [loadSongs]);

  const activateSong = async (song: SongLibraryItem) => {
    if (song.status !== "ready" || activatingId) return;
    setActivatingId(song.song_id);
    try {
      const response = await fetch(`/api/audio/library/${song.song_id}/activate`, {
        method: "POST",
      });
      if (!response.ok) throw new Error(`Activation failed: ${response.status}`);
      const data = (await response.json()) as SongActivationResponse;

      const store = useAudioStore.getState();
      store.pause();
      store.seek(0);
      store.setUploadPhase("idle");
      store.setAudioData(
        data.audio_id,
        data.stems as Stem[],
        data.duration,
        data.filename || songLabel(song),
        data.song_id,
      );
      if (!useSongIntelligenceStore.getState().hydrateFromPayload(data.audio_id, data)) {
        useSongIntelligenceStore.getState().clear();
      }
      notify.success("Song activated.");
    } catch (err) {
      console.error("[DataOverlay] Failed to activate song:", err);
      notify.error("Song activation failed.");
    } finally {
      setActivatingId(null);
    }
  };

  const showEmpty = !loading && !error && songs.length === 0;
  const showError = !loading && !!error;

  return (
    <div className="data-panel">
      <div className="data-panel__title-row">
        <div className="data-panel__title">ARCHIVE</div>
        <button
          className={`data-refresh ${loading ? "data-refresh--spinning" : ""}`}
          onClick={() => void loadSongs()}
          disabled={loading}
          title="Re-scan archive"
        >
          <span className="data-refresh__glyph" aria-hidden>⟲</span>
          <span>REFRESH</span>
        </button>
      </div>

      <div
        className={[
          "data-panel__status",
          error ? "data-panel__status--error" : "",
        ]
          .filter(Boolean)
          .join(" ")}
      >
        {error ? (
          <span>! {error}</span>
        ) : loading && songs.length === 0 ? (
          <span>▸ scanning…</span>
        ) : (
          <>
            <span>▸ catalog index</span>
            <span className="data-row__sep">·</span>
            <span>
              {songs.length} record{songs.length === 1 ? "" : "s"}
            </span>
            {lastSync && (
              <>
                <span className="data-row__sep">·</span>
                <span>synced {formatClock(lastSync)}</span>
              </>
            )}
          </>
        )}
      </div>

      <div className="data-panel__divider" />

      {showError ? (
        <div className="data-empty">
          <AsciiArt />
        </div>
      ) : showEmpty ? (
        <div className="data-empty">
          <AsciiArt />
        </div>
      ) : (
        <div className="data-panel__list">
          {loading && songs.length === 0 && <div className="data-scanbar" aria-hidden />}
          {songs.map((song, idx) => {
            const ready = song.status === "ready";
            const activating = activatingId === song.song_id;
            const active = activeLibrarySongId === song.song_id;
            const rowClass = [
              "data-row",
              !ready ? "data-row--unavailable" : "",
              activating ? "data-row--activating" : "",
              active ? "data-row--active" : "",
            ]
              .filter(Boolean)
              .join(" ");
            const rowStyle: CSSProperties = {
              animationDelay: `${Math.min(idx, ROW_STAGGER_CAP) * ROW_FADE_STAGGER_MS}ms`,
            };
            return (
              <button
                key={song.song_id}
                type="button"
                className={rowClass}
                style={rowStyle}
                disabled={!ready || !!activatingId}
                onClick={() => void activateSong(song)}
                title={song.unavailable_reason || songLabel(song)}
              >
                {activating && <span className="data-row__scanbar" aria-hidden />}
                <div className="data-row__line1">
                  <span className="data-row__name">{songLabel(song)}</span>
                  <span className="data-row__stats">
                    <span className="data-row__duration">{formatTime(song.duration)}</span>
                    <span>
                      <span className="data-row__bpm">{Math.round(song.bpm || 0) || "—"}</span>
                      <span className="data-row__bpm-unit">BPM</span>
                    </span>
                  </span>
                </div>
                <div className="data-row__line2">
                  <span className="data-row__source">{song.source_type || "—"}</span>
                  <span className="data-row__sep">·</span>
                  <StemList stems={song.stems} />
                  <span className="data-row__sep">·</span>
                  <span className="data-row__age">
                    {ready
                      ? formatAge(song.updated_at)
                      : song.unavailable_reason || "stems purged"}
                  </span>
                  {active && <span className="data-row__loaded-tag">LOADED</span>}
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
