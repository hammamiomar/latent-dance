/**
 * AudioPlayerWindow - Win95 styled audio control
 *
 * Pure renderer — all upload/polling logic lives in useAudioUpload.
 * Upload lifecycle state comes from useAudioStore (single source of truth).
 *
 * Visual states derived from store:
 *   empty:      no audioId, uploadPhase idle
 *   loading:    uploading | processing | loading_stems
 *   generating: audioId present + isGenerating
 *   ready:      audioId present + uploadPhase ready
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { useShallow } from "zustand/shallow";
import { useAudioMixer } from "../hooks/useAudioMixer";
import { useAudioUpload } from "../hooks/useAudioUpload";
import { Win95Window, Win95Button, Win95Slider } from "./ui/Win95Window";
import { useAudioStore } from "../stores/useAudioStore";
import { useDestinationStore } from "../stores/useDestinationStore";
import type { PhysicalStem } from "../types/sae";

// =============================================================================
// Types
// =============================================================================

interface AudioPlayerWindowProps {
  isOpen: boolean;
  onClose: () => void;
  onMinimize: () => void;
  isMinimized: boolean;
  initialPosition?: { x: number; y: number };
  // Playback callbacks
  onPlay?: (time: number) => void;
  onPause?: () => void;
  onSeek?: (time: number) => void;
  // Generation callbacks
  onGenerate?: () => Promise<void>;
  onStopGeneration?: () => void;
  onNewSong?: () => void;
  // State from parent
  isGenerating?: boolean;
  wsStatus?: "connected" | "connecting" | "disconnected" | "error";
  // Audio ready callback
  onAudioReady?: (audioId: string) => void;
  // Time sync
  onTimeSync?: (time: number) => void;
  // Track metadata
  bpm?: number;
}

const STEMS: PhysicalStem[] = ["bass", "drums", "vocals", "other"];

const STEM_LABELS: Record<PhysicalStem, string> = {
  bass: "BAS",
  drums: "DRM",
  vocals: "VOX",
  other: "OTH",
};

// =============================================================================
// Helpers (module-level for use by sub-components)
// =============================================================================

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// =============================================================================
// Ready State Component (extracted for destination awareness)
// =============================================================================

interface ReadyStateProps {
  filename: string | null;
  duration: number;
  bpm: number | null;
  onGenerate: () => void;
  onNewSong: () => void;
}

function ReadyState({ filename, duration, bpm, onGenerate, onNewSong }: ReadyStateProps) {
  // Check destination readiness
  const latentDest = useDestinationStore((s) => s.latent);
  const promptDest = useDestinationStore((s) => s.prompt);

  const isLatentReady = latentDest.destinationA !== null || latentDest.destinationB !== null;
  const isPromptReady = promptDest.destinationA !== null || promptDest.destinationB !== null;
  const canGenerate = isLatentReady && isPromptReady;

  return (
    <>
      <div className="win95-inset p-2">
        <div className="flex items-center justify-between">
          <div
            className="text-xs truncate flex-1 pr-2"
            style={{ color: "var(--color-text-primary)" }}
            title={filename || ""}
          >
            {filename}
          </div>
          <div className="text-xxs" style={{ color: "var(--color-text-dim)" }}>
            {formatTime(duration)}
            {bpm && bpm > 0 && (
              <span style={{ color: "var(--color-accent)" }}> • {Math.round(bpm)}</span>
            )}
          </div>
        </div>
      </div>

      {/* Destination guidance */}
      <div className="win95-inset p-2">
        <div className="text-xxs text-center" style={{ color: "var(--color-text-dim)" }}>
          {!canGenerate ? (
            <>
              Set destinations via the orbs:
              <div className="flex justify-center gap-4 mt-1">
                <span style={{ color: isLatentReady ? "var(--color-accent)" : "var(--color-text-dim)" }}>
                  {isLatentReady ? "✓" : "○"} Latent
                </span>
                <span style={{ color: isPromptReady ? "var(--color-accent)" : "var(--color-text-dim)" }}>
                  {isPromptReady ? "✓" : "○"} Prompt
                </span>
              </div>
            </>
          ) : (
            <span style={{ color: "var(--color-accent)" }}>Ready to generate</span>
          )}
        </div>
      </div>

      <Win95Button
        onClick={onGenerate}
        variant="primary"
        className="w-full py-2 text-sm"
        disabled={!canGenerate}
        style={!canGenerate ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}
      >
        {canGenerate ? "▶ GENERATE" : "SET DESTINATIONS"}
      </Win95Button>

      <button
        onClick={onNewSong}
        className="text-xxs text-center"
        style={{ color: "var(--color-text-dim)" }}
      >
        ← different song
      </button>
    </>
  );
}

// =============================================================================
// Component
// =============================================================================

export function AudioPlayerWindow({
  isOpen,
  onClose,
  onMinimize,
  isMinimized,
  initialPosition = { x: 80, y: 200 },
  onPlay,
  onPause,
  onSeek,
  onGenerate,
  onStopGeneration,
  onNewSong,
  isGenerating = false,
  wsStatus = "disconnected",
  onAudioReady,
  onTimeSync,
  bpm,
}: AudioPlayerWindowProps) {
  // ===========================================================================
  // Audio Store (single source of truth)
  // ===========================================================================

  // Narrow selector: the player renders currentTime (so it ticks with
  // playback by design), but must not re-render on store fields it never
  // shows (solo state, library ids, ...).
  const {
    audioId,
    filename,
    duration,
    currentTime,
    isPlaying,
    stemVolumes,
    stemMuted,
    masterVolume,
    play,
    pause,
    seek,
    setStemVolume,
    toggleStemMute,
    setMasterVolume,
    clearAudio,
    // Upload lifecycle
    uploadPhase,
    uploadStatusLabel,
    uploadProgress,
  } = useAudioStore(
    useShallow((s) => ({
      audioId: s.audioId,
      filename: s.filename,
      duration: s.duration,
      currentTime: s.currentTime,
      isPlaying: s.isPlaying,
      stemVolumes: s.stemVolumes,
      stemMuted: s.stemMuted,
      masterVolume: s.masterVolume,
      play: s.play,
      pause: s.pause,
      seek: s.seek,
      setStemVolume: s.setStemVolume,
      toggleStemMute: s.toggleStemMute,
      setMasterVolume: s.setMasterVolume,
      clearAudio: s.clearAudio,
      uploadPhase: s.uploadPhase,
      uploadStatusLabel: s.uploadStatusLabel,
      uploadProgress: s.uploadProgress,
    })),
  );

  // ===========================================================================
  // Audio Mixer
  // ===========================================================================

  const {
    load: loadAudio,
    play: playAudio,
    pause: pauseAudio,
    seek: seekAudio,
    stop: stopAudio,
  } = useAudioMixer({ onTimeSync });

  // ===========================================================================
  // Upload hook (all upload + polling logic)
  // ===========================================================================

  const { uploadFile, uploadYoutube, cancelUpload } = useAudioUpload({ onAudioReady });

  // ===========================================================================
  // Local State (UI-only, not upload lifecycle)
  // ===========================================================================

  const [inputMode, setInputMode] = useState<"file" | "youtube">("file");
  const [youtubeUrl, setYoutubeUrl] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const seekTimeoutRef = useRef<number | null>(null);

  // ===========================================================================
  // Derived State
  // ===========================================================================

  // Load stems when audioId appears and they're not already loaded/loading.
  // Handles both fresh uploads and page reloads with cached audioId.
  // loadAudio has its own concurrent guard — safe to call redundantly.
  useEffect(() => {
    if (audioId && uploadPhase !== 'loading_stems' && uploadPhase !== 'ready') {
      loadAudio(audioId);
    }
  }, [audioId, uploadPhase, loadAudio]);

  // Visual state derived cleanly from store
  const visualState = (() => {
    if (!audioId && uploadPhase === 'idle') return 'empty';
    if (uploadPhase === 'uploading' || uploadPhase === 'processing' || uploadPhase === 'loading_stems') return 'loading';
    if (uploadPhase === 'error') return 'empty';
    if (isGenerating) return 'generating';
    return 'ready';
  })();

  // ===========================================================================
  // Playback Controls
  // ===========================================================================

  const handlePlayPause = useCallback(() => {
    if (isPlaying) {
      pauseAudio();
      pause();
      onPause?.();
    } else {
      playAudio(currentTime);
      play();
      onPlay?.(currentTime);
    }
  }, [isPlaying, currentTime, play, pause, playAudio, pauseAudio, onPlay, onPause]);

  const handleSeek = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!progressRef.current || duration === 0) return;

      const rect = progressRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percent = Math.max(0, Math.min(1, x / rect.width));
      const newTime = percent * duration;

      seekAudio(newTime);
      seek(newTime);

      if (seekTimeoutRef.current) clearTimeout(seekTimeoutRef.current);
      seekTimeoutRef.current = window.setTimeout(() => {
        onSeek?.(newTime);
        seekTimeoutRef.current = null;
      }, 150);
    },
    [duration, seek, seekAudio, onSeek]
  );

  // ===========================================================================
  // Generation Controls
  // ===========================================================================

  const handleGenerate = useCallback(async () => {
    if (!audioId) return;

    // loadAudio has concurrent guard — safe to call even if already loaded
    await loadAudio(audioId);
    await onGenerate?.();

    playAudio(0);
    play();
    onPlay?.(0);
  }, [audioId, loadAudio, onGenerate, playAudio, play, onPlay]);

  const handleStop = useCallback(() => {
    pauseAudio();
    pause();
    seek(0);
    onStopGeneration?.();
  }, [pauseAudio, pause, seek, onStopGeneration]);

  // ===========================================================================
  // File Upload
  // ===========================================================================

  const handleFileUpload = useCallback(
    async (file: File) => {
      await uploadFile(file);
      // Reset file input so re-uploading the same file works
      if (fileInputRef.current) fileInputRef.current.value = '';
    },
    [uploadFile]
  );

  const handleYoutubeSubmit = useCallback(async () => {
    if (!youtubeUrl.trim()) return;
    await uploadYoutube(youtubeUrl);
    setYoutubeUrl("");
  }, [youtubeUrl, uploadYoutube]);

  const handleNewSong = useCallback(() => {
    stopAudio();
    pause();
    seek(0);
    clearAudio();
    cancelUpload();
    setYoutubeUrl("");
    onNewSong?.();
  }, [stopAudio, pause, seek, clearAudio, cancelUpload, onNewSong]);

  // ===========================================================================
  // Progress
  // ===========================================================================

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  // ===========================================================================
  // Render
  // ===========================================================================

  return (
    <Win95Window
      title="◉ AUDIO"
      isOpen={isOpen}
      onClose={onClose}
      onMinimize={onMinimize}
      isMinimized={isMinimized}
      initialPosition={initialPosition}
      width={340}
      zIndex={160}
    >
      <div className="flex flex-col gap-2">
        {/* ================================================================= */}
        {/* EMPTY: Song selection                                             */}
        {/* ================================================================= */}

        {visualState === "empty" && (
          <>
            <div className="flex gap-1">
              <button
                className={`win95-button text-xs flex-1 ${inputMode === "file" ? "win95-button--primary" : ""}`}
                onClick={() => setInputMode("file")}
              >
                FILE
              </button>
              <button
                className={`win95-button text-xs flex-1 ${inputMode === "youtube" ? "win95-button--primary" : ""}`}
                onClick={() => setInputMode("youtube")}
              >
                YOUTUBE
              </button>
            </div>

            {inputMode === "file" && (
              <div className="win95-inset p-3">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileUpload(file);
                  }}
                />
                <Win95Button
                  className="w-full text-xs py-2"
                  onClick={() => fileInputRef.current?.click()}
                >
                  SELECT AUDIO FILE
                </Win95Button>
              </div>
            )}

            {inputMode === "youtube" && (
              <div className="win95-inset p-2 flex gap-1">
                <input
                  type="text"
                  placeholder="paste YouTube URL..."
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleYoutubeSubmit()}
                  className="flex-1 bg-transparent border-none outline-none text-xs"
                  style={{ color: "var(--color-text-primary)" }}
                />
                <Win95Button
                  className="text-xs px-3"
                  onClick={handleYoutubeSubmit}
                  disabled={!youtubeUrl.trim()}
                >
                  GO
                </Win95Button>
              </div>
            )}
          </>
        )}

        {/* ================================================================= */}
        {/* LOADING: Processing status                                        */}
        {/* ================================================================= */}

        {visualState === "loading" && (
          <div className="win95-inset p-4 text-center">
            <div className="text-xs mb-2" style={{ color: "var(--color-accent)" }}>
              {uploadStatusLabel || "Processing..."}
            </div>
            <div className="w-full h-1 rounded overflow-hidden" style={{ background: "var(--color-void-deep)" }}>
              <div
                className="h-full"
                style={{
                  background: "var(--color-accent)",
                  width: `${Math.max(2, Math.round(uploadProgress * 100))}%`,
                  transition: "width 0.5s ease",
                }}
              />
            </div>
          </div>
        )}

        {/* ================================================================= */}
        {/* READY: Song loaded, configure and generate                        */}
        {/* ================================================================= */}

        {visualState === "ready" && (
          <ReadyState
            filename={filename}
            duration={duration}
            bpm={bpm ?? null}
            onGenerate={handleGenerate}
            onNewSong={handleNewSong}
          />
        )}

        {/* ================================================================= */}
        {/* GENERATING: Playback controls                                     */}
        {/* ================================================================= */}

        {visualState === "generating" && (
          <>
            <div className="win95-inset p-2">
              <div className="flex items-center justify-between">
                <div
                  className="text-xs truncate flex-1 pr-2"
                  style={{ color: "var(--color-accent)" }}
                  title={filename || ""}
                >
                  {filename}
                </div>
                <div className="text-xxs" style={{ color: "var(--color-text-muted)" }}>
                  {isPlaying ? "LIVE" : "PAUSED"}
                </div>
              </div>
            </div>

            {/* Progress bar */}
            <div
              ref={progressRef}
              className="win95-inset h-5 cursor-pointer relative"
              onClick={handleSeek}
            >
              <div
                className="absolute inset-y-0 left-0"
                style={{ width: `${progress}%`, background: "var(--color-accent-dim)" }}
              />
              <div
                className="absolute top-0 bottom-0 w-1"
                style={{
                  left: `${progress}%`,
                  background: "var(--color-accent)",
                  boxShadow: "0 0 4px var(--color-accent-glow)",
                }}
              />
              <div
                className="absolute inset-0 flex items-center justify-center text-xxs pointer-events-none"
                style={{ color: "var(--color-text-muted)" }}
              >
                {formatTime(currentTime)} / {formatTime(duration)}
              </div>
            </div>

            {/* Transport */}
            <div className="flex gap-2">
              <Win95Button
                onClick={handlePlayPause}
                variant={isPlaying ? "primary" : "default"}
                className="flex-1 py-2"
              >
                {isPlaying ? "⏸ PAUSE" : "▶ PLAY"}
              </Win95Button>
              <Win95Button onClick={handleStop} className="px-3 py-2" title="Stop generation">
                ⏹
              </Win95Button>
              <Win95Button onClick={handleNewSong} className="px-3 py-2" title="New song">
                ↻
              </Win95Button>
            </div>

            {bpm && bpm > 0 && (
              <div className="text-center text-xxs" style={{ color: "var(--color-text-dim)" }}>
                {Math.round(bpm)} BPM
              </div>
            )}
          </>
        )}

        {/* ================================================================= */}
        {/* Stem Mixer (visible when loaded)                                  */}
        {/* ================================================================= */}

        {(visualState === "ready" || visualState === "generating") && (
          <div className="flex gap-1 pt-2 border-t" style={{ borderColor: "var(--color-panel-border)" }}>
            {STEMS.map((stem) => {
              return (
                <div
                  key={stem}
                  className="flex-1 flex flex-col items-center gap-1"
                >
                  <Win95Button
                    className={`text-xxs w-full px-1 ${stemMuted[stem] ? "" : "win95-button--primary"}`}
                    onClick={() => toggleStemMute(stem)}
                    style={{ borderColor: stemMuted[stem] ? undefined : `var(--color-stem-${stem})` }}
                  >
                    {STEM_LABELS[stem]}
                  </Win95Button>
                  <Win95Slider
                    value={stemVolumes[stem] * 100}
                    min={0}
                    max={100}
                    onChange={(v) => setStemVolume(stem, v / 100)}
                    className="w-full"
                  />
                </div>
              );
            })}
            <div className="flex-1 flex flex-col items-center gap-1">
              <Win95Button className="text-xxs w-full">MST</Win95Button>
              <Win95Slider
                value={masterVolume * 100}
                min={0}
                max={100}
                onChange={(v) => setMasterVolume(v / 100)}
                className="w-full"
              />
            </div>
          </div>
        )}

        {/* ================================================================= */}
        {/* Connection status                                                 */}
        {/* ================================================================= */}

        <div
          className="flex items-center justify-center gap-2 pt-1 text-xxs"
          style={{ color: "var(--color-text-dim)" }}
        >
          <span
            className="w-2 h-2 rounded-full"
            style={{
              background:
                wsStatus === "connected"
                  ? "var(--color-success)"
                  : wsStatus === "connecting"
                    ? "var(--color-warning)"
                    : "var(--color-text-dim)",
            }}
          />
          {wsStatus === "connected" ? "ONLINE" : wsStatus.toUpperCase()}
        </div>
      </div>
    </Win95Window>
  );
}
