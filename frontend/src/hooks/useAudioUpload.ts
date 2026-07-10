/**
 * useAudioUpload - Upload + polling lifecycle hook
 *
 * Extracts all upload/processing logic from AudioPlayerWindow.
 * Reads/writes useAudioStore directly — no stale closures from local state.
 *
 * Flow: upload → poll status → setAudioData → onAudioReady
 */

import { useCallback, useEffect, useRef } from 'react';
import { useAudioStore } from '../stores/useAudioStore';
import { notify } from '../stores/useNotificationStore';
import { useSongIntelligenceStore } from '../stores/useSongIntelligenceStore';

const POLL_INTERVAL_MS = 2000;
const MAX_CONSECUTIVE_FAILURES = 5;

// Map backend status strings to user-facing labels
function statusToLabel(status: string): string {
  switch (status) {
    case 'uploading':     return 'Uploading...';
    case 'downloading':   return 'Downloading...';
    case 'loading_cache': return 'Checking cache...';
    case 'separating':    return 'Separating stems...';
    case 'caching':       return 'Caching stems...';
    case 'extracting':    return 'Detecting BPM + virtual stems...';
    case 'extracting_features': return 'Extracting features...';
    case 'complete':      return 'Ready.';
    case 'error':         return 'Failed.';
    default:              return 'Processing...';
  }
}

interface UseAudioUploadOptions {
  onAudioReady?: (audioId: string) => void;
}

interface UseAudioUploadReturn {
  uploadFile: (file: File) => Promise<void>;
  uploadYoutube: (url: string) => Promise<void>;
  cancelUpload: () => void;
}

export function useAudioUpload(options: UseAudioUploadOptions = {}): UseAudioUploadReturn {
  const onAudioReadyRef = useRef(options.onAudioReady);
  onAudioReadyRef.current = options.onAudioReady;

  // Polling refs (not state — avoids re-renders)
  const pollIntervalRef = useRef<number | null>(null);
  const pollAudioIdRef = useRef<string | null>(null);
  const failureCountRef = useRef(0);

  // ============================================================================
  // Polling
  // ============================================================================

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current !== null) {
      window.clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    pollAudioIdRef.current = null;
    failureCountRef.current = 0;
  }, []);

  const pollStatus = useCallback(async () => {
    const audioId = pollAudioIdRef.current;
    if (!audioId) return;

    const { setUploadStatus, setUploadError, setAudioData, setUploadPhase } =
      useAudioStore.getState();

    try {
      const res = await fetch(`/api/audio/status/${audioId}`);
      if (!res.ok) throw new Error(`Status fetch failed: ${res.status}`);
      const data = await res.json();

      // Reset failure counter on success
      failureCountRef.current = 0;

      const label = statusToLabel(data.status || 'processing');
      const progress = Math.max(0, Math.min(1, data.progress ?? 0));
      setUploadPhase('processing');
      setUploadStatus(label, progress);

      if (data.status === 'complete') {
        stopPolling();
        // Read filename from store (always fresh, no stale closure)
        const { filename } = useAudioStore.getState();
        setAudioData(audioId, data.stems || [], data.duration || 0, filename ?? undefined);
        if (!useSongIntelligenceStore.getState().hydrateFromPayload(audioId, data)) {
          useSongIntelligenceStore.getState().clear();
        }
        // Phase transitions to 'loading_stems' when mixer.load() is called
        onAudioReadyRef.current?.(audioId);
      }

      if (data.status === 'error') {
        stopPolling();
        setUploadError(data.error || 'Audio processing failed.');
        notify.error('Audio processing failed.');
      }
    } catch (error) {
      failureCountRef.current += 1;
      console.warn(`Status poll error (${failureCountRef.current}/${MAX_CONSECUTIVE_FAILURES}):`, error);

      if (failureCountRef.current >= MAX_CONSECUTIVE_FAILURES) {
        stopPolling();
        setUploadError('Lost connection to server during processing.');
        notify.error('Lost connection to server. Try uploading again.');
      }
    }
  }, [stopPolling]);

  const startPolling = useCallback((audioId: string) => {
    stopPolling();
    pollAudioIdRef.current = audioId;
    failureCountRef.current = 0;

    // Immediate first poll, then interval
    pollStatus();
    pollIntervalRef.current = window.setInterval(pollStatus, POLL_INTERVAL_MS);
  }, [pollStatus, stopPolling]);

  // Cleanup on unmount
  useEffect(() => stopPolling, [stopPolling]);

  // ============================================================================
  // Upload actions
  // ============================================================================

  const uploadFile = useCallback(async (file: File) => {
    if (!file) return;

    const { startUpload, setUploadError } = useAudioStore.getState();

    // Immediate feedback before the fetch blocks
    startUpload(file.name);
    useSongIntelligenceStore.getState().clear();

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/audio/upload?async=1', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      const data = await response.json();
      startPolling(data.audio_id);
    } catch (error) {
      console.error('Upload error:', error);
      stopPolling();
      setUploadError('Upload failed. Try again.');
      notify.error('Upload failed. Try again.');
    }
  }, [startPolling, stopPolling]);

  const uploadYoutube = useCallback(async (url: string) => {
    if (!url.trim()) return;

    const { startUpload, setUploadError } = useAudioStore.getState();

    // Immediate feedback — show the URL as filename
    startUpload(url);
    useSongIntelligenceStore.getState().clear();
    useAudioStore.getState().setUploadStatus('Downloading...', 0);

    try {
      const response = await fetch('/api/audio/youtube?async=1', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) throw new Error('Download failed');
      const data = await response.json();
      startPolling(data.audio_id);
    } catch (error) {
      console.error('YouTube error:', error);
      stopPolling();
      setUploadError('YouTube download failed.');
      notify.error('YouTube download failed.');
    }
  }, [startPolling, stopPolling]);

  const cancelUpload = useCallback(() => {
    stopPolling();
    useAudioStore.getState().clearAudio();
    useSongIntelligenceStore.getState().clear();
  }, [stopPolling]);

  return { uploadFile, uploadYoutube, cancelUpload };
}
