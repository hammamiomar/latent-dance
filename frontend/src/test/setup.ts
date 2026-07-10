import '@testing-library/jest-dom';
import { afterEach, vi } from 'vitest';
import { __resetTransport } from '../lib/transport';

// The transport is a module singleton — never let an attached fake socket
// leak between tests.
afterEach(() => {
  __resetTransport();
});

// Mock Web Audio API
class MockAudioContext {
  currentTime = 0;
  state: AudioContextState = 'running';
  destination = {};

  createGain() {
    return {
      gain: { value: 1, setTargetAtTime: vi.fn() },
      connect: vi.fn(),
    };
  }

  createBufferSource() {
    return {
      buffer: null,
      connect: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      disconnect: vi.fn(),
      onended: null,
    };
  }

  async decodeAudioData(_buffer: ArrayBuffer): Promise<AudioBuffer> {
    return {
      duration: 180, // 3 minutes
      length: 180 * 44100,
      numberOfChannels: 2,
      sampleRate: 44100,
      getChannelData: () => new Float32Array(180 * 44100),
      copyFromChannel: vi.fn(),
      copyToChannel: vi.fn(),
    };
  }

  async resume() {
    this.state = 'running';
  }

  async close() {
    this.state = 'closed';
  }
}

// @ts-expect-error - Mock global AudioContext
globalThis.AudioContext = MockAudioContext;

// Mock performance.now for consistent timing tests
let mockTime = 0;
vi.spyOn(performance, 'now').mockImplementation(() => mockTime);

// Helper to advance mock time
export function advanceTime(ms: number) {
  mockTime += ms;
}

export function resetMockTime() {
  mockTime = 0;
}

// Mock requestAnimationFrame
let rafCallbacks: Array<{ id: number; callback: FrameRequestCallback }> = [];
let rafId = 0;

globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => {
  const id = ++rafId;
  rafCallbacks.push({ id, callback });
  return id;
};

globalThis.cancelAnimationFrame = (id: number) => {
  rafCallbacks = rafCallbacks.filter((cb) => cb.id !== id);
};

// Helper to flush RAF callbacks
export function flushRAF() {
  const callbacks = [...rafCallbacks];
  rafCallbacks = [];
  callbacks.forEach(({ callback }) => callback(mockTime));
}

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  binaryType = 'arraybuffer';
  onopen: (() => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: ((error: unknown) => void) | null = null;

  constructor(_url: string) {
    // Auto-connect after a tick
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      this.onopen?.();
    }, 0);
  }

  send = vi.fn();

  close(code = 1000, reason = '') {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ code, reason });
  }

  // Test helpers
  simulateMessage(data: unknown) {
    this.onmessage?.({ data });
  }

  simulateDisconnect(code = 1006) {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ code, reason: 'Connection lost' });
  }
}

// @ts-expect-error - Mock global WebSocket
globalThis.WebSocket = MockWebSocket;

// Mock fetch for audio loading
globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
  if (url.includes('/api/audio/')) {
    return {
      ok: true,
      arrayBuffer: async () => new ArrayBuffer(1024),
      json: async () => ({ audio_id: 'test-audio', duration: 180 }),
    };
  }
  return { ok: false };
});
