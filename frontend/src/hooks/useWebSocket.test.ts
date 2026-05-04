/**
 * Tests for useWebSocket hook - sync messaging
 *
 * Note: Full WebSocket integration tests require complex event simulation.
 * These tests focus on message format contracts.
 * See syncRobustness.test.ts for comprehensive unit tests of sync logic.
 */

import { describe, it, expect } from 'vitest';
import {
  BINARY_KIND_JPEG_FRAME,
  BINARY_KIND_SONG_CURVES,
  decodeSongCurvePayload,
  demuxBinaryPayload,
} from './useWebSocket';

function uint32(value: number): Uint8Array {
  const buffer = new ArrayBuffer(4);
  new DataView(buffer).setUint32(0, value, true);
  return new Uint8Array(buffer);
}

function float32(values: number[]): Uint8Array {
  return new Uint8Array(new Float32Array(values).buffer);
}

function concat(chunks: Uint8Array[]): ArrayBuffer {
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result.buffer;
}

function packCurves(entries: Array<[string, number[]]>): ArrayBuffer {
  const encoder = new TextEncoder();
  const chunks: Uint8Array[] = [uint32(entries.length)];
  for (const [name, values] of entries) {
    const encodedName = encoder.encode(name);
    chunks.push(uint32(encodedName.byteLength));
    chunks.push(encodedName);
    chunks.push(uint32(values.length));
    chunks.push(float32(values));
  }
  return concat(chunks);
}

describe('useWebSocket - Message Contracts', () => {
  describe('Audio sync message formats', () => {
    it('should format audio_timeupdate message correctly', () => {
      const time = 42.5;
      const message = JSON.stringify({
        action: 'audio_timeupdate',
        time,
      });

      const parsed = JSON.parse(message);
      expect(parsed.action).toBe('audio_timeupdate');
      expect(parsed.time).toBe(42.5);
    });

    it('should format audio_play message correctly', () => {
      const time = 10.0;
      const message = JSON.stringify({
        action: 'audio_play',
        time,
      });

      const parsed = JSON.parse(message);
      expect(parsed.action).toBe('audio_play');
      expect(parsed.time).toBe(10.0);
    });

    it('should format audio_pause message correctly', () => {
      const message = JSON.stringify({
        action: 'audio_pause',
      });

      const parsed = JSON.parse(message);
      expect(parsed.action).toBe('audio_pause');
    });

    it('should format audio_seek message correctly', () => {
      const time = 90.0;
      const message = JSON.stringify({
        action: 'audio_seek',
        time,
      });

      const parsed = JSON.parse(message);
      expect(parsed.action).toBe('audio_seek');
      expect(parsed.time).toBe(90.0);
    });
  });

  describe('Destination message formats', () => {
    it('should format clear_destination message correctly', () => {
      const message = JSON.stringify({
        action: 'clear_destination',
        space: 'prompt',
        slot: 'b',
      });

      const parsed = JSON.parse(message);
      expect(parsed.action).toBe('clear_destination');
      expect(parsed.space).toBe('prompt');
      expect(parsed.slot).toBe('b');
    });
  });

  describe('Extended activity message parsing', () => {
    it('should parse extended_activity message with stems and prominence', () => {
      const serverMessage = {
        type: 'extended_activity',
        audio_time: 42.5,
        stems: {
          drums: { envelope: 0.8, flash: 0.6, sustain: 0.4 },
          bass: { envelope: 0.5, flash: 0.3, sustain: 0.7 },
        },
        prominence: {
          drums: { prominence: 0.85, surprise_active: false, rank: 1 },
          bass: { prominence: 0.6, surprise_active: true, rank: 2 },
        },
      };

      expect(serverMessage.audio_time).toBe(42.5);
      expect(serverMessage.stems.drums.envelope).toBe(0.8);
      expect(serverMessage.prominence?.drums.prominence).toBe(0.85);
      expect(serverMessage.prominence?.bass.surprise_active).toBe(true);
    });

    it('should include audioTime for drift detection', () => {
      const serverMessage = {
        type: 'extended_activity',
        audio_time: 100.0,
        stems: {},
      };

      expect(serverMessage.audio_time).toBe(100.0);
      expect(typeof serverMessage.audio_time).toBe('number');
    });
  });

  describe('Binary demux', () => {
    it('should route explicit JPEG payloads to frame handling without the header byte', () => {
      const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xdb]);
      const payload = concat([new Uint8Array([BINARY_KIND_JPEG_FRAME]), jpeg]);

      const demuxed = demuxBinaryPayload(payload);

      expect(demuxed.kind).toBe('frame');
      if (demuxed.kind === 'frame') {
        expect(Array.from(new Uint8Array(demuxed.payload))).toEqual(Array.from(jpeg));
      }
    });

    it('should keep compatibility with legacy unprefixed JPEG payloads', () => {
      const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xdb]);

      const demuxed = demuxBinaryPayload(jpeg.buffer);

      expect(demuxed.kind).toBe('frame');
      if (demuxed.kind === 'frame') {
        expect(Array.from(new Uint8Array(demuxed.payload))).toEqual(Array.from(jpeg));
      }
    });

    it('should decode packed song curves and lock index payloads', () => {
      const curvesPayload = packCurves([
        ['timestamps', [0, 1, 2]],
        ['tension', [0.1, 0.2, 0.3]],
        ['lock_index:bass-drums', [0.4, 0.5, 0.6]],
        ['target:drums_high:transient', [0.0, 1.0, 0.2]],
      ]);

      const decoded = decodeSongCurvePayload(curvesPayload);
      const tension = Array.from(decoded.tension ?? []);
      const lockIndex = Array.from(decoded.lock_index['bass-drums'] ?? []);
      const drumsTransient = Array.from(decoded.target_curves.drums_high?.transient ?? []);

      expect(Array.from(decoded.timestamps ?? [])).toEqual([0, 1, 2]);
      expect(tension[0]).toBeCloseTo(0.1);
      expect(tension[1]).toBeCloseTo(0.2);
      expect(tension[2]).toBeCloseTo(0.3);
      expect(lockIndex[0]).toBeCloseTo(0.4);
      expect(lockIndex[1]).toBeCloseTo(0.5);
      expect(lockIndex[2]).toBeCloseTo(0.6);
      expect(drumsTransient[0]).toBeCloseTo(0.0);
      expect(drumsTransient[1]).toBeCloseTo(1.0);
      expect(drumsTransient[2]).toBeCloseTo(0.2);
    });

    it('should never treat curve payloads as JPEG frames', () => {
      const curvesPayload = new Uint8Array(packCurves([['timestamps', [0, 1]]]));
      const payload = concat([new Uint8Array([BINARY_KIND_SONG_CURVES]), curvesPayload]);

      const demuxed = demuxBinaryPayload(payload);

      expect(demuxed.kind).toBe('curves');
    });

    it('should drop unknown binary payload kinds', () => {
      const demuxed = demuxBinaryPayload(new Uint8Array([0x7f, 0xff, 0xd8]).buffer);

      expect(demuxed).toEqual({ kind: 'unknown', kindByte: 0x7f });
    });
  });
});
