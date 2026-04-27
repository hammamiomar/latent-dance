import { useRef, useCallback, useEffect } from 'react';

const GRID = 16;
const CELL = 16;
const SIZE = GRID * CELL;

const PRESETS: Record<string, number[]> = {
  floor:   [...Array(128).fill(0), ...Array(128).fill(1)],
  ceiling: [...Array(128).fill(1), ...Array(128).fill(0)],
  center:  [...Array(64).fill(0), ...Array(128).fill(1), ...Array(64).fill(0)],
  fill:    Array(256).fill(1),
  clear:   Array(256).fill(0),
};

interface SpatialGridProps {
  mask: number[];
  onChange: (mask: number[]) => void;
  blockColor: string;
}

export function SpatialGrid({ mask, onChange, blockColor }: SpatialGridProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const localMask = useRef([...mask]);
  const painting = useRef(false);
  const paintVal = useRef(1);
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const draw = useCallback(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#0c0e0d';
    ctx.fillRect(0, 0, SIZE, SIZE);

    for (let i = 0; i < 256; i++) {
      const col = i % GRID;
      const row = Math.floor(i / GRID);
      const x = col * CELL;
      const y = row * CELL;

      if (localMask.current[i] > 0.5) {
        ctx.fillStyle = blockColor + '25';
        ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
        ctx.fillStyle = blockColor + '80';
        ctx.fillRect(x + 3, y + 3, CELL - 6, CELL - 6);
      } else {
        ctx.fillStyle = '#101310';
        ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
      }
    }
  }, [blockColor]);

  useEffect(() => {
    localMask.current = [...mask];
    draw();
  }, [mask, draw]);

  useEffect(() => {
    return () => { clearTimeout(debounceTimer.current); };
  }, []);

  const cellAt = useCallback((e: React.PointerEvent): number => {
    const canvas = canvasRef.current;
    if (!canvas) return -1;
    const rect = canvas.getBoundingClientRect();
    const sx = SIZE / rect.width;
    const sy = SIZE / rect.height;
    const col = Math.floor((e.clientX - rect.left) * sx / CELL);
    const row = Math.floor((e.clientY - rect.top) * sy / CELL);
    return (col >= 0 && col < GRID && row >= 0 && row < GRID) ? row * GRID + col : -1;
  }, []);

  const emit = useCallback(() => {
    clearTimeout(debounceTimer.current);
    debounceTimer.current = setTimeout(() => onChange([...localMask.current]), 100);
  }, [onChange]);

  const paint = useCallback((idx: number) => {
    if (idx < 0 || localMask.current[idx] === paintVal.current) return;
    localMask.current[idx] = paintVal.current;
    draw();
    emit();
  }, [draw, emit]);

  const onDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    painting.current = true;
    paintVal.current = e.button === 2 ? 0 : 1;
    paint(cellAt(e));
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, [cellAt, paint]);

  const onMove = useCallback((e: React.PointerEvent) => {
    if (!painting.current) return;
    paint(cellAt(e));
  }, [cellAt, paint]);

  const onUp = useCallback(() => { painting.current = false; }, []);

  const applyPreset = useCallback((preset: number[]) => {
    localMask.current = [...preset];
    draw();
    onChange([...preset]);
  }, [draw, onChange]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '6px' }}>
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        aria-label="16x16 spatial mask grid"
        style={{
          width: '100%',
          aspectRatio: '1',
          cursor: 'crosshair',
          borderRadius: '2px',
          border: '1px solid var(--color-win95-dark)',
          boxShadow:
            'inset 1px 1px 0 var(--color-win95-dark), inset -1px -1px 0 var(--color-win95-light)',
          imageRendering: 'pixelated',
        }}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onContextMenu={(e) => e.preventDefault()}
      />
      <div style={{ display: 'flex', gap: '3px' }}>
        {Object.entries(PRESETS).map(([name, preset]) => (
          <PresetButton
            key={name}
            name={name}
            blockColor={blockColor}
            onClick={() => applyPreset(preset)}
          />
        ))}
      </div>
    </div>
  );
}

function PresetButton({
  name,
  blockColor,
  onClick,
}: {
  name: string;
  blockColor: string;
  onClick: () => void;
}) {
  const ref = useRef<HTMLButtonElement>(null);

  return (
    <button
      ref={ref}
      onClick={onClick}
      style={{
        flex: 1,
        padding: '4px 0',
        fontSize: '8.5px',
        fontFamily: 'var(--font-mono)',
        textTransform: 'uppercase',
        letterSpacing: '0.04em',
        background: 'var(--color-void-deep)',
        color: 'var(--color-text-muted)',
        border: '1px solid var(--color-void-elevated)',
        borderRadius: '2px',
        cursor: 'pointer',
        transition: 'background 0.08s, color 0.08s, border-color 0.08s',
      }}
      onMouseEnter={() => {
        const el = ref.current;
        if (!el) return;
        el.style.background = blockColor + '20';
        el.style.color = 'var(--color-text-primary)';
        el.style.borderColor = blockColor + '60';
      }}
      onMouseLeave={() => {
        const el = ref.current;
        if (!el) return;
        el.style.background = 'var(--color-void-deep)';
        el.style.color = 'var(--color-text-muted)';
        el.style.borderColor = 'var(--color-void-elevated)';
      }}
    >
      {name}
    </button>
  );
}
