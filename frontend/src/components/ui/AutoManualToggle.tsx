/**
 * AutoManualToggle - Switch between simple and complex configuration
 *
 * SIMPLE uses defaults, COMPLEX exposes advanced controls.
 */

interface AutoManualToggleProps {
  value: 'auto' | 'manual';
  onChange: (value: 'auto' | 'manual') => void;
}

export function AutoManualToggle({ value, onChange }: AutoManualToggleProps) {
  return (
    <div className="auto-manual-toggle">
      {/* SIMPLE option */}
      <button
        type="button"
        onClick={() => onChange('auto')}
        className={`auto-manual-toggle__option ${value === 'auto' ? 'auto-manual-toggle__option--active' : ''}`}
      >
        <span className="auto-manual-toggle__diamond">
          {value === 'auto' ? '◆' : '◇'}
        </span>
        SIMPLE
      </button>

      {/* COMPLEX option */}
      <button
        type="button"
        onClick={() => onChange('manual')}
        className={`auto-manual-toggle__option ${value === 'manual' ? 'auto-manual-toggle__option--active' : ''}`}
      >
        <span className="auto-manual-toggle__diamond">
          {value === 'manual' ? '◆' : '◇'}
        </span>
        COMPLEX
      </button>
    </div>
  );
}

