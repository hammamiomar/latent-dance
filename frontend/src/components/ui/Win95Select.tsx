/**
 * Win95Select - Generic dropdown in Win95 style
 */

interface Win95SelectProps<T extends string> {
  value: T;
  options: { value: T; label: string; description?: string }[];
  onChange: (value: T) => void;
  label?: string;
  compact?: boolean;
}

export function Win95Select<T extends string>({
  value,
  options,
  onChange,
  label,
  compact = false,
}: Win95SelectProps<T>) {
  return (
    <div className={`flex flex-col gap-0.5 ${compact ? 'flex-1' : ''}`}>
      {label && (
        <span
          className="text-xxs uppercase tracking-wider"
          style={{ color: 'var(--color-text-dim)' }}
        >
          {label}
        </span>
      )}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="win95-select"
        title={options.find((o) => o.value === value)?.description}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
