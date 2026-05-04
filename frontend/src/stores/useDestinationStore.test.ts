import { describe, expect, it, beforeEach } from 'vitest';
import { useDestinationStore } from './useDestinationStore';

describe('useDestinationStore', () => {
  beforeEach(() => {
    useDestinationStore.getState().reset();
  });

  it('promotes B into A when clearing A', () => {
    const store = useDestinationStore.getState();
    store.setDestination('prompt', 'a', { type: 'prompt', label: 'A', prompt: 'A' });
    store.setDestination('prompt', 'b', { type: 'prompt', label: 'B', prompt: 'B' });
    store.setBlendPosition('prompt', 0.8);

    store.clearDestination('prompt', 'a');

    const prompt = useDestinationStore.getState().prompt;
    expect(prompt.destinationA?.label).toBe('B');
    expect(prompt.destinationB).toBeNull();
    expect(prompt.blendPosition).toBe(0);
  });

  it('clears B while keeping A', () => {
    const store = useDestinationStore.getState();
    store.setDestination('latent', 'a', { type: 'seed', label: 'Seed 1', seed: 1 });
    store.setDestination('latent', 'b', { type: 'seed', label: 'Seed 2', seed: 2 });
    store.setBlendPosition('latent', 0.8);

    store.clearDestination('latent', 'b');

    const latent = useDestinationStore.getState().latent;
    expect(latent.destinationA?.label).toBe('Seed 1');
    expect(latent.destinationB).toBeNull();
    expect(latent.blendPosition).toBe(0);
  });
});
