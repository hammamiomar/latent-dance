"""Audio clock for playback synchronization.

Provides PLL-style drift correction for tight audio-visual sync.
"""


class AudioClock:
    """PLL-style audio clock for drift correction.

    Maintains a local estimate of audio playback time,
    with continuous correction based on client updates.
    """

    def __init__(self, rate: float = 1.0, alpha: float = 0.5):
        """Initialize audio clock.

        Args:
            rate: Playback rate (1.0 = normal speed)
            alpha: Smoothing factor for drift correction (0-1)
        """
        self.rate = rate
        self.alpha = alpha
        self.t0_client = 0.0  # Last known client time
        self.t0_server = None  # Corresponding server time
        self.playing = False

    def play(self, ct: float, now: float):
        """Start playback from given time."""
        self.t0_client = float(ct)
        self.t0_server = now
        self.playing = True

    def pause(self):
        """Pause playback."""
        self.playing = False

    def seek(self, ct: float, now: float):
        """Seek to new time."""
        self.t0_client = float(ct)
        self.t0_server = now

    def update(self, ct: float, now: float):
        """Update clock with client time report (drift correction)."""
        if self.t0_server is None:
            self.play(ct, now)
            return

        # Predict where we think the client should be
        predicted = self.t0_client + self.rate * (now - self.t0_server)

        # Calculate drift and apply correction
        drift = float(ct) - predicted
        self.t0_server -= self.alpha * drift / max(self.rate, 1e-6)

    def now(self, now: float) -> float:
        """Get current audio time estimate."""
        if not self.playing or self.t0_server is None:
            return self.t0_client
        return self.t0_client + self.rate * (now - self.t0_server)
