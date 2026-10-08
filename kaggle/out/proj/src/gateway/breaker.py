"""Circuit breaker for one backend.

closed     normal; `failure_threshold` consecutive failures open the breaker
open       every request is refused for `open_s` seconds
half_open  one probe request is allowed; success closes, failure re-opens
"""

from __future__ import annotations

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, open_s: float = 5.0) -> None:
        self.failure_threshold = failure_threshold
        self.open_s = open_s
        self.state = CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    def peek(self, now: float) -> bool:
        """Would a request be allowed right now? Unlike allow(), does not claim the probe."""
        if self.state == OPEN:
            return now - self._opened_at >= self.open_s
        return self.state == CLOSED or not self._probe_in_flight

    def allow(self, now: float) -> bool:
        if self.state == OPEN and now - self._opened_at >= self.open_s:
            self.state = HALF_OPEN
            self._probe_in_flight = False
        if self.state == CLOSED:
            return True
        if self.state == HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record(self, now: float, ok: bool) -> None:
        if ok:
            self.state, self._failures, self._probe_in_flight = CLOSED, 0, False
            return
        self._failures += 1
        if self.state == HALF_OPEN or self._failures >= self.failure_threshold:
            self.state, self._opened_at, self._probe_in_flight = OPEN, now, False
