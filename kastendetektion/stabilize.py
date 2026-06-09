"""
Zeitliche Stabilisierung der Slot-Zustände gegen Flackern.

Mehrheitsentscheid je Slot-Index über ein gleitendes Fenster der letzten N Frames.
"""

from __future__ import annotations

from collections import Counter, deque

from kastendetektion.states import SlotState


class SlotStateStabilizer:
    """Gleitender Mehrheitsentscheid je Slot über die letzten ``window`` Frames."""

    def __init__(self, window: int = 10) -> None:
        if window < 1:
            raise ValueError("window muss >= 1 sein")
        self.window = window
        self._history: dict[int, deque[SlotState]] = {}

    def update(self, states: list[SlotState]) -> list[SlotState]:
        """Fügt die Zustände eines Frames hinzu und gibt die stabilisierten Zustände zurück."""
        stabilized: list[SlotState] = []
        for idx, state in enumerate(states):
            hist = self._history.setdefault(idx, deque(maxlen=self.window))
            hist.append(state)
            stabilized.append(self._majority(hist))
        return stabilized

    def reset(self) -> None:
        self._history.clear()

    @staticmethod
    def _majority(hist: deque[SlotState]) -> SlotState:
        counts = Counter(hist)
        # most_common ist bei Gleichstand einfügereihenfolgeabhängig -> deterministisch genug.
        return counts.most_common(1)[0][0]
