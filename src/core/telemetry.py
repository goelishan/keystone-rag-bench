import time
from contextlib import contextmanager
from typing import Dict


class TelemetryRecorder:
    def __init__(self):
        self.timings: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self.timings[name] = end - start

    def get_timings(self) -> Dict[str, float]:
        return self.timings.copy()
