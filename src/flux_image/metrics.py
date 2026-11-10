"""Windows driver memory counters, sampled without running another CUDA process."""
from __future__ import annotations

import os
import threading


class GPUMemorySampler:
    def __init__(self):
        self.done = threading.Event()
        self.thread = None
        self.result = {"available": False, "samples": 0, "peak_dedicated_mib": 0.0, "peak_shared_mib": 0.0}

    def start(self):
        if os.name == "nt":
            self.thread = threading.Thread(target=self._sample, name="flux-vram-metrics", daemon=True)
            self.thread.start()
        return self

    def stop(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=3)
        return dict(self.result)

    def _sample(self):
        import win32pdh
        query = win32pdh.OpenQuery()
        handles = {}
        try:
            while not self.done.is_set():
                try:
                    _, instances = win32pdh.EnumObjectItems(None, None, "GPU Process Memory", win32pdh.PERF_DETAIL_WIZARD)
                    for name in instances:
                        if not name.startswith(f"pid_{os.getpid()}_") or name in handles:
                            continue
                        handles[name] = {
                            kind: win32pdh.AddCounter(query, win32pdh.MakeCounterPath(
                                (None, "GPU Process Memory", name, None, -1, counter)))
                            for kind, counter in (("dedicated", "Dedicated Usage"), ("shared", "Shared Usage"))
                        }
                    if handles:
                        win32pdh.CollectQueryData(query)
                        totals = {"dedicated": 0, "shared": 0}
                        for counters in handles.values():
                            for kind, counter in counters.items():
                                _, value = win32pdh.GetFormattedCounterValue(counter, win32pdh.PDH_FMT_LARGE)
                                totals[kind] += value
                        self.result["available"] = True
                        self.result["samples"] += 1
                        for kind, value in totals.items():
                            field = f"peak_{kind}_mib"
                            self.result[field] = max(self.result[field], round(value / 1024**2, 2))
                except Exception as exc:
                    # Telemetry must not alter image inference behavior.
                    self.result["last_counter_error"] = type(exc).__name__
                self.done.wait(0.5)
        finally:
            win32pdh.CloseQuery(query)
