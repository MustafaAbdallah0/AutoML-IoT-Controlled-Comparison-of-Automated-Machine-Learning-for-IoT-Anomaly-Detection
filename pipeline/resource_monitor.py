"""
resource_monitor.py
Background-thread monitor of a process AND all of its (recursive) children - Auto-sklearn
spawns worker processes and H2O runs in a separate JVM, so the whole tree must be sampled.

FIX (v2): CPU is now computed from cumulative cpu_times() deltas of every live process between
two samples. The earlier version called psutil's cpu_percent(interval=None) on freshly created
Process objects every loop, which always returns 0.0 on the first call, so CPU of all child
processes (workers, JVM) was lost and peak_cpu_percent only reflected the main Python process.

Reported:
  peak_rss_mb        peak of the SUM of resident set sizes over the process tree
                     (shared pages are counted once per process, so forked workers inflate it)
  peak_cpu_percent   peak per-interval CPU use of the whole tree, 100 = one full core
  mean_cpu_percent   total sampled CPU seconds / elapsed wall time * 100
Processes that live shorter than one sampling interval are not seen.
"""
import os
import time
import threading

try:
    import psutil
except ImportError:
    raise ImportError("psutil is required for resource monitoring. Install with: pip install psutil")


class ResourceMonitor:
    def __init__(self, pid=None, interval=1.0):
        self.pid = pid or os.getpid()
        self.interval = interval
        self.peak_rss_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.peak_num_processes = 0
        self.cpu_seconds = 0.0
        self.samples = []
        self._t_start = None
        self._t_stop = None
        self._stop_event = threading.Event()
        self._thread = None

    def _poll(self):
        try:
            root = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        last = {}                       # (pid, create_time) -> cumulative cpu seconds at last sample
        t_prev = time.time()
        while not self._stop_event.is_set():
            now = time.time()
            dt = max(now - t_prev, 1e-6)
            t_prev = now
            try:
                procs = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                procs = [root]
            total_rss, delta_cpu, alive = 0, 0.0, 0
            for p in procs:
                try:
                    key = (p.pid, p.create_time())
                    ct = p.cpu_times()
                    tot = ct.user + ct.system
                    total_rss += p.memory_info().rss
                    if key in last:
                        delta_cpu += max(0.0, tot - last[key])
                    last[key] = tot
                    alive += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            cpu_pct = 100.0 * delta_cpu / dt
            self.cpu_seconds += delta_cpu
            rss_mb = total_rss / (1024 ** 2)
            self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
            self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_pct)
            self.peak_num_processes = max(self.peak_num_processes, alive)
            self.samples.append({"t": now, "rss_mb": round(rss_mb, 1), "cpu_pct": round(cpu_pct, 1),
                                 "num_processes": alive})
            self._stop_event.wait(self.interval)

    def start(self):
        self._stop_event.clear()
        self._t_start = time.time()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._t_stop = time.time()

    def summary(self):
        elapsed = max((self._t_stop or time.time()) - (self._t_start or time.time()), 1e-6)
        return {
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "peak_cpu_percent": round(self.peak_cpu_percent, 1),
            "mean_cpu_percent": round(100.0 * self.cpu_seconds / elapsed, 1),
            "peak_num_processes": self.peak_num_processes,
            "num_samples": len(self.samples),
            "available_cpu_cores": os.cpu_count(),
        }
