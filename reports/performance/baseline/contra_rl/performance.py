"""Opt-in phase timing and resource sampling. GPU counters are system-wide."""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import threading
import time
import torch

_ACTIVE = None
_NULL = nullcontext()


def span(name):
    return _ACTIVE.span(name) if _ACTIVE else _NULL


class TrainingProfile:
    def __init__(self, device='cpu', sync=False, interval=.5):
        self.device, self.sync, self.interval = device, sync, interval
        self.seconds, self.calls = defaultdict(float), defaultdict(int)
        self.samples = []
        self.stop_event = threading.Event()

    def synchronize(self):
        if self.device == 'mps':
            torch.mps.synchronize()
        elif self.device == 'cuda':
            torch.cuda.synchronize()

    @contextmanager
    def span(self, name):
        if self.sync:
            self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                self.synchronize()
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1

    def start(self):
        global _ACTIVE
        self.synchronize()
        self.started = time.perf_counter()
        self.cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        _ACTIVE = self
        self.thread.start()

    def _monitor(self):
        previous_time = time.perf_counter()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        previous_cpu = usage.ru_utime + usage.ru_stime
        while not self.stop_event.wait(self.interval):
            now = time.perf_counter()
            usage = resource.getrusage(resource.RUSAGE_SELF)
            cpu = usage.ru_utime + usage.ru_stime
            sample = dict(seconds=now - self.started,
                          process_cpu_percent=100 * (cpu - previous_cpu) / (now - previous_time))
            previous_cpu, previous_time = cpu, now
            try:
                result = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(os.getpid())],
                                        capture_output=True, text=True, timeout=2)
                sample['rss_bytes'] = int(result.stdout.strip()) * 1024
            except (ValueError, OSError, subprocess.TimeoutExpired):
                pass
            try:
                result = subprocess.run(['/usr/sbin/ioreg', '-r', '-c', 'IOAccelerator', '-l'],
                                        capture_output=True, text=True, timeout=2)
                match = re.search(r'"Device Utilization %"=(\d+)', result.stdout)
                if match:
                    sample['system_gpu_device_percent'] = int(match.group(1))
                else:
                    sample['gpu_counter_status'] = 'Unavailable: IORegistry counter missing or access restricted'
            except (OSError, subprocess.TimeoutExpired) as e:
                sample['gpu_counter_status'] = str(e)
            self.samples.append(sample)

    def finish(self, summary, path, extra=None):
        global _ACTIVE
        self.synchronize()
        elapsed = time.perf_counter() - self.started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.stop_event.set()
        self.thread.join(timeout=3)
        _ACTIVE = None
        cpu_seconds = (usage.ru_utime + usage.ru_stime - self.cpu_start.ru_utime - self.cpu_start.ru_stime)
        result = dict(device=self.device, phase_timers_synchronized=self.sync,
                      phase_timing_note='Nested phase totals overlap; without sync MPS timings measure host dispatch, not isolated GPU kernels',
                      measured_seconds=elapsed, steps_per_second=summary['steps'] / elapsed,
                      emulator_frames_per_second=summary['run_totals']['emulator_frames'] / elapsed,
                      updates_per_second=summary['updates'] / elapsed, steps=summary['steps'], updates=summary['updates'],
                      process_cpu_percent_mean=100 * cpu_seconds / elapsed,
                      cpu_percent_note='100% = one CPU core; 1000% = all 10 cores on this Mac',
                      peak_rss_bytes=usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                      phases={k: dict(seconds=v, calls=self.calls[k], mean_ms=v * 1000 / self.calls[k]) for k,v in self.seconds.items()},
                      samples=self.samples, gpu_counter_note='System-wide IORegistry Device Utilization %, not process GPU usage; no power/thermal data without sudo',
                      summary=summary, extra=extra or {})
        if self.device == 'mps':
            result['mps_memory'] = dict(allocated_bytes=torch.mps.current_allocated_memory(),
                                        driver_allocated_bytes=torch.mps.driver_allocated_memory())
        for key in ['process_cpu_percent', 'system_gpu_device_percent']:
            values = [s[key] for s in self.samples if key in s]
            if values:
                result[key + '_sample_mean'] = sum(values) / len(values)
                result[key + '_sample_max'] = max(values)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(result, indent=2))
        return result
