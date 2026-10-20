"""Throttled training progress on stderr; count actual emulator frames."""
import sys
import time


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class FrameProgress:
    def __init__(self, steps, skip, enabled=True, stream=None, clock=time.perf_counter):
        self.steps, self.skip = steps, skip
        self.stream = sys.stderr if stream is None else stream
        self.enabled, self.clock = enabled, clock
        self.tty = self.stream.isatty()
        self.started = self.last_render = clock()
        self.frames = self.step = 0
        self.finished = False

    def __enter__(self):
        self.render(force=True)
        return self

    def update(self, frames, step):
        self.frames += frames
        self.step = step
        self.render()

    def render(self, force=False, status='Training'):
        if not self.enabled:
            return
        now = self.clock()
        if not force and now - self.last_render < (0.5 if self.tty else 30.):
            return
        self.last_render = now
        elapsed = now - self.started
        # Remaining decisions can advance at most skip frames each. Revise the
        # estimate when a terminal transition advances fewer frames.
        total = self.frames + (self.steps - self.step) * self.skip
        fraction = self.frames / max(1, total)
        filled = int(20 * fraction)
        rate = self.frames / elapsed if elapsed > 0 else 0.
        eta = duration(elapsed * (self.steps - self.step) / self.step) if self.step else '--:--:--'
        line = (f'{status} [{"#" * filled}{"-" * (20 - filled)}] {fraction:6.2%} '
                f'{self.frames:,}/~{total:,} frames | {rate:,.0f} frames/s | '
                f'elapsed {duration(elapsed)} | ETA {eta}')
        self.stream.write(('\r\033[2K' if self.tty else '') + line + ('' if self.tty else '\n'))
        self.stream.flush()

    def write(self, message):
        if self.enabled and self.tty:
            self.stream.write('\r\033[2K')
            self.stream.flush()
        print(message, flush=True)
        if self.enabled and self.tty:
            self.render(force=True)

    def __exit__(self, exc_type, exc, traceback):
        self.render(force=True, status='Done' if self.finished else 'Stopped')
        if self.enabled and self.tty:
            self.stream.write('\n')
            self.stream.flush()
