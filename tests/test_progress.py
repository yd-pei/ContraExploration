import io

from contra_rl.cli import parse_args
from contra_rl.progress import FrameProgress


def test_actual_frames_and_revised_total_with_early_terminal():
    stream = io.StringIO()
    now = [0.]
    with FrameProgress(2, 4, stream=stream, clock=lambda: now[0]) as progress:
        now[0] = 1.
        progress.update(4, 1)
        now[0] = 2.
        progress.update(1, 2)  # Terminal step advanced only one frame.
        progress.finished = True
    final = stream.getvalue().splitlines()[-1]
    assert 'Done' in final and '100.00%' in final
    assert '5/~5 frames' in final and 'ETA 00:00:00' in final
    assert '\r' not in stream.getvalue()


def test_failure_reports_partial_progress():
    stream = io.StringIO()
    try:
        with FrameProgress(10, 4, stream=stream) as progress:
            progress.update(3, 1)
            raise RuntimeError('interrupted')
    except RuntimeError:
        pass
    final = stream.getvalue().splitlines()[-1]
    assert 'Stopped' in final and '3/~39 frames' in final
    assert '100.00%' not in final


def test_progress_can_be_disabled():
    assert parse_args(['train']).progress
    assert not parse_args(['train', '--no-progress']).progress
    stream = io.StringIO()
    with FrameProgress(1, 4, enabled=False, stream=stream) as progress:
        progress.update(4, 1)
    assert stream.getvalue() == ''
