"""Child processes in their own process groups, stopped together with their descendants.

The HumanScore drivers (``scripts/humanscore_single.py``, ``scripts/humanscore_multi.py``,
``python -m hmb.cli``, ``python -m hmb.batch``) start every engine, worker and metric
subprocess with :func:`popen` or :func:`run`: the child gets a new session, so it and
every process it starts share one process group, and :func:`terminate` stops that
whole group (SIGTERM, then SIGKILL after a grace period).

Ctrl-C already raises ``KeyboardInterrupt`` in the driver; inside
:func:`raise_on_termination` SIGTERM and SIGHUP raise :class:`Terminated` (a
``KeyboardInterrupt``) too, so the same cleanup paths run on ``kill`` or a closed
terminal.  Without it SIGTERM ends the driver at once and its engines keep running
as orphans (on the GPU).
"""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from typing import Iterable, Iterator, Optional, Sequence

GRACE_SECONDS = 20.0  # time between SIGTERM and SIGKILL
TERMINATION_SIGNALS = tuple(getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name))
_GROUP = "_hmb_process_group"  # Popen attribute holding the child's process-group id


class Terminated(KeyboardInterrupt):
    """SIGTERM/SIGHUP received inside :func:`raise_on_termination` (raised in the main thread)."""

    def __init__(self, signum: int):
        self.signum = int(signum)
        super().__init__(f"terminated by {signal.Signals(self.signum).name}")

    @property
    def exit_status(self) -> int:
        """Shell convention for a process ended by a signal: 128 + the signal number."""
        return 128 + self.signum


def exit_status(exc: BaseException) -> int:
    """Shell exit status of an interrupted run: 128 + the signal (130 for Ctrl-C)."""
    return exc.exit_status if isinstance(exc, Terminated) else 128 + int(signal.SIGINT)


def describe(exc: BaseException) -> str:
    """``"interrupted"`` (Ctrl-C) or ``"terminated by SIGTERM"`` etc."""
    return str(exc) if isinstance(exc, Terminated) else "interrupted"


def _raise_terminated(signum, frame):
    raise Terminated(signum)


@contextlib.contextmanager
def raise_on_termination(signals: Sequence[int] = TERMINATION_SIGNALS) -> Iterator[None]:
    """Within the block, ``signals`` (default SIGTERM, SIGHUP) raise :class:`Terminated`.

    Only signals with the default disposition are taken over: an ignored SIGHUP
    (``nohup``) or a handler of the calling program is kept, and so is the handler of
    an enclosing block.  Outside the main thread (where Python cannot set handlers)
    the block runs unchanged.  The previous handlers are restored on exit.
    """
    installed = {}
    if threading.current_thread() is threading.main_thread():
        for sig in signals:
            try:
                if signal.getsignal(sig) == signal.SIG_DFL:
                    installed[sig] = signal.signal(sig, _raise_terminated)
            except (OSError, ValueError):
                pass
    try:
        yield
    finally:
        for sig, previous in installed.items():
            signal.signal(sig, previous)


def popen(cmd, **kwargs) -> subprocess.Popen:
    """``subprocess.Popen`` in a new session: the child leads a process group of its own."""
    kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    setattr(proc, _GROUP, proc.pid)
    return proc


def _signal(proc: subprocess.Popen, sig: int) -> None:
    """Send ``sig`` to the process group of ``proc`` (or to ``proc`` alone if it has none)."""
    group = getattr(proc, _GROUP, None)
    try:
        if group is not None:
            os.killpg(group, sig)  # the child and everything it started
        elif proc.poll() is None:
            proc.send_signal(sig)
    except (ProcessLookupError, PermissionError):
        pass  # the group is gone


def terminate(procs: Iterable[Optional[subprocess.Popen]], grace: float = GRACE_SECONDS) -> None:
    """Stop the running ``procs`` (from :func:`popen`) and their process groups.

    SIGTERM goes to every group at once; leaders that have not exited after
    ``grace`` seconds, and group members that outlive their leader, get SIGKILL.
    A second interrupt while waiting skips to the SIGKILL.  Processes that had
    already exited are left alone.
    """
    running = [proc for proc in procs if proc is not None and proc.poll() is None]
    try:
        for proc in running:
            _signal(proc, signal.SIGTERM)
        deadline = time.monotonic() + grace
        for proc in running:
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
    finally:
        for proc in running:
            _signal(proc, signal.SIGKILL)
        for proc in running:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # e.g. stuck in a driver call; the kernel ends it
                pass


def run(cmd, *, check: bool = False, grace: float = GRACE_SECONDS, **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run`` without output capture, the child in its own process group.

    An interrupt (Ctrl-C, or SIGTERM/SIGHUP, which raise :class:`Terminated` while the
    child runs) terminates the child's whole group before it propagates.
    """
    with raise_on_termination():
        proc = popen(cmd, **kwargs)
        try:
            code = proc.wait()
        except BaseException:
            terminate([proc], grace)
            raise
    if check and code:
        raise subprocess.CalledProcessError(code, proc.args)
    return subprocess.CompletedProcess(proc.args, code)
