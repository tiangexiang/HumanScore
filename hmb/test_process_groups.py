"""hmb.process_groups: children in their own process groups, stopped with their descendants on
Ctrl-C/SIGTERM.  The signal tests send real signals to a separate driver process."""
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hmb import process_groups
from hmb.test_batch import recorded_pids, wait_until_gone

REPO = Path(__file__).resolve().parents[1]

# A child that starts a grandchild (optionally ignoring SIGTERM), records both pids, then waits.
CHILD = r'''
import os, signal, subprocess, sys, time
pids, ignore = sys.argv[1], sys.argv[2] == "ignore"
grandchild = subprocess.Popen([sys.executable, "-c",
    "import signal, sys, time\n"
    "if sys.argv[1] == 'ignore': signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "time.sleep(300)", sys.argv[2]])
with open(pids, "a") as fh:
    fh.write(f"child {os.getpid()}\ngrandchild {grandchild.pid}\n")
print("child ready", flush=True)
time.sleep(300)
'''


def wait_for(path: Path, names, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pids = recorded_pids(path)
        if all(name in pids for name in names):
            return pids
        time.sleep(0.05)
    raise AssertionError(f"{names} not recorded in {path}: {recorded_pids(path)}")


def kill_all(pids) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_terminate_stops_the_whole_group_and_kills_what_ignores_sigterm(tmp_path):
    pids = tmp_path / "pids.txt"
    proc = process_groups.popen([sys.executable, "-c", CHILD, str(pids), "ignore"])
    try:
        recorded = wait_for(pids, ("child", "grandchild"))
        assert os.getpgid(proc.pid) == proc.pid != os.getpgrp()  # a group of its own
        assert os.getpgid(recorded["grandchild"]) == proc.pid
        started = time.monotonic()
        process_groups.terminate([proc], grace=1.0)
        assert proc.returncode == -signal.SIGTERM  # the leader exited on SIGTERM ...
        assert wait_until_gone(list(recorded.values())) == []  # ... the grandchild got SIGKILL after the grace
        assert time.monotonic() - started < 10
        process_groups.terminate([proc])  # already stopped: nothing to do
    finally:
        kill_all(recorded_pids(pids).values())


def test_raise_on_termination_takes_over_only_default_handlers_and_restores_them():
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    with process_groups.raise_on_termination():
        handler = signal.getsignal(signal.SIGTERM)
        assert handler not in (signal.SIG_DFL, signal.SIG_IGN)
        with pytest.raises(process_groups.Terminated) as info:
            handler(signal.SIGTERM, None)
        assert isinstance(info.value, KeyboardInterrupt)  # existing Ctrl-C cleanup paths handle it
        assert process_groups.exit_status(info.value) == 143
        assert process_groups.describe(info.value) == "terminated by SIGTERM"
        with process_groups.raise_on_termination():  # nested: the outer handler stays
            assert signal.getsignal(signal.SIGTERM) is handler
        assert signal.getsignal(signal.SIGTERM) is handler
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL

    seen = []

    def in_thread():
        with process_groups.raise_on_termination():  # Python sets handlers only in the main thread
            seen.append(signal.getsignal(signal.SIGTERM))

    thread = threading.Thread(target=in_thread)
    thread.start()
    thread.join()
    assert seen == [signal.SIG_DFL]
    assert process_groups.exit_status(KeyboardInterrupt()) == 130
    assert process_groups.describe(KeyboardInterrupt()) == "interrupted"

    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)  # nohup
    try:
        with process_groups.raise_on_termination():
            assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)


def test_run_returns_like_subprocess_run(tmp_path):
    done = process_groups.run([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert done.returncode == 3
    with pytest.raises(subprocess.CalledProcessError):
        process_groups.run([sys.executable, "-c", "import sys; sys.exit(3)"], check=True)
    assert process_groups.run([sys.executable, "-c", "print('ok')"], cwd=str(tmp_path), check=True).returncode == 0


# Runs one of the child-owning calls in a separate process, which the test then signals.
DRIVER = r'''
import os, signal, sys
signal.signal(signal.SIGINT, signal.default_int_handler)   # in case the test runner ignores them
signal.signal(signal.SIGTERM, signal.SIG_DFL)
sys.path.insert(0, sys.argv[1])
from hmb import process_groups
mode, child, pids = sys.argv[2], sys.argv[3], sys.argv[4]
cmd = [sys.executable, "-c", child, pids, "default"]
try:
    if mode == "run":
        process_groups.run(cmd)
    else:
        from hmb.tracking import sam3_backend
        sam3_backend._run_worker(cmd, dict(os.environ), quiet=True)
except KeyboardInterrupt as exc:
    print(process_groups.describe(exc), file=sys.stderr, flush=True)
    sys.exit(process_groups.exit_status(exc))
'''


@pytest.mark.parametrize("mode", ["run", "sam3_worker"])
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_to_the_driver_stops_its_child_group(tmp_path, mode, signum):
    """``process_groups.run`` (hmb.cli's metric and fit subprocesses) and the SAM 3.1 worker of the
    multi-person script: a signal to the driver alone stops the child and the grandchild."""
    pids = tmp_path / "pids.txt"
    driver = subprocess.Popen([sys.executable, "-c", DRIVER, str(REPO), mode, CHILD, str(pids)],
                              stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        recorded = wait_for(pids, ("child", "grandchild"))
        assert os.getpgid(recorded["child"]) != os.getpgid(driver.pid)
        driver.send_signal(signum)
        _, err = driver.communicate(timeout=60)
        assert driver.returncode == 128 + signum
        assert ("terminated by SIGTERM" if signum == signal.SIGTERM else "interrupted") in err
        assert wait_until_gone(list(recorded.values())) == []
    finally:
        driver.kill()
        kill_all(recorded_pids(pids).values())
        driver.wait()
