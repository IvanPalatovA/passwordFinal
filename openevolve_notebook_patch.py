"""Runtime patches for running OpenEvolve safely from a notebook.

These patches solve two notebook-specific issues:
1. Evaluator timeouts in OpenEvolve only cancel the await, not the underlying
   synchronous torch work running in a thread.
2. Lingering multiprocessing worker processes may survive after an interrupted
   or timed-out run and keep gigabytes of RAM allocated.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


def _kill_process_tree(pid: int, grace_seconds: float = 3.0) -> None:
    """Terminate a process and all of its descendants."""
    if pid <= 0:
        return

    if psutil is not None:
        try:
            parent = psutil.Process(pid)
        except psutil.Error:
            return

        try:
            targets = parent.children(recursive=True)
        except (psutil.Error, PermissionError):
            targets = []
        targets.append(parent)

        for proc in targets:
            try:
                proc.terminate()
            except psutil.Error:
                pass

        _, alive = psutil.wait_procs(targets, timeout=grace_seconds)

        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass

        psutil.wait_procs(alive, timeout=grace_seconds)
        return

    # Fallback without psutil: terminate the parent only.
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _run_evaluate_in_subprocess(
    evaluation_file: str,
    program_path: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """Execute the evaluator in a subprocess that can be force-killed."""
    helper_path = Path(__file__).with_name("openevolve_eval_subprocess.py")
    result_fd, result_path = tempfile.mkstemp(prefix="oe_eval_", suffix=".pkl")
    os.close(result_fd)

    env = os.environ.copy()
    # Ensure the evaluator subprocess flushes output promptly (helps in notebooks).
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        [
            sys.executable,
            str(helper_path),
            evaluation_file,
            program_path,
            result_path,
        ],
        # Don't silence output: first evaluation can take a while (model load/download).
        stdout=None,
        stderr=None,
        cwd=os.getcwd(),
        env=env,
    )

    try:
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc.pid)
            raise asyncio.TimeoutError(
                f"Evaluation subprocess timed out after {timeout_seconds}s"
            )

        if not os.path.exists(result_path):
            raise RuntimeError(
                "Evaluator subprocess exited without writing a result. "
                f"Return code: {proc.returncode}"
            )

        import pickle

        with open(result_path, "rb") as fh:
            payload = pickle.load(fh)

        if payload.get("status") == "ok":
            return payload["result"]

        error = payload.get("error", {})
        raise RuntimeError(
            "Evaluator subprocess failed with "
            f"{error.get('type')}: {error.get('message')}\n"
            f"{error.get('traceback', '')}"
        )
    finally:
        try:
            if proc.poll() is None:
                _kill_process_tree(proc.pid)
        except Exception:
            pass
        try:
            os.unlink(result_path)
        except OSError:
            pass


async def _direct_evaluate_hard_timeout(self, program_path: str):
    """Monkey-patched Evaluator._direct_evaluate with hard subprocess timeout."""
    return await asyncio.to_thread(
        _run_evaluate_in_subprocess,
        self.evaluation_file,
        program_path,
        self.config.timeout,
    )


def apply_hard_timeout_patch() -> None:
    """Patch OpenEvolve Evaluator to enforce a real timeout."""
    from openevolve.evaluator import Evaluator

    if getattr(Evaluator._direct_evaluate, "__name__", "") == "_direct_evaluate_hard_timeout":
        return

    Evaluator._direct_evaluate = _direct_evaluate_hard_timeout


def _looks_like_python_process(proc) -> bool:
    try:
        name = (proc.name() or "").lower()
        cmdline = " ".join(proc.cmdline()).lower()
    except Exception:
        return False

    if "resource_tracker" in cmdline:
        return False

    return "python" in name or "python" in cmdline


def _collect_descendant_python_processes(root_pid: int | None = None) -> List[Any]:
    if psutil is None:
        return []

    root_pid = root_pid or os.getpid()
    try:
        root = psutil.Process(root_pid)
    except psutil.Error:
        return []

    targets = []
    try:
        descendants = root.children(recursive=True)
    except (psutil.Error, PermissionError):
        return []

    for proc in descendants:
        if proc.pid == os.getpid():
            continue
        if _looks_like_python_process(proc):
            targets.append(proc)
    return targets


def cleanup_python_descendants(timeout: float = 3.0) -> List[Tuple[int, str]]:
    """Force-stop lingering Python descendants of the current notebook kernel."""
    if psutil is None:
        children = mp.active_children()
        for proc in children:
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in children:
            try:
                proc.join(timeout=timeout)
            except Exception:
                pass
        return [(proc.pid or -1, proc.name) for proc in children if proc.pid]

    targets = _collect_descendant_python_processes()
    info = []

    for proc in targets:
        try:
            info.append((proc.pid, " ".join(proc.cmdline()) or proc.name()))
        except Exception:
            info.append((proc.pid, "<unknown>"))

    for proc in targets:
        try:
            proc.terminate()
        except psutil.Error:
            pass

    _, alive = psutil.wait_procs(targets, timeout=timeout)

    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass

    psutil.wait_procs(alive, timeout=timeout)
    return info


def describe_python_descendants() -> List[Tuple[int, str]]:
    """Return current descendant Python processes for diagnostics."""
    if psutil is None:
        return []

    desc = []
    for proc in _collect_descendant_python_processes():
        try:
            desc.append((proc.pid, " ".join(proc.cmdline()) or proc.name()))
        except Exception:
            desc.append((proc.pid, "<unknown>"))
    return desc
