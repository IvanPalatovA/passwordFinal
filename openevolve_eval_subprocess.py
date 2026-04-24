"""Isolated evaluator runner for hard timeouts.

Usage:
    python openevolve_eval_subprocess.py <evaluation_file> <program_path> <result_path>
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
import traceback


def _load_evaluate_function(evaluation_file: str):
    spec = importlib.util.spec_from_file_location("openevolve_eval_module", evaluation_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec from {evaluation_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "evaluate"):
        raise AttributeError(f"Evaluation file {evaluation_file} must define evaluate()")

    return module.evaluate


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        raise SystemExit(
            "Usage: python openevolve_eval_subprocess.py "
            "<evaluation_file> <program_path> <result_path>"
        )

    _, evaluation_file, program_path, result_path = argv

    try:
        evaluate = _load_evaluate_function(evaluation_file)
        payload = {"status": "ok", "result": evaluate(program_path)}
    except Exception as exc:
        payload = {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }

    with open(result_path, "wb") as fh:
        pickle.dump(payload, fh)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
