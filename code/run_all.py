"""
run_all.py -- reproduce the entire CRADLE study in one command.

    cd code && python3 run_all.py

Steps:
  1. verify.py        -> verification triangle (PASS/FAIL, validation report)
  2. experiments.py   -> all result CSVs + headline.json
  3. make_figures.py  -> results/figures/*.{png,pdf}
  4. gen_tables.py    -> paper/tables/*.tex
  5. emit_macros.py   -> paper/macros.tex
  6. pytest           -> unit / reproduction test suite
"""
from __future__ import annotations

import os
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("\n" + "#" * 76 + f"\n# {' '.join(cmd)}\n" + "#" * 76)
    return subprocess.call(cmd, cwd=HERE, env={**os.environ})


def main():
    rc = 0
    rc |= run([sys.executable, "verify.py"])
    rc |= run([sys.executable, "experiments.py"])
    rc |= run([sys.executable, "make_figures.py"])
    rc |= run([sys.executable, "gen_tables.py"])
    rc |= run([sys.executable, "emit_macros.py"])
    try:
        rc |= run([sys.executable, "-m", "pytest", "tests/", "-q"])
    except Exception as e:
        print("pytest skipped:", e)
    print("\nDONE." if rc == 0 else "\nDONE (with failures).")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
