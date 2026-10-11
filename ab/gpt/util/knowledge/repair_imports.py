"""Post-hoc import repair for a finished web_alter run (identical for every condition).

Many generated networks fail only because a standard import is missing (NameError: Optional, Callable,
Any, Tensor, OrderedDict, partial, ...). This makes a COPY of RUN_DIR as OUT_DIR in which:
  * networks whose error is "NameError: name 'X' is not defined" with X provided by HEADER get the fixed
    HEADER inserted at the top of new_nn.py and their result files removed (-> re-evaluated);
  * all other networks are copied unchanged WITH their results (a header of imports cannot change the
    outcome of code that did not fail on a missing name).
The original run is never modified. Then evaluate OUT_DIR with run_eval_chunks.

Usage: python -m ab.gpt.util.knowledge.repair_imports RUN_DIR OUT_DIR
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HEADER = """# --- post-hoc import repair (identical for all conditions) ---
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
# --- end of import repair ---
"""
PROVIDED = {"math", "OrderedDict", "partial", "Any", "Callable", "Dict", "List", "Optional", "Sequence",
            "Tuple", "Type", "Union", "torch", "Tensor", "nn", "F"}
RESULTS = ("eval_info.json", "eval_summary.json", "error.txt", "eval_verification_failed.txt")


def insert_header(code):
    lines = code.splitlines(keepends=True)
    i = 0  # keep "from __future__" imports (and anything before them) first
    for n, line in enumerate(lines):
        if line.startswith("from __future__"):
            i = n + 1
    return "".join(lines[:i]) + HEADER + "".join(lines[i:])


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        sys.exit(f"{out} exists")
    dirs = sorted((run / "synth_nn").glob("B*"), key=lambda d: int(d.name[1:]))
    pending = [d.name for d in dirs if not any((d / f).exists() for f in RESULTS)]
    if pending:
        sys.exit(f"{run.name}: {len(pending)} networks have no result yet ({pending[:5]}...), wait for Eval")
    shutil.copytree(run, out, symlinks=True, ignore=shutil.ignore_patterns("_eval_chunk", "eval.log*"))
    repaired = []
    for d in (out / "synth_nn").glob("B*"):
        err = d / "error.txt"
        m = re.match(r"NameError: name '(\w+)' is not defined", err.read_text(errors="replace")) if err.exists() else None
        if m and m.group(1) in PROVIDED:
            code = d / "new_nn.py"
            code.write_text(insert_header(code.read_text()))
            for f in RESULTS:
                (d / f).unlink(missing_ok=True)
            repaired.append({"B": int(d.name[1:]), "missing": m.group(1)})
    meta = json.loads((out / "run_meta.json").read_text())
    meta.update(repaired_from=str(run), repair_header_md5=hashlib.md5(HEADER.encode()).hexdigest(),
                repaired=len(repaired))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    (out / "repair_log.json").write_text(json.dumps(sorted(repaired, key=lambda r: r["B"]), indent=1))
    names = {}
    for r in repaired:
        names[r["missing"]] = names.get(r["missing"], 0) + 1
    print(f"{run.name} -> {out.name}: {len(repaired)} networks repaired for re-evaluation {names}")


if __name__ == "__main__":
    main()
