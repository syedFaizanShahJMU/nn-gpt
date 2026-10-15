"""Smoke check v4 = Experiment 2 smoke check (exp2.smoke, unchanged and imported) + the end of the full traceback.

Why (E45): an error raised inside a library is reported by the v3 repair message only with the last line of the
generated code, e.g. `Conv2dNormActivation(` for "GELU.__init__() got an unexpected keyword argument 'inplace'".
The keyword is passed by torchvision itself (ConvNormActivation passes inplace=True to activation_layer by default),
so the model cannot find it in its own code and the repair rounds fail. v4 additionally returns `trace`: the last
frames of the traceback across ALL files (generated code and libraries), so the repair message can show them.
v4b (T02: with only the frame lines, Llama-3.3 still did not repair the GELU case in 2 of 2 jobs): frames of this
checker itself (exp2/exp4 smoke.py, importlib) are left out, and when the error is raised inside a library, the source
lines just before that library line are shown too (e.g. torchvision's `params = {} if inplace is None else
{"inplace": inplace}`), so the cause is visible.
Same subprocess interface and output line as exp2.smoke:
    python -m ab.gpt.util.knowledge.exp4.smoke CODE_FILE DATAFRAME_DF   ->   SMOKE_RESULT {json}
"""
import json
import linecache
import sys
import traceback
from pathlib import Path

from ab.gpt.util.knowledge.exp2.smoke import run

N_FRAMES = 4


HARNESS = ("knowledge/exp2/smoke.py", "knowledge/exp4/smoke.py", "<frozen importlib")
CONTEXT = 3  # source lines shown before the line where a library raised the error


def short_trace(tb, code_file, n=N_FRAMES):
    frames = [f for f in traceback.extract_tb(tb) if not any(h in f.filename for h in HARNESS)][-n:]
    out = []
    for i, f in enumerate(frames):
        lib = f.filename != code_file
        name = "/".join(Path(f.filename).parts[-2:]) if lib else "model code"
        out.append(f"{name}, line {f.lineno}, in {f.name}: {(f.line or '').strip()[:160]}")
        if lib and i == len(frames) - 1:  # error raised inside a library: show the lines just before it
            ctx = [linecache.getline(f.filename, k).rstrip() for k in range(max(1, f.lineno - CONTEXT), f.lineno + 1)]
            if any(c.strip() for c in ctx):
                out.append("    source of that library line:\n" + "\n".join("      " + c[:160] for c in ctx))
    return "\n".join(out)


def main():
    code_file, df_file = sys.argv[1], sys.argv[2]
    try:
        run(code_file, df_file)
        res = {"ok": True}
    except BaseException as e:  # noqa: BLE001 - report everything, including SystemExit from generated code
        msg = f"{type(e).__name__}: {e}"
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == code_file]
        where = f"line {frames[-1].lineno}: {frames[-1].line}" if frames else ""
        infra = "out of memory" in msg.lower() or "cuda error" in msg.lower()
        res = {"ok": False, "error": msg[:600], "where": where[:300], "infra": infra,
               "trace": short_trace(e.__traceback__, code_file)[:1500]}
    print("SMOKE_RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
