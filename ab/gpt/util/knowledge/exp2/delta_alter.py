"""Experiment 2 generator: delta (diff) edits + self-repair from a runtime smoke check.

Separate from Experiment 1: new files only, outputs in their own folder, nothing existing is modified.
Reuses (read-only) web_alter's network set, job pairing (same networks and seeds in both conditions),
vLLM backend, and nn-gpt's extract_delta/apply_delta (as in the lab's delta NAS method).
Output layout is identical to web_alter, so run_eval_chunks / analyse_runs work unchanged.

Per job:  delta prompt (+ web suggestions in the web condition) -> extract + apply diff
          -> smoke check -> if it fails: condition-neutral repair prompt (code + error, no suggestions)
          -> up to --max_repairs rounds. The last code that applied is saved for Eval.

Usage: python -m ab.gpt.util.knowledge.exp2.delta_alter --set SET.csv --vllm_url URL [--extra_info JSON]
"""
import argparse
import ast
import difflib
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from ab.gpt.act.alter.web_alter import VLLMBackend, build_jobs, family, load_rows
from ab.gpt.util.Const import new_nn_file, new_out_file
from ab.gpt.util.Util import extract_delta
from ab.gpt.util.nn.DeltaUtil import apply_delta

REPO = Path(__file__).resolve().parents[5]

SYSTEM = ("You are an expert software engineer specialising in neural network optimisation. "
          "You improve PyTorch models by writing precise unified diffs against the exact code you are given.")

PROMPT = """Task: write a unified diff (delta) that improves the ARCHITECTURE of the baseline model below, \
to increase its accuracy on CIFAR-10 image classification after 1 training epoch.

=== BASELINE MODEL (current accuracy: @ACC@) ===
<nn>
@CODE@
</nn>

@EXTRA@=== RULES ===
1. Output ONLY a unified diff inside <delta></delta> tags, written against the baseline exactly as shown.
2. Context lines (starting with a space) and removed lines (starting with '-') must match the baseline exactly.
3. Change only the architecture (layers, blocks, forward pass). Do not change supported_hyperparameters,
   train_setup or learn.
4. Use only hyperparameters the baseline already reads from prm; never add new prm keys.
5. If you use a new class, function or name, define or import it in the diff (e.g. `from typing import Optional`).
6. Do not assume a fixed input resolution: images may be resized (e.g. to 299 px). Never hard-code spatial sizes.
7. Keep the diff small and focused: a few targeted changes, not a rewrite.

Format:
<delta>
--- baseline.py
+++ improved.py
@@ -L,N +L,M @@
 unchanged line from the baseline
-removed line from the baseline
+added line
</delta>
"""

EXTRA_HEADER = "=== WEB RESEARCH SUGGESTIONS for this architecture (apply those that fit, as small changes) ===\n"

REPAIR_RUN = """The model below fails when it is built and trained:

<nn>
@CODE@
</nn>

Error: @ERR@
@WHERE@
Fix this error with a minimal unified diff written against THIS code exactly as shown. Keep the intended \
architecture changes; do not change supported_hyperparameters, train_setup or learn unless the error is there; \
never add new prm keys; define or import every name you use. Output ONLY the diff inside <delta></delta> tags.
"""

REPAIR_APPLY = """Your diff could not be applied to the code below: @ERR@.

<nn>
@CODE@
</nn>

Write the diff again against THIS code exactly as shown: every context line (starting with a space) and every \
removed line (starting with '-') must be copied exactly from the code. Output ONLY the diff inside \
<delta></delta> tags.
"""


def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace(f"@{k}@", str(v))
    return template


def msgs_for(user):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def smoke(code, row, workdir, timeout):
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "smoke_nn.py").write_text(code)
    row.to_pickle(workdir / "dataframe.df")
    try:
        p = subprocess.run([sys.executable, "-m", "ab.gpt.util.knowledge.exp2.smoke", str(workdir / "smoke_nn.py"),
                            str(workdir / "dataframe.df")], cwd=REPO, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"TimeoutError: building and training 2 batches took more than {timeout}s",
                "where": "", "infra": False}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("SMOKE_RESULT "):
            return json.loads(line[len("SMOKE_RESULT "):])
    tail = (p.stderr.strip().splitlines() or ["no output"])[-1]
    return {"ok": False, "error": f"SmokeCrash: {tail[:400]}", "where": "", "infra": True}


def changed_lines(a, b):
    return sum(1 for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
               if l[:1] in "+-" and not l.startswith(("+++", "---")))


def run_job(backend, job, out_dir, idx, max_repairs, smoke_timeout):
    row, base = job["row"], job["row"]["nn_code"]
    rec = dict(B=idx, nn=row["nn"], fam=family(row["nn"]), sample=job["k"], seed=job["seed"],
               extra_info=bool(job["extra"]))
    extra = (EXTRA_HEADER + job["extra"].strip() + "\n\n") if job["extra"] else ""
    msgs = msgs_for(fill(PROMPT, ACC=f"{float(row['accuracy']):.4f}", CODE=base, EXTRA=extra))
    work = out_dir / "work" / f"B{idx}"
    current, code, attempts, transcript = base, None, [], []
    t0 = time.time()
    for rnd in range(max_repairs + 1):
        try:
            out, finish, _, _ = backend.generate(msgs, job["seed"] + 1000 * rnd)
        except Exception as e:
            attempts.append({"round": rnd, "stage": "llm", "error": str(e)[:300]})
            break
        transcript.append({"round": rnd, "messages": msgs, "answer": out, "finish": finish})
        delta = extract_delta(out)
        new = apply_delta(current, delta) if delta else None
        if not new:
            err = "no diff found in the answer" if not delta else \
                "context or removed lines do not match the code exactly"
            attempts.append({"round": rnd, "stage": "apply", "ok": False, "error": err})
            msgs = msgs_for(fill(REPAIR_APPLY, ERR=err, CODE=current))
            continue
        code = current = new
        res = smoke(code, row, work / f"r{rnd}", smoke_timeout)
        attempts.append({"round": rnd, "stage": "smoke", **res})
        if res.get("ok") or res.get("infra"):
            break
        where = f"Location: {res['where']}\n" if res.get("where") else ""
        msgs = msgs_for(fill(REPAIR_RUN, CODE=current, ERR=res["error"], WHERE=where))
    smokes = [a for a in attempts if a["stage"] == "smoke"]
    parses = False
    if code:
        try:
            ast.parse(code)
            parses = True
        except SyntaxError:
            pass
    d = out_dir / ("synth_nn" if code else "failed") / f"B{idx}"
    d.mkdir(parents=True, exist_ok=True)
    (d / new_out_file).write_text("\n\n===== next round =====\n\n".join(t["answer"] for t in transcript))
    (d / "transcript.json").write_text(json.dumps(transcript, indent=1))
    (d / "attempts.json").write_text(json.dumps(attempts, indent=1))
    (d / f"original_{row['nn']}.py").write_text(base)
    row.to_pickle(d / "dataframe.df")
    if code:
        (d / new_nn_file).write_text(code)
    rec.update(code=bool(code), parses=parses, rounds=len(transcript),
               apply_failures=sum(a["stage"] == "apply" for a in attempts),
               smoke_first=smokes[0].get("ok") if smokes else None, smoke_last=smokes[-1].get("ok") if smokes else None,
               smoke_infra=any(a.get("infra") for a in smokes),
               changed_lines=changed_lines(base, code) if code else None, seconds=round(time.time() - t0, 1))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True)
    p.add_argument("--extra_info", default=None, help="JSON {nn or family: text}; omit for the baseline condition")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--max_repairs", type=int, default=2)
    p.add_argument("--smoke_timeout", type=float, default=300)
    p.add_argument("--vllm_url", required=True)
    p.add_argument("--vllm_model", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--dataset", default="cifar-10")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=str(Path.home() / "web_alter_runs_exp2"))
    p.add_argument("--name", default=None)
    a = p.parse_args()

    extra = json.load(open(a.extra_info)) if a.extra_info else {}
    cond = "web" if a.extra_info else "base"
    out_dir = Path(a.out) / (a.name or f"{datetime.now():%Y%m%d-%H%M}_{cond}")
    if out_dir.exists():
        sys.exit(f"[delta_alter] {out_dir} exists -- choose another --name")
    rows = load_rows(a.set, "img-classification", a.dataset)
    jobs = build_jobs(rows, a.samples, a.seed, extra)[:a.limit]
    backend = VLLMBackend(a.vllm_url, a.vllm_model, a)
    out_dir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        commit = "unknown"
    prompts_md5 = hashlib.md5((SYSTEM + PROMPT + EXTRA_HEADER + REPAIR_RUN + REPAIR_APPLY).encode()).hexdigest()
    meta = dict(vars(a), experiment="exp2-delta-repair", condition=cond, backend=backend.name, nn_gpt_commit=commit,
                nn_dataset=version("nn-dataset"), prompts_md5=prompts_md5, networks=len(rows), jobs=len(jobs),
                with_extra_info=sum(bool(j["extra"]) for j in jobs), started=datetime.now().isoformat(timespec="seconds"))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[delta_alter] {cond}: {len(jobs)} jobs, {len(rows)} networks, {backend.name} -> {out_dir}", flush=True)

    n = {"code": 0, "smoke_ok": 0, "repaired": 0}
    with open(out_dir / "results.jsonl", "w") as log, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, backend, j, out_dir, i, a.max_repairs, a.smoke_timeout) for i, j in enumerate(jobs)]
        for done, f in enumerate(as_completed(futs), 1):
            r = f.result()
            log.write(json.dumps(r) + "\n")
            log.flush()
            n["code"] += r["code"]
            n["smoke_ok"] += bool(r["smoke_last"])
            n["repaired"] += bool(r["smoke_last"]) and r["smoke_first"] is False
            print(f"[{done}/{len(jobs)}] B{r['B']} {r['nn'][:26]:26s} code={r['code']} rounds={r['rounds']} "
                  f"smoke first={r['smoke_first']} last={r['smoke_last']} changed={r['changed_lines']} "
                  f"{r['seconds']}s", flush=True)
    print(f"[delta_alter] done: {len(jobs)} jobs | code {n['code']} | smoke OK {n['smoke_ok']} "
          f"(of which repaired {n['repaired']}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
