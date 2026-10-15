"""Base pipeline v4 = Experiment 3 generator (exp3.edit_alter, unchanged and imported) + fixes from the Exp 3 audit.

All fixes apply identically to BOTH conditions (base and web); the web suggestions are still the only difference.
  1. Protected functions are ENFORCED, not only stated (E41/E42: in GoogLeNet the model changed learn() in 16 of 20
     jobs, 1 of them trained). After every answer, supported_hyperparameters, train_setup and learn are compared with
     the original by syntax tree (formatting and comments ignored); a change is rejected like an invalid edit, with a
     message naming the function. The repair message no longer allows changing learn "if the error is there".
  2. Output contract stated as a rule: forward() returns ONE tensor (batch, classes) in training and evaluation
     (E41: GoogLeNet's misleading return-type hint led the model to return a namedtuple).
  3. Repair messages show the end of the full traceback, including library frames (E45: a torchvision default,
     ConvNormActivation(inplace=True) -> GELU(inplace=True), was invisible in the model's own code; 8 of 8 jobs
     were not repaired in 2 rounds). Requires exp4.smoke.
  4. CUDA out-of-memory in the smoke check (another lab job on the GPU) is retried once after --oom_wait seconds
     (Exp 3: 11 smoke checks hit OOM and their code went to Eval untested).
  5. A repair answer that returns the failing code unchanged gets the error and traceback again, with "your answer is
     identical to the code that fails", instead of the edit-check message "make at least one architecture change"
     (E46: 13 jobs in Exp 3; T01: seen twice in the v4 test).
  6. budget_for() is shared with the two-stage pipeline, so stage 1 can be told the size limit (E46: 29 jobs in Exp 3
     had answers rejected as too large, 6 of them ended without code).
Records per job additionally: protected_rejects, smoke_oom_retries, unchanged_repairs.

Usage: python -m ab.gpt.util.knowledge.exp4.edit_alter --set SET.csv --vllm_url URL [--extra_info JSON] --name NAME
"""
import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from ab.gpt.act.alter.web_alter import build_jobs, family, load_rows
from ab.gpt.util.Const import new_nn_file, new_out_file
from ab.gpt.util.knowledge.exp2 import delta_alter as d2
from ab.gpt.util.knowledge.exp3 import edit_alter as ea

REPO = ea.REPO
Backend = ea.Backend
PROTECTED = ("supported_hyperparameters", "train_setup", "learn")

RULE_OUTPUT = ("8. forward() must return ONE tensor of shape (batch, number of classes), in training AND in evaluation "
               "mode. Never change what forward() returns: no tuples, namedtuples or auxiliary outputs.\n")
assert "\n\n@FMT_EXAMPLE@" in ea.PROMPT
PROMPT = ea.PROMPT.replace("\n\n@FMT_EXAMPLE@", "\n" + RULE_OUTPUT + "\n@FMT_EXAMPLE@")

REPAIR_RUN = """The model below fails when it is built and trained:

<nn>
@CODE@
</nn>

Error: @ERR@
@WHERE@@TRACE@
Fix the CAUSE of this error with a minimal change to THIS code exactly as shown. The error can come from a library \
call in your code (see the traceback): then change how you call it. Never delete or weaken error checks (raise, \
assert) and never disable parts of the model just to make the error disappear; if one of your own changes caused the \
error, adjust or undo that change. Keep the other intended architecture changes; never change \
supported_hyperparameters, train_setup or learn; forward() must return one tensor (batch, number of classes); never \
add new prm keys; define or import every name you use. @OUT@
"""

REPAIR_PROTECTED = """Your edit was rejected: it changed @NAMES@, which must stay exactly as in the baseline \
(only the architecture may change: layers, blocks, forward pass; forward() must return one tensor).

<nn>
@CODE@
</nn>

Make your intended architecture change again, against THIS code exactly as shown, without changing \
supported_hyperparameters, train_setup or learn. @OUT@
"""

REPAIR_UNCHANGED = """Your answer is identical to the code that fails, so the error is still there:

Error: @ERR@
@WHERE@@TRACE@
Change THIS code so that this error no longer occurs (the cause can be a library call in your code, see the \
traceback). Never delete or weaken error checks and never disable parts of the model; never change \
supported_hyperparameters, train_setup or learn. @OUT@

<nn>
@CODE@
</nn>
"""

REPAIR_TRUNC = ea.REPAIR_TRUNC
n_checks = ea.n_checks


def protected_dumps(code):
    """{name: [syntax-tree dumps]} of the protected functions (formatting and comments do not count)."""
    out = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in PROTECTED:
            out.setdefault(node.name, []).append(ast.dump(node))
    return {k: sorted(v) for k, v in out.items()}


def protected_changed(base, new):
    a, b = protected_dumps(base), protected_dumps(new)
    return [k for k in PROTECTED if a.get(k) != b.get(k)]


def smoke(code, row, workdir, timeout):
    """Same as exp2 smoke(), but runs exp4.smoke (adds `trace`)."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "smoke_nn.py").write_text(code)
    row.to_pickle(workdir / "dataframe.df")
    try:
        p = subprocess.run([sys.executable, "-m", "ab.gpt.util.knowledge.exp4.smoke", str(workdir / "smoke_nn.py"),
                            str(workdir / "dataframe.df")], cwd=REPO, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"TimeoutError: building and training 2 batches took more than {timeout}s",
                "where": "", "infra": False, "trace": ""}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("SMOKE_RESULT "):
            return json.loads(line[len("SMOKE_RESULT "):])
    tail = (p.stderr.strip().splitlines() or ["no output"])[-1]
    return {"ok": False, "error": f"SmokeCrash: {tail[:400]}", "where": "", "infra": True, "trace": ""}


def budget_for(code):
    """Change budget in added + removed lines (same formula as Experiments 2 and 3)."""
    return max(40, int(0.3 * len(code.splitlines())))


def is_oom(res):
    return "out of memory" in (res.get("error") or "").lower()


def fmt_where(res):
    return f"Location in the model code: {res['where']}\n" if res.get("where") else ""


def fmt_trace(res):
    return f"Traceback (last calls):\n{res['trace']}\n" if res.get("trace") else ""


def run_job(backend, job, out_dir, idx, a):
    row, base = job["row"], job["row"]["nn_code"]
    rec = dict(B=idx, nn=row["nn"], fam=family(row["nn"]), sample=job["k"], seed=job["seed"],
               extra_info=bool(job["extra"]))
    extra = (job.get("extra_header", d2.EXTRA_HEADER) + job["extra"].strip() + "\n\n") if job["extra"] else ""
    f = d2.FMT["whole"]
    budget = budget_for(base)
    msgs = d2.msgs_for(d2.fill(PROMPT, ACC=f"{float(row['accuracy']):.4f}", CODE=base, EXTRA=extra, BUDGET=budget,
                               FMT_RULES=f["rules"], FMT_EXAMPLE=f["example"]))
    work = out_dir / "work" / f"B{idx}"
    current, code, attempts, transcript = base, None, [], []
    n_prot = n_oom_retry = n_unchanged = 0
    last_fail = None  # last failing smoke result, for fix 5
    t0 = time.time()
    for rnd in range(a.max_repairs + 1):
        try:
            out, finish, n_in, max_out = backend.generate(msgs, job["seed"] + 1000 * rnd)
        except Exception as e:
            attempts.append({"round": rnd, "stage": "llm", "error": f"{type(e).__name__}: {e}"[:300]})
            break
        transcript.append({"round": rnd, "user": msgs[-1]["content"], "answer": out, "finish": finish,
                           "prompt_tokens": n_in, "max_tokens": max_out})
        msgs = msgs + [{"role": "assistant", "content": out}]  # repairs continue the conversation (Self-Debug)
        new, err = d2.apply_edit("whole", current, out, budget)
        if not new:
            cut = finish == "length" and d2.extract_whole(out) is None
            attempts.append({"round": rnd, "stage": "apply", "ok": False, "truncated": cut,
                             "error": "answer cut off at the output limit" if cut else err})
            if not cut and last_fail and "changes nothing" in (err or ""):  # fix 5
                n_unchanged += 1
                repair = d2.fill(REPAIR_UNCHANGED, ERR=last_fail["error"], WHERE=fmt_where(last_fail),
                                 TRACE=fmt_trace(last_fail), CODE=current, OUT=f["out"])
            else:
                repair = REPAIR_TRUNC if cut else d2.fill(d2.REPAIR_APPLY, ERR=err, CODE=current, HINT="",
                                                          OUT=f["out"])
            msgs = msgs + [{"role": "user", "content": repair}]
            continue
        changed = protected_changed(base, new)
        if changed:  # fix 1: enforce the rule; the rejected code is not kept
            n_prot += 1
            attempts.append({"round": rnd, "stage": "protected", "ok": False, "changed": changed})
            msgs = msgs + [{"role": "user", "content": d2.fill(REPAIR_PROTECTED, NAMES=" and ".join(changed),
                                                               CODE=current, OUT=f["out"])}]
            continue
        code = current = new
        res = smoke(code, row, work / f"r{rnd}", a.smoke_timeout)
        if is_oom(res):  # fix 4: another job filled the GPU -> wait and retry once
            n_oom_retry += 1
            attempts.append({"round": rnd, "stage": "smoke_oom_retry", **res})
            time.sleep(a.oom_wait)
            res = smoke(code, row, work / f"r{rnd}b", a.smoke_timeout)
        attempts.append({"round": rnd, "stage": "smoke", **res})
        if res.get("ok") or res.get("infra"):
            break
        last_fail = res
        msgs = msgs + [{"role": "user", "content": d2.fill(REPAIR_RUN, CODE=current, ERR=res["error"],
                                                           WHERE=fmt_where(res), TRACE=fmt_trace(res), OUT=f["out"])}]
    smokes = [x for x in attempts if x["stage"] == "smoke"]
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
               apply_failures=sum(x["stage"] == "apply" for x in attempts),
               protected_rejects=n_prot, smoke_oom_retries=n_oom_retry, unchanged_repairs=n_unchanged,
               truncated=sum(bool(x.get("truncated")) for x in attempts),
               llm_errors=sum(x["stage"] == "llm" for x in attempts),
               finish=[t["finish"] for t in transcript],
               smoke_first=smokes[0].get("ok") if smokes else None, smoke_last=smokes[-1].get("ok") if smokes else None,
               smoke_infra=any(x.get("infra") for x in smokes),
               removed_checks=max(0, n_checks(base) - n_checks(code)) if code else None,
               changed_lines=d2.changed_lines(base, code) if code else None, seconds=round(time.time() - t0, 1))
    return rec


def prompts_md5():
    return hashlib.md5((d2.SYSTEM + PROMPT + d2.EXTRA_HEADER + REPAIR_RUN + REPAIR_PROTECTED + REPAIR_UNCHANGED + d2.REPAIR_APPLY
                        + REPAIR_TRUNC + json.dumps(d2.FMT["whole"])).encode()).hexdigest()


def add_args(p):
    p.add_argument("--set", required=True)
    p.add_argument("--extra_info", default=None, help="JSON {nn or family: text}; omit for the baseline condition")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--max_repairs", type=int, default=2)
    p.add_argument("--smoke_timeout", type=float, default=300)
    p.add_argument("--oom_wait", type=float, default=120, help="seconds to wait before retrying an OOM smoke check")
    p.add_argument("--vllm_url", required=True)
    p.add_argument("--vllm_model", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--min_new_tokens", type=int, default=1024)
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--ctx_margin", type=int, default=64)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--dataset", default="cifar-10")
    p.add_argument("--only", default=None, help="comma-separated network names (for tests)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=str(Path.home() / "web_alter_runs_exp4"))
    p.add_argument("--name", default=None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    a = p.parse_args()
    extra = json.load(open(a.extra_info)) if a.extra_info else {}
    cond = "web" if a.extra_info else "base"
    out_dir = Path(a.out) / (a.name or f"{datetime.now():%Y%m%d-%H%M}_{cond}")
    if out_dir.exists():
        sys.exit(f"[edit_alter v4] {out_dir} exists -- choose another --name")
    rows = load_rows(a.set, "img-classification", a.dataset)
    jobs = build_jobs(rows, a.samples, a.seed, extra)
    if a.only:  # filter after pairing, so a test job is identical to the same job in the full run
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j["row"]["nn"] in keep]
        if not jobs:
            sys.exit(f"[edit_alter v4] none of {sorted(keep)} in {a.set}")
    jobs = jobs[:a.limit]
    backend = Backend(a.vllm_url, a.vllm_model, a)
    out_dir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        commit = "unknown"
    meta = dict(vars(a), experiment="exp4-base-v4", condition=cond, backend=backend.name, nn_gpt_commit=commit,
                nn_dataset=version("nn-dataset"), prompts_md5=prompts_md5(), networks=len({j["row"]["nn"] for j in jobs}),
                jobs=len(jobs), with_extra_info=sum(bool(j["extra"]) for j in jobs),
                started=datetime.now().isoformat(timespec="seconds"))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[edit_alter v4] {cond}: {len(jobs)} jobs, {meta['networks']} networks, {backend.name} -> {out_dir}", flush=True)
    n = {"code": 0, "smoke_ok": 0, "prot": 0}
    with open(out_dir / "results.jsonl", "w") as log, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, backend, j, out_dir, i, a) for i, j in enumerate(jobs)]
        for done, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            log.write(json.dumps(r) + "\n")
            log.flush()
            n["code"] += r["code"]
            n["smoke_ok"] += bool(r["smoke_last"])
            n["prot"] += r["protected_rejects"] > 0
            print(f"[{done}/{len(jobs)}] B{r['B']} {r['nn'][:26]:26s} code={r['code']} rounds={r['rounds']} "
                  f"prot={r['protected_rejects']} smoke first={r['smoke_first']} last={r['smoke_last']} "
                  f"changed={r['changed_lines']} {r['seconds']}s", flush=True)
    print(f"[edit_alter v4] done: {len(jobs)} jobs | code {n['code']} | smoke OK {n['smoke_ok']} | "
          f"jobs with a protected-function rejection {n['prot']} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
