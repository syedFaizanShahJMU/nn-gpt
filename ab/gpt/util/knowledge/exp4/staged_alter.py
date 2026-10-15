"""Two-stage pipeline v4 = Experiment 3 two-stage pipeline (exp3.staged_alter: stage 1 decides one change as JSON,
web suggestions only in the web condition; unchanged and imported) + base pipeline v4 for stage 2 (exp4.edit_alter:
protected functions enforced, output-contract rule, traceback in repair messages, OOM retry of the smoke check,
unchanged repair answers get the error again).
Stage 1 v4 additionally states the change budget of stage 2 (at most N added or removed lines, same formula), because
in Exp 3 stage 1 planned changes that stage 2 could not implement within the limit (E46: 29 jobs had answers rejected
as too large, 6 ended without code). Otherwise stage 1 is exactly exp3's (same prompts, retry, log, fields).

Usage: python -m ab.gpt.util.knowledge.exp4.staged_alter --set SET.csv --vllm_url URL [--extra_info JSON] --name NAME
"""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from ab.gpt.act.alter.web_alter import build_jobs, load_rows
from ab.gpt.util.knowledge.exp3 import staged_alter as s3
from ab.gpt.util.knowledge.exp3.comprehension import code_index, parse_json
from ab.gpt.util.knowledge.exp4 import edit_alter as ea

_OLD = "Do NOT write any code.\n"
assert _OLD in s3.STAGE1
STAGE1 = s3.STAGE1.replace(_OLD, "Do NOT write any code. The change must be implementable as a small edit: at most "
                                 "@BUDGET@ added or removed lines in total.\n")


def stage1(backend, job, out_dir, idx, a):
    """exp3 stage 1 with the budget sentence (code identical to exp3.staged_alter.stage1 otherwise)."""
    row = job["row"]
    sugg = s3.SUGG_BLOCK.replace("@SUGG@", job["extra"].strip()) if job["extra"] else ""
    user = (STAGE1.replace("@ACC@", f"{float(row['accuracy']):.4f}").replace("@BUDGET@", str(ea.budget_for(row["nn_code"])))
            .replace("@CODE@", row["nn_code"]).replace("@SUGG@", sugg))
    msgs = [{"role": "system", "content": s3.SYSTEM1}, {"role": "user", "content": user}]
    log, plan = [], None
    for attempt in range(2):
        try:
            out, finish, n_in, _ = backend.generate(msgs, job["seed"] + 500 + 1000 * attempt)
        except Exception as e:
            log.append({"attempt": attempt, "error": f"{type(e).__name__}: {e}"[:300]})
            break
        log.append({"attempt": attempt, "user": msgs[-1]["content"], "answer": out, "finish": finish, "prompt_tokens": n_in})
        js = parse_json(out)
        if isinstance(js, dict) and str(js.get("change") or "").strip():
            plan = js
            break
        msgs = msgs + [{"role": "assistant", "content": out}, {"role": "user", "content": s3.STAGE1_RETRY}]
    d = out_dir / "stage1"
    d.mkdir(exist_ok=True)
    (d / f"B{idx}.json").write_text(json.dumps({"nn": row["nn"], "web": bool(job["extra"]), "plan": plan, "log": log}, indent=1))
    info = dict(stage1_ok=plan is not None)
    if plan is None:
        return "", info
    classes, _ = code_index(row["nn_code"])
    used = str(plan.get("used_suggestion") or "none").strip()
    info.update(used_suggestion=None if used.lower() in ("none", "-1", "") else used[:120],
                plan_location_ok=str(plan.get("location") or "").split(".")[0].strip("` ") in classes,
                plan_modules=plan.get("modules"))
    mods = ", ".join(m for m in plan.get("modules") or [] if isinstance(m, str)) or "none"
    text = (f"Change: {str(plan.get('change')).strip()}\nWhere: {str(plan.get('location') or '').strip()}\n"
            f"Modules: {mods}\nWhy: {str(plan.get('reason') or '').strip()}")
    return text, info


def run_job(backend, job, out_dir, idx, a):
    plan, info = stage1(backend, job, out_dir, idx, a)
    rec = ea.run_job(backend, dict(job, extra=plan, extra_header=s3.PLAN_HEADER), out_dir, idx, a)
    rec.update(info, web=bool(job["extra"]), extra_info=bool(job["extra"]))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ea.add_args(p)
    a = p.parse_args()
    extra = json.load(open(a.extra_info)) if a.extra_info else {}
    cond = "staged_web" if a.extra_info else "staged_base"
    out_dir = Path(a.out) / (a.name or f"{datetime.now():%Y%m%d-%H%M}_{cond}")
    if out_dir.exists():
        sys.exit(f"[staged v4] {out_dir} exists -- choose another --name")
    rows = load_rows(a.set, "img-classification", a.dataset)
    jobs = build_jobs(rows, a.samples, a.seed, extra)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j["row"]["nn"] in keep]
        if not jobs:
            sys.exit(f"[staged v4] none of {sorted(keep)} in {a.set}")
    jobs = jobs[:a.limit]
    backend = ea.Backend(a.vllm_url, a.vllm_model, a)
    out_dir.mkdir(parents=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ea.REPO, text=True).strip()
    except Exception:
        commit = "unknown"
    md5 = hashlib.md5((s3.SYSTEM1 + STAGE1 + s3.SUGG_BLOCK + s3.STAGE1_RETRY + s3.PLAN_HEADER
                       + ea.prompts_md5()).encode()).hexdigest()
    meta = dict(vars(a), experiment="exp4-staged-v4", condition=cond, backend=backend.name, nn_gpt_commit=commit,
                nn_dataset=version("nn-dataset"), prompts_md5=md5, networks=len({j["row"]["nn"] for j in jobs}),
                jobs=len(jobs), with_extra_info=sum(bool(j["extra"]) for j in jobs),
                started=datetime.now().isoformat(timespec="seconds"))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[staged v4] {cond}: {len(jobs)} jobs, {meta['networks']} networks, {backend.name} -> {out_dir}", flush=True)
    n = {"code": 0, "smoke_ok": 0, "plan": 0, "used": 0, "prot": 0}
    with open(out_dir / "results.jsonl", "w") as log, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, backend, j, out_dir, i, a) for i, j in enumerate(jobs)]
        for done, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            log.write(json.dumps(r) + "\n")
            log.flush()
            n["code"] += r["code"]
            n["smoke_ok"] += bool(r["smoke_last"])
            n["plan"] += r["stage1_ok"]
            n["used"] += bool(r.get("used_suggestion"))
            n["prot"] += r["protected_rejects"] > 0
            print(f"[{done}/{len(jobs)}] B{r['B']} {r['nn'][:26]:26s} plan={r['stage1_ok']} "
                  f"used={str(r.get('used_suggestion'))[:30]!r} code={r['code']} rounds={r['rounds']} "
                  f"prot={r['protected_rejects']} smoke first={r['smoke_first']} last={r['smoke_last']} "
                  f"changed={r['changed_lines']} {r['seconds']}s", flush=True)
    print(f"[staged v4] done: {len(jobs)} jobs | plan {n['plan']} | uses a web suggestion {n['used']} | code {n['code']} "
          f"| smoke OK {n['smoke_ok']} | protected-function rejections in {n['prot']} jobs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
