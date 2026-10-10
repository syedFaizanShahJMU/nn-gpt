"""Collect Eval results of web_alter runs and compare two conditions (paired).

Usage:
  python -m ab.gpt.util.knowledge.analyse_runs RUN_DIR                # status of one run
  python -m ab.gpt.util.knowledge.analyse_runs BASE_RUN_DIR WEB_RUN_DIR  # + paired comparison
Writes RUN_DIR/eval_results.csv. Safe to run while Eval is still running (unfinished = pending).

Status per generation: ok (trained, accuracy), low_acc (trained, rejected by LEMUR's minimum accuracy),
error (training failed), verify_fail (code check failed), no_code, pending (not evaluated yet).
"""
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd


def collect(run_dir):
    run_dir = Path(run_dir)
    rows = []
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        d = run_dir / "synth_nn" / f"B{r['B']}"
        status, acc, err = "pending", None, ""
        if not r.get("code"):
            status = "no_code"
        elif (d / "eval_info.json").exists():
            acc = json.loads((d / "eval_info.json").read_text())["eval_results"]["accuracy"]
            status = "ok"
        elif (d / "eval_verification_failed.txt").exists():
            status = "verify_fail"
        elif (d / "error.txt").exists():
            err = (d / "error.txt").read_text(errors="replace").strip().splitlines()[0][:160]
            m = re.search(r"Accuracy is too low: ([0-9]+\.[0-9]+)", err)
            status, acc = ("low_acc", float(m.group(1))) if m else ("error", None)
        orig = None
        if (d / "dataframe.df").exists():
            orig = float(pd.read_pickle(d / "dataframe.df").get("accuracy"))
        rows.append(dict(B=r["B"], nn=r["nn"], fam=r["fam"], sample=r["sample"], seed=r["seed"],
                         parses=r.get("parses"), status=status, acc=acc, orig_acc=orig, error=err))
    df = pd.DataFrame(rows).sort_values(["nn", "sample"])
    df.to_csv(run_dir / "eval_results.csv", index=False)
    return df


def summary(name, df):
    n, st = len(df), df.status.value_counts()
    done = n - st.get("pending", 0)
    ok = df[df.status == "ok"]
    print(f"== {name}: {n} generations | evaluated {done} | " +
          " | ".join(f"{k} {v}" for k, v in st.items()))
    if done:
        print(f"   trained OK: {len(ok)}/{done} = {len(ok) / done:.1%} of evaluated")
    if len(ok):
        print(f"   accuracy (OK only): mean {ok.acc.mean():.4f}, median {ok.acc.median():.4f}, max {ok.acc.max():.4f}"
              f" | beat original: {(ok.acc > ok.orig_acc).sum()}/{len(ok)}")
    errs = df[df.status == "error"].error.str.extract(r"^(\w+(?:Error|Exception))")[0].value_counts()
    if len(errs):
        print("   error types: " + ", ".join(f"{k} {v}" for k, v in errs.head(6).items()))


def binom_two_sided(k, n):
    if n == 0:
        return float("nan")
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(base, web):
    m = base.merge(web, on=["nn", "sample"], suffixes=("_b", "_w"))
    both_done = m[(m.status_b != "pending") & (m.status_w != "pending")]
    ok_b, ok_w = both_done.status_b.eq("ok"), both_done.status_w.eq("ok")
    print(f"\n== paired: {len(both_done)} pairs evaluated in both conditions")
    print(f"   trained OK: base {ok_b.sum()} | web {ok_w.sum()} | "
          f"only base {(ok_b & ~ok_w).sum()} | only web {(~ok_b & ok_w).sum()} "
          f"(exact McNemar p = {binom_two_sided((ok_b & ~ok_w).sum(), (ok_b ^ ok_w).sum()):.3f})")
    p = both_done[ok_b & ok_w].copy()
    if not len(p):
        return
    p["diff"] = p.acc_w - p.acc_b
    w, l = (p["diff"] > 0).sum(), (p["diff"] < 0).sum()
    print(f"   accuracy, {len(p)} pairs where both trained: mean diff (web - base) {p['diff'].mean():+.4f}, "
          f"median {p['diff'].median():+.4f} | web better {w}, worse {l}, tie {len(p) - w - l}")
    try:
        from scipy.stats import wilcoxon
        if len(p) >= 6:
            print(f"   Wilcoxon signed-rank (pairs) p = {wilcoxon(p['diff']).pvalue:.3f}")
    except ImportError:
        pass
    net = p.groupby("nn")["diff"].mean()  # samples of one network are not independent
    wn, ln = (net > 0).sum(), (net < 0).sum()
    print(f"   per network ({len(net)} networks, mean over its pairs): web better {wn}, worse {ln} "
          f"(sign test p = {binom_two_sided(min(wn, ln), wn + ln):.3f})")
    print(net.sort_values().round(4).to_string())


if __name__ == "__main__":
    runs = [collect(r) for r in sys.argv[1:3]]
    for r, df in zip(sys.argv[1:3], runs):
        summary(Path(r).name, df)
    if len(runs) == 2:
        compare(*runs)
