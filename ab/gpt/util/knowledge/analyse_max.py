"""Best-model view of web_alter / delta_alter runs (advisor, 13 Oct: report average AND maximum
accuracy; compare against the original networks given as examples).

Reads only results.jsonl and the Eval outputs (via analyse_runs.collect); safe while Eval is running.

Usage:
  python -m ab.gpt.util.knowledge.analyse_max RUN_DIR
  python -m ab.gpt.util.knowledge.analyse_max BASE_RUN_DIR WEB_RUN_DIR
Writes RUN_DIR/per_network.csv.

Definitions (fixed before looking at web results):
  acc0      accuracy of a generation, failures (error/no_code/verify_fail/low_acc) counted as 0
  best      per network: highest accuracy among its trained (status ok) generations, 0 if none
  gain      best - accuracy of the original network (the example given to the LLM)
  complete  network has no pending generation
"""
import sys
from pathlib import Path

import pandas as pd

from ab.gpt.util.knowledge.analyse_runs import binom_two_sided, collect


def fill_orig(df, run_dir):
    """Original accuracy also for generations without code (their folder is failed/, not synth_nn/)."""
    for i, r in df[df.orig_acc.isna()].iterrows():
        f = Path(run_dir) / "failed" / f"B{r.B}" / "dataframe.df"
        if f.exists():
            df.at[i, "orig_acc"] = float(pd.read_pickle(f).get("accuracy"))
    return df


def per_network(df):
    d = df.copy()
    d["acc0"] = d.acc.where(d.status == "ok", 0.0)
    g = d.groupby("nn")
    ok = d[d.status == "ok"].groupby("nn").acc
    idx = g.size().index
    out = pd.DataFrame({
        "fam": g.fam.first(),
        "n": g.size(),
        "pending": d.status.eq("pending").groupby(d.nn).sum(),
        "n_ok": d.status.eq("ok").groupby(d.nn).sum(),
        "mean_ok": ok.mean().reindex(idx),
        "mean_acc0": d[d.status != "pending"].groupby("nn").acc0.mean().reindex(idx),
        "best": ok.max().reindex(idx).fillna(0.0),
        "orig": g.orig_acc.first(),  # first non-null value per network
    })
    out["gain"] = out.best - out.orig
    out["complete"] = out.pending.eq(0)
    return out.sort_values("gain")


def report(name, df, net):
    done = df[df.status != "pending"]
    ok = done[done.status == "ok"]
    c = net[net.complete]
    print(f"== {name}: {len(net)} networks ({len(c)} complete) | {len(done)}/{len(df)} generations evaluated")
    if len(ok):
        top = ok.loc[ok.acc.idxmax()]
        print(f"   average: mean acc of trained models {ok.acc.mean():.4f} | "
              f"mean acc incl. failures as 0 {done.acc.where(done.status == 'ok', 0).mean():.4f}")
        print(f"   maximum: best single model {top.acc:.4f} ({top.nn}, B{top.B}, original {top.orig_acc:.4f})")
    if len(c):
        print(f"   per network (complete only): mean best {c.best.mean():.4f} | mean original {c.orig.mean():.4f} | "
              f"mean gain {c.gain.mean():+.4f} | best beats original in {(c.gain > 0).sum()}/{len(c)} networks")


def compare(nb, nw):
    m = nb.join(nw, lsuffix="_b", rsuffix="_w", how="inner")
    m = m[m.complete_b & m.complete_w]
    if not len(m):
        print("\n== paired best-model comparison: no network complete in both conditions yet")
        return
    m["d"] = m.best_w - m.best_b
    w, l = (m.d > 0).sum(), (m.d < 0).sum()
    print(f"\n== paired best-model comparison: {len(m)} networks complete in both conditions")
    print(f"   mean best: base {m.best_b.mean():.4f} | web {m.best_w.mean():.4f} | mean diff {m.d.mean():+.4f}, "
          f"median {m.d.median():+.4f}")
    print(f"   web best higher in {w}, lower in {l}, tie {len(m) - w - l} (sign test p = {binom_two_sided(min(w, l), w + l):.3f})")
    try:
        from scipy.stats import wilcoxon
        if (m.d != 0).sum() >= 6:
            print(f"   Wilcoxon signed-rank on per-network best p = {wilcoxon(m.d).pvalue:.3f}")
    except ImportError:
        pass
    print(f"   beats original: base {(m.gain_b > 0).sum()}/{len(m)} | web {(m.gain_w > 0).sum()}/{len(m)} networks")
    print(m[["best_b", "best_w", "orig_b", "d"]].rename(columns={"orig_b": "orig"}).sort_values("d").round(4).to_string())


if __name__ == "__main__":
    runs = sys.argv[1:3]
    nets = []
    for r in runs:
        df = fill_orig(collect(r), r)
        net = per_network(df)
        net.to_csv(Path(r) / "per_network.csv")
        report(Path(r).name, df, net)
        nets.append(net)
    if len(nets) == 2:
        compare(*nets)
