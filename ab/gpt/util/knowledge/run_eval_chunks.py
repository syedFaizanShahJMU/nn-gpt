"""Evaluate web_alter runs in small chunks, so results are saved as they finish.

Eval.py writes eval_info.json / error.txt only after ALL models of one call are done, so a crash
or reboot loses everything evaluated so far. This driver calls Eval on chunks of model folders
(symlinked into RUN_DIR/_eval_chunk), skips folders that already have a result (resumable), and
waits until a run's generation has finished before evaluating it.
A folder that still has no result after 2 Eval attempts gets error.txt "EvalNoResultError".

Usage: python -m ab.gpt.util.knowledge.run_eval_chunks RUN_DIR [RUN_DIR ...] [--chunk 10]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DONE = ("eval_info.json", "error.txt", "eval_verification_failed.txt")


def has_result(d):
    return any((d / f).exists() for f in DONE)


def generation_finished(run):
    jobs = json.loads((run / "run_meta.json").read_text())["jobs"]
    res = run / "results.jsonl"
    return res.exists() and len(res.read_text().splitlines()) >= jobs


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--chunk", type=int, default=10)
    a = p.parse_args()
    for run in map(Path, a.runs):
        while not generation_finished(run):
            log(f"{run.name}: generation not finished, waiting")
            time.sleep(300)
        attempts = {}
        while True:
            todo = sorted((d for d in (run / "synth_nn").glob("B*") if d.is_dir() and not has_result(d)),
                          key=lambda d: int(d.name[1:]))
            for d in [d for d in todo if attempts.get(d.name, 0) >= 2]:
                (d / "error.txt").write_text("EvalNoResultError: no result after 2 Eval attempts\n")
                log(f"{run.name}/{d.name}: no result after 2 attempts -> error.txt")
            todo = [d for d in todo if attempts.get(d.name, 0) < 2]
            if not todo:
                break
            chunk = todo[:a.chunk]
            cdir = run / "_eval_chunk"
            shutil.rmtree(cdir, ignore_errors=True)  # removes only the symlinks, never the targets
            cdir.mkdir()
            for d in chunk:
                (cdir / d.name).symlink_to(d.resolve())
                attempts[d.name] = attempts.get(d.name, 0) + 1
            left = len(todo)
            log(f"{run.name}: evaluating {', '.join(d.name for d in chunk)} ({left} without result)")
            with open(run / "eval.log", "a") as f:
                subprocess.run([sys.executable, "-u", "-m", "ab.gpt.act.eval.Eval", "--custom_synth_dir", str(cdir),
                                "--no-save_to_db"], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
            n_ok = sum((d / "eval_info.json").exists() for d in chunk)
            log(f"{run.name}: chunk done, trained OK {n_ok}/{len(chunk)}")
        shutil.rmtree(run / "_eval_chunk", ignore_errors=True)
        log(f"{run.name}: all models have a result")


if __name__ == "__main__":
    main()
