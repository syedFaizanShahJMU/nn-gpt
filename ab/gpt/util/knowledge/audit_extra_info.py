"""Audit gist suggestions (extra_info_details.jsonl) with rules fixed BEFORE looking at the results.

Flags per network:
  TRAIN    the reply mentions a training / optimizer / data term (excluded by the question)
  PRESENT  the reply asks to ADD a block the network already has (from its block signature)
  EMPTY    no reply (error or empty)
Usage: python -m ab.gpt.util.knowledge.audit_extra_info OUT_DIR
"""
import json
import re
import sys
from pathlib import Path

TRAIN_TERMS = re.compile(r"weight decay|learning rate|\blr\b|optimi[sz]er|\bsgd\b|\badam|augment|epoch|batch size|"
                         r"schedul|warm-?up|label smoothing|mixup|cutmix|cutout|pre-?train|transfer learning",
                         re.IGNORECASE)


def audit(rec):
    reply = rec.get("reply") or ""
    flags = []
    if not reply:
        return ["EMPTY"], []
    if TRAIN_TERMS.search(reply):
        flags.append("TRAIN")
    present = []
    for clause in re.split(r"[.;]|\bthen\b", reply, flags=re.IGNORECASE):
        if re.search(r"\badd", clause, re.IGNORECASE):
            present += [b.rstrip("*") for b in rec["signature"]
                        if re.search(rf"\b{re.escape(b.rstrip('*'))}\b", clause, re.IGNORECASE)]
    if present:
        flags.append("PRESENT")
    return flags, sorted(set(present))


def main():
    out_dir = Path(sys.argv[1])
    latest = {}
    for line in (out_dir / "extra_info_details.jsonl").read_text().splitlines():
        rec = json.loads(line)
        latest[rec["nn"]] = rec  # a retry overwrites the failed attempt
    counts = {"TRAIN": 0, "PRESENT": 0, "EMPTY": 0}
    rows = []
    for name, rec in latest.items():
        flags, present = audit(rec)
        for f in flags:
            counts[f] += 1
        rows.append({"nn": name, "flags": flags, "present": present, "reply": rec.get("reply"),
                     "seconds": rec.get("seconds")})
        print(f"{name[:28]:28s} {','.join(flags) or 'ok':14s} {('adds present: ' + ', '.join(present)) if present else ''}")
        print(f"    {rec.get('reply') or rec.get('error')}")
    n = len(latest)
    clean = sum(1 for r in rows if not r["flags"])
    print(f"\n{n} networks | clean {clean} | TRAIN {counts['TRAIN']} | PRESENT {counts['PRESENT']} | EMPTY {counts['EMPTY']}")
    (out_dir / "audit.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
