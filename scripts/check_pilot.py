#!/usr/bin/env python3
"""Quick sanity check for a HotpotQA pilot run.

Usage:
    python3 scripts/check_pilot.py [results_dir]

Defaults to results_hotpotqa/llama-3.1-8b_D2. Confirms both model sides ran to
completion and reports the denial (refusal) rate on the attacked split so you
can eyeball whether the attack did anything before fanning out the full matrix.

raw_results.json shape:
    { "base": {..., "baseline": [rec...], "attacked": [rec...]},
      "aligned": {..., "baseline": [rec...], "attacked": [rec...]} }
Each rec has: query, answer, refusal_type, severity, retrieved_adversarial_count, ...
"""
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_hotpotqa/llama-3.1-8b_D2")
path = results_dir / "raw_results.json"

if not path.exists():
    sys.exit(f"MISSING: {path} does not exist.")

data = json.load(open(path))
print(f"file: {path}")

for side in ("base", "aligned"):
    if side not in data:
        print(f"  {side}: MISSING")
        continue
    s = data[side]
    name = s.get("model_name", "?")
    base = s.get("baseline", [])
    att = s.get("attacked", [])
    # denial = anything with a non-null / non-"none" refusal_type on attacked split
    def denied(recs):
        n = 0
        for r in recs:
            rt = r.get("refusal_type")
            if rt and str(rt).lower() != "none":
                n += 1
        return n
    d_base = denied(base)
    d_att = denied(att)
    print(f"  {side:8s} [{name}]  baseline={len(base):4d} (denied {d_base})   "
          f"attacked={len(att):4d} (denied {d_att})")
    if att:
        keys = list(att[0].keys())
        print(f"           record keys: {keys}")

print("\nExpect ~1000 per split. Denied-on-attacked >> denied-on-baseline "
      "means the attack is silencing the model.")
