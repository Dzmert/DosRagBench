#!/usr/bin/env python3
"""Quick sanity check for a HotpotQA pilot run.

Usage:
    python3 scripts/check_pilot.py [results_dir]

Defaults to results_hotpotqa/llama-3.1-8b_D2. Prints record count, the keys on
the first record, and whether gold_doc_ids is present (confirms --keep-all-gold
carried through the run).
"""
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_hotpotqa/llama-3.1-8b_D2")
path = results_dir / "raw_results.json"

if not path.exists():
    sys.exit(f"MISSING: {path} does not exist.")

data = json.load(open(path))

# Records may be a list, or a dict with a 'results'/side split -- handle both.
if isinstance(data, dict):
    print("top-level keys:", list(data.keys()))
    # find the first list of records we can inspect
    records = None
    for v in data.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            records = v
            break
    if records is None:
        sys.exit("Could not find a list of records inside the dict.")
else:
    records = data

print(f"{len(records)} records")
first = records[0]
print("record keys:", list(first.keys()))
print("has gold_doc_ids:", "gold_doc_ids" in first)
if "gold_doc_ids" in first:
    print("example gold_doc_ids:", first["gold_doc_ids"])
