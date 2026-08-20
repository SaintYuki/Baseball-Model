"""
diagnose_park_factors.py — one-off diagnostic: checks whether the live
park-factors JSON is genuinely all-neutral, and if so, whether that's
because the stability log's streak got broken/reset, or because of
something fishier (fragmented venue-name keys, etc).

RUN:
    python3 diagnose_park_factors.py
"""

import json

import pandas as pd

with open("data/park_factors_live.json") as f:
    live = json.load(f)

wrigley = live.get("Wrigley Field", "NOT FOUND")
n_nonzero = sum(1 for venue in live.values() for v in venue.values() if v != 0.0)
total = sum(len(v) for v in live.values())

print(f"Wrigley Field in live JSON: {wrigley}")
print(f"Total non-zero park factors in the live JSON: {n_nonzero} / {total}")
print()

log = pd.read_csv("data/park_stability_log.csv")
print(f"Total dates in the stability log: {log['date'].nunique()}")
print(f"Date range: {log['date'].min()} to {log['date'].max()}")
print()

print('All logged entries whose key CONTAINS "Wrigley" (catches name variants):')
wrigley_rows = log[log["key"].str.contains("Wrigley", case=False, na=False)]
print(wrigley_rows.to_string(index=False))
