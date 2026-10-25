#!/usr/bin/env python
"""FRED (米連邦準備) から海外市場データを取得: data/external_daily.json に保存。

- VIXCLS : 米VIX 終値 (1990〜)
- DEXJPUS: USD/JPY 終値 (1971〜・FREDの更新は1-5営業日遅れ)
- SP500  : S&P500 終値 (FREDは直近10年のみ)

保存形式: {"YYYY-MM-DD": {"spx":..., "vix":..., "usdjpy":...}} (その日に値のない項目は省略)
stooq は404で利用不可のため FRED を採用 (2026-10確認)。
使い方: .venv/bin/python scripts/fetch_external.py
"""
import csv
import io
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "external_daily.json"
SERIES = {"SP500": "spx", "VIXCLS": "vix", "DEXJPUS": "usdjpy"}


def fetch(sid: str) -> dict:
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                     timeout=60)
    r.raise_for_status()
    out = {}
    for row in csv.DictReader(io.StringIO(r.text)):
        v = row[sid]
        if v not in ("", ".", None):
            out[row["observation_date"]] = float(v)
    return out


def main():
    series = {col: fetch(sid) for sid, col in SERIES.items()}
    dates = sorted(set().union(*[set(s) for s in series.values()]))
    merged = {d: {col: s[d] for col, s in series.items() if d in s}
              for d in dates}
    DATA.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(merged, separators=(",", ":")))
    tail = dates[-1]
    print(f"external data: {len(dates)} days ({dates[0]}〜{tail}) "
          f"/ spx:{len(series['spx'])} vix:{len(series['vix'])} "
          f"usdjpy:{len(series['usdjpy'])} rows")


if __name__ == "__main__":
    main()
