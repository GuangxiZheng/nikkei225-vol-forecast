#!/usr/bin/env python
"""nikkei225jp.com から 日経225先物の日足OHLC(1990年〜)と日経VI(2016年〜)を取得。

- RL111_1990.json : var R111 = [[ms, open, high, low, close, ""], ...] (日経225先物)
- S161.json       : var S161 = [[ms, VI], ...] (日経平均VI・日次)
※ Referer ヘッダ必須(ホットリンク対策)。SSRF対策として URL はリテラル定数のみ
  (変数経由の URL 生成なし・リダイレクト無効・https のみ)。
"""
import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HDR = {"User-Agent": "nk225-vol-analysis-research/0.1",
       "Referer": "https://nikkei225jp.com/data/vix.php"}


def parse_array(text: str):
    """'var XXX = [...];' 形式から配列を取り出す。"""
    m = re.search(r"=\s*(\[.*\])\s*;?\s*$", text, re.S)
    if not m:
        raise ValueError("配列が見つからない")
    return json.loads(m.group(1))


def main():
    DATA.mkdir(exist_ok=True)

    # URLは完全リテラル(固定2件のみ・他ホストへの接続経路は存在しない)
    r1 = requests.get(
        "https://nikkei225jp.com/_data/_nfsDATA/json_DAY/RL111_1990.json",
        headers=HDR, timeout=60, allow_redirects=False)
    r1.raise_for_status()
    fut = parse_array(r1.text)

    r2 = requests.get(
        "https://nikkei225jp.com/_data/_nfsDATA/json_DAY/S161.json",
        headers=HDR, timeout=60, allow_redirects=False)
    r2.raise_for_status()
    vi = parse_array(r2.text)

    (DATA / "n225_fut_daily.json").write_text(json.dumps(fut), encoding="utf-8")
    (DATA / "nkvi_daily.json").write_text(json.dumps(vi), encoding="utf-8")
    print(f"futures OHLC: {len(fut)} rows / VI: {len(vi)} rows")


if __name__ == "__main__":
    main()
