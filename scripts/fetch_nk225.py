#!/usr/bin/env python
"""nikkei225jp.com から 日経225先物の日足OHLC(1990年〜)と日経VI(2016年〜)を取得。

- RL111_1990.json : var R111 = [[ms, open, high, low, close, ""], ...] (日経225先物)
- S161.json       : var S161 = [[ms, VI], ...] (日経平均VI・日次)
※ Referer ヘッダ必須(ホットリンク対策)。SSRF対策として URL はリテラル定数のみ
  (変数経由の URL 生成なし・リダイレクト無効・https のみ)。

ラベル仕様 (2026-10検証確定): 全バーのmsは「実セッションの前日」のUTC正午夜。
vol_backtest.load_data が +1日して実日付に復元する。

フォールバック: プライマリが直近平日より遅れている場合のみ、Yahoo Finance ^N225
の確定足で不足分を補う (vol_backtest と同じ+1日規約でms化・今日の未確定足は含めない)。
毎回プライマリから再構築するため重複は発生しない (冪等)。
"""
import calendar
import datetime
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


def real_dates_of(fut):
    """バーのmsから実日付 (ラベル+1日) の一覧を返す。"""
    utc = datetime.timezone.utc
    return sorted(datetime.datetime.fromtimestamp(r[0] / 1000, utc).date()
                  + datetime.timedelta(days=1) for r in fut)


def yahoo_daily():
    """Yahoo ^N225 の日足 (実日付→OHLC辞書)。未確定の今日の足は呼び出し側で除外。"""
    r = requests.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/%5EN225?range=1mo&interval=1d",
        headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    tz = datetime.timezone(datetime.timedelta(hours=9))
    quote = j["indicators"]["quote"][0]
    keys = ["open", "high", "low", "close"]
    out = {}
    for t, vals in zip(j["timestamp"], zip(*[quote[k] for k in keys])):
        q = dict(zip(keys, vals))
        if q["close"] is None:
            continue
        d = datetime.datetime.fromtimestamp(t, tz).date().isoformat()
        out[d] = q
    return out


def expected_latest_session():
    """直近の確定済み平日 (今日が平日なら今日=夜に確定。今は未確定でも対象日)。"""
    d = datetime.date.today()
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d


def fallback_tail(known_dates):
    """primary に無い確定足を Yahoo から作る (実日付-1日のmsで出力)。"""
    y = yahoo_daily()
    last_p = max(known_dates).isoformat()
    today = datetime.date.today().isoformat()
    tail = []
    for d in sorted(y):
        if d <= last_p or d >= today:
            continue
        q = y[d]
        label_ms = calendar.timegm(date_minus_one(d).timetuple()) * 1000
        tail.append([label_ms, q["open"], q["high"], q["low"], q["close"], ""])
    return tail


def date_minus_one(iso_date):
    return datetime.date.fromisoformat(iso_date) - datetime.timedelta(days=1)


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

    # フォールバック: プライマリの最新実日付が直近平日より古い時だけ Yahoo で補完
    known = real_dates_of(fut)
    last_p = max(known)
    expected = expected_latest_session()
    if last_p < expected:
        tail = fallback_tail(known)
        if tail:
            fut = sorted(fut + tail, key=lambda r: r[0])
            newest = datetime.datetime.fromtimestamp(
                tail[-1][0] / 1000, datetime.timezone.utc).date() + datetime.timedelta(days=1)
            print(f"fallback: Yahoo ^N225 から {len(tail)}本を補完 (実日付〜{newest})")
        else:
            print("fallback: 補完する確定足なし (ソースは1営業日遅れ・許容範囲)")
    else:
        print("primary: 最新営業日まで取得済み")

    (DATA / "n225_fut_daily.json").write_text(json.dumps(fut), encoding="utf-8")
    (DATA / "nkvi_daily.json").write_text(json.dumps(vi), encoding="utf-8")
    print(f"futures OHLC: {len(fut)} rows / VI: {len(vi)} rows")


if __name__ == "__main__":
    main()
