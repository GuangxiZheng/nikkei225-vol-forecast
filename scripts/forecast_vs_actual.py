#!/usr/bin/env python
"""日次予測と実績の突合 + 大きく外れた時の原因調査。

毎日の予測を results/forecast_log.jsonl に記録し、実績(実現ボラ)が確定した
予測を results/forecast_scorecard.jsonl に突合していく。

原因分類 (事前宣言・|誤差|>15pt の big_miss のみ深掘りする):
  - ジャンプ型突発     : 対象窓内に 3σ(予測時点のσ22)超の一日 → 日次モデルの
                         構造的限界。ジャンプ特徴量(maxz5)は検証で有意にならず
  - 市場の含みが織り込み済み : 予測時点のVIが実績に近い → 予測が市場を読み切れず
                         (パラメータ不足 or ロジック問題の疑い・月次チェックで精査)
  - 市場全体が予想外   : VIも実績を下回っていた → 真のショック(誰にも予測不能)
  - 過大評価           : 実績が予測を大きく下回る (静穏化/含みの巻き戻り)
  - 系統的バイアス     : 直近26件(h=5)の符号付き誤差の平均が±3pt超 →
                         パラメータ不足/ロジック問題の疑いとして月次チェックへ

判定の目安: 80%区間内なら正常(実カバレッジ65〜71%なので区間外は約3割ある)、
|誤差|>15pt (合意週の平均誤差6.8ptの約2.2倍) を big_miss として扱う。

使い方: .venv/bin/python scripts/forecast_vs_actual.py
  (--log PATH / --scorecard PATH でテスト用の書き先を変えられる)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vol_backtest import build_features, load_data  # noqa: E402
from volmodel import COLS, target_h  # noqa: E402

LOG = ROOT / "results" / "forecast_log.jsonl"
SCORECARD = ROOT / "results" / "forecast_scorecard.jsonl"
BIG_MISS = 15.0
BIAS_PT, BIAS_N = 3.0, 26


def jload(path):
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        return []


def main():
    args = sys.argv[1:]
    log_path = Path(args[args.index("--log") + 1]) if "--log" in args else LOG
    sc_path = Path(args[args.index("--scorecard") + 1]) if "--scorecard" in args else SCORECARD

    latest = json.loads((ROOT / "results" / "model_latest.json").read_text())
    asof = latest["data_asof"]

    # 1) 本日の予測を記録 (同じ data_asof は1件だけ)
    entries = jload(log_path)
    if not any(e["data_asof"] == asof for e in entries):
        entry = {**latest, "run_at": pd.Timestamp.now().isoformat(timespec="seconds")}
        entries.append(entry)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 2) 実績が確定した予測を突合
    df = build_features(load_data())
    by_date = df.set_index(df["date"].dt.strftime("%Y-%m-%d"))
    done = {(e["data_asof"], e["h"]) for e in jload(sc_path)}
    rows, newly = [], []
    for e in entries:
        if e["data_asof"] not in by_date.index:
            continue
        i = by_date.index.get_loc(e["data_asof"])
        vi_t = float(by_date["vi"].iloc[i])
        sig22 = float(by_date["ret"].iloc[max(0, i - 21):i + 1].std())
        for p in e["path_forecast"]:
            h = p["h"]
            if (e["data_asof"], h) in done:
                continue
            end = i + h
            if end >= len(df) or by_date["ret"].iloc[i + 1:end + 1].isna().any():
                continue  # 実績未確定
            actual = float(target_h(by_date, h).iloc[i])
            pred = float(p["ensemble"])
            err = actual - pred
            inside = p["pi80_lo"] <= actual <= p["pi80_hi"]
            row = {"data_asof": e["data_asof"], "h": h, "pred": round(pred, 2),
                   "actual": round(actual, 2), "err": round(err, 2),
                   "inside_pi80": bool(inside), "vi_t": round(vi_t, 2),
                   "har": p["har_vi"], "garch": p["garch"]}
            if abs(err) > BIG_MISS:
                win = by_date["ret"].iloc[i + 1:end + 1]
                jump = bool((win.abs() > 3 * sig22).any())
                if err > 0:
                    cause = ("ジャンプ型突発(3σ超の一日) — 日次モデルの構造的限界"
                             if jump else
                             "市場の含み(VI)が織り込み済み — 予測が読み切れず(パラメータ不足/"
                             "ロジック問題の疑い・月次チェックで精査)" if vi_t > actual * 0.8 else
                             "市場全体が予想外のショック(真のサプライズ)")
                else:
                    cause = ("含みの巻き戻り型の過大評価" if vi_t < pred * 0.8
                             else "静穏化の過大評価")
                row["cause"] = cause
                row["jump_in_window"] = jump
            rows.append(row)
            newly.append(row)
    if rows:
        with sc_path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3) 要約
    ev = jload(sc_path)
    print(f"予測の振り返り: 記録済み予測 {len(entries)}件 / 突合済み {len(ev)}件"
          f" (今回新規 {len(newly)}件)")
    if not ev:
        print("  実績確定の予測はまだありません (初回評価は明日以降に順次開始)")
        return
    for r in newly:
        tag = "80%区間内" if r["inside_pi80"] else "区間外"
        line = (f"  [{r['data_asof']} h={r['h']}] 予測 {r['pred']:.1f}% → "
                f"実績 {r['actual']:.1f}% (誤差 {r['err']:+.1f}pt, {tag})")
        print(line + (f"\n    → 原因: {r['cause']}" if "cause" in r else ""))
    h5 = [e for e in ev if e["h"] == 5][-BIAS_N:]
    if h5:
        if len(h5) < BIAS_N:
            print(f"  バイアス判定はh=5の実績が{BIAS_N}件貯まってから (現在 {len(h5)}件)")
        else:
            bias = float(np.mean([e["err"] for e in h5]))
            note = (" — 系統的バイアスの疑い(パラメータ不足 or ロジック問題)・月次チェックで精査"
                    if abs(bias) > BIAS_PT else " (基準 ±3pt 以内)")
            print(f"  直近{len(h5)}件(h=5)の平均符号誤差 {bias:+.1f}pt{note}")


if __name__ == "__main__":
    main()
