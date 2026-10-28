#!/usr/bin/env python
"""買いシグナル週の実ストラドル・バックテスト (プレミアム/スキュー/コスト込み)。

long_vol_report.md で点灯した買いシグナル週について、
「金曜終値で ATM ストラドル(コール+プット)を買い、5/10営業日後に手仕舞う」取引を
Black-Scholes で再現し、実取引に近い手取り損益を計算する。

近似の明示 (すべて感度分析でカバー):
  - IV = 予測時点のVI (ATM近似) + スキュー上乗せ 0/10/20%
  - 満期 14暦日のストラドル。手仕舞い時のIVはその日の実VI
  - 往復コスト (スプレッド+手数料) をプレミアムの 1/2/3%
  - 金利・配当・早期行使は無視 (日経オプションは欧州型・短期ATMなら影響小)
  - ボラは百分率→小数に換算してBSへ渡す。手仕舞いは日次フレームで正確に
    5/10営業日後の行を取る (金曜のみの表を使わない)

使い方: .venv/bin/python scripts/straddle_backtest.py
"""
import sys
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vol_backtest import EVAL_FROM, build_features, load_data  # noqa: E402
from volmodel import target_h, walkforward_direct  # noqa: E402

REPORT = ROOT / "results" / "straddle_backtest_report.md"
TAU = 200.0


def bs_straddle(S, K, T, vol, r=0.0):
    """ATM付近のストラドル理論価格 (call+put)。volは小数。"""
    def N(x):
        return 0.5 * (1 + erf(x / sqrt(2)))
    T = max(T, 1 / 365)
    vol = max(vol, 0.01)
    d1 = (log(S / K) + (r + vol ** 2 / 2) * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)
    call = S * N(d1) - K * exp(-r * T) * N(d2)
    put = K * exp(-r * T) * N(-d2) - S * N(-d1)
    return call + put


def main():
    df = build_features(load_data())
    df["y5"] = target_h(df, 5)
    df["pred"] = walkforward_direct(df, 5)
    ev = df[(df["date"] >= EVAL_FROM) & (df["date"].dt.dayofweek == 4)].copy()
    ev = ev.dropna(subset=["pred", "vi", "rv_fwd"]).reset_index(drop=True)
    ev["spread"] = ev["pred"] ** 2 - ev["vi"] ** 2

    lines = [
        "# 実ストラドル・バックテスト (買いシグナル週・コスト込み)",
        "",
        "取引の再現: シグナル金曜の終値で ATM ストラドルを購入 (満期14暦日) →",
        "5営業日後 / 10営業日後 (日次フレームで厳密に计数) に IV=その日の実VI で手仕舞い。",
        "プレミアム・スキュー・コストはBS近似+感度分析 (前提はスクリプト冒頭)。",
        "",
    ]
    base5 = []
    for _, d0 in ev[ev["spread"] > TAU].iterrows():
        i_daily = df.index[df["date"] == d0["date"]][0]
        s0 = float(df["close"].iloc[i_daily])
        vol0 = float(df["vi"].iloc[i_daily]) / 100
        prem0 = bs_straddle(s0, s0, 14 / 365, vol0)
        lines += [f"## シグナル週 {d0['date'].date()} (VI {vol0 * 100:.1f}%)",
                  "",
                  f"- 先物 {s0:,.0f} ／ ストラドル取得プレミアム ≈ "
                  f"**{prem0 / s0:.1%} of 元本** (IV {vol0 * 100:.1f}%・14日満期)",
                  "",
                  "| 手仕舞い | 価格の動き | スキュー+0%/コスト1% | "
                  "スキュー+10%/コスト2% (基準) | スキュー+20%/コスト3% (最悪想定) |",
                  "|---|---|---|---|---|"]
        for hold_bd, t_left in [(5, 7 / 365), (10, 1 / 365)]:
            row = df.iloc[i_daily + hold_bd]
            s1 = float(row["close"])
            v1 = float(row["vi"])
            vol1 = v1 / 100 if not np.isnan(v1) else vol0
            gross = bs_straddle(s1, s0, t_left, vol1) - prem0
            cells = []
            for skew, cost in [(0.00, 0.01), (0.10, 0.02), (0.20, 0.03)]:
                entry_iv = vol0 * (1 + skew)
                net = gross - (entry_iv - vol0) / vol0 * prem0 - cost * prem0
                cells.append(f"{net / s0:+.1%}")
                if hold_bd == 5:
                    base5.append({"date": str(d0["date"].date()), "skew": skew,
                                  "cost": cost, "net": net / s0})
            lines.append(f"| {hold_bd}営業日後 ({row['date'].date()}) | "
                         f"{s1 / s0 - 1:+.1%} ({s1:,.0f}) | "
                         f"{cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")

    base = [x for x in base5 if x["skew"] == 0.10 and x["cost"] == 0.02]
    best = max(x["net"] for x in base5)
    worst = min(x["net"] for x in base5)
    if base:
        verdict = ("**利益が残る** — コスト・プレミアム込みでも買いシグナルの優位性は生存"
                   if all(x["net"] > 0 for x in base) else
                   "**利益が消える** — コスト込みでは買い判断の根拠にならない")
        lines += ["## 判定", "",
                  f"- 5営業日保有・基準ケース (スキュー+10%・コスト2%): {verdict}",
                  f"- 感度の範囲 (全9通りの5営業日保有): 最良 {best:+.1%} ～ 最悪 {worst:+.1%} of 元本",
                  "- 注: ラベル補正 (実日付=ラベル+1日) の再ベースライン後の結果。"
                  "旧ラベルでは2024-08/2025-04が「大当たり」に見えたが、ラベルずれにより"
                  "暴落後の安値で買うシミュレーションになっていたアーティファクト。",
                  "- n=2 (歴史上の点灯2回) のため本結果も確定打ではない。"
                  "最終確認は今後の点灯 (正真正銘の未来データ) で行う。", ""]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved: {REPORT}")


if __name__ == "__main__":
    main()
