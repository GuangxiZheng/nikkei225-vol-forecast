#!/usr/bin/env python
"""「ボラを買う側 (long)」に特化した分析 — ゴール「いつ買うか」への第一段。

既存検証 (vol_backtest.py) のタイミング戦略は short/long 混在で判定したため、
買い側だけが儲かる条件なのかは分かれていなかった。本分析では:

  - 予測は本番モデル (v2.1・walkforward・未来参照ゼロ) の金曜エントリ
  - τ=200 は既存検証で前半データから凍結済みの値をそのまま使う (再選択しない)
  - long 条件: predRV² > VI² + τ (オプションが理論的に大幅割安な週)
  - P&L: バリアンススワップ近似 (RV_fwd² − VI²)×5/252 / 週 (コスト・スキュー未反映)

答える問い:
  1. 「買い」条件は歴史的に何週 firing したか (いつ買えたか)
  2. その週の損益は正か (買えば儲かったか) ・最悪週はどれだけ痛いか
  3. 前半/後半で一貫するか (時分割検証の型)

使い方: .venv/bin/python scripts/long_vol_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vol_backtest import ANNUAL, EVAL_FROM, HORIZON, build_features, load_data  # noqa: E402
from volmodel import target_h, walkforward_direct  # noqa: E402

REPORT = ROOT / "results" / "long_vol_report.md"
TAU = 200.0  # vol_backtest で前半から凍結済みの値 (再選択しない)


def main():
    df = build_features(load_data())
    df["y5"] = target_h(df, 5)
    df["pred"] = walkforward_direct(df, 5)

    ev = df[(df["date"] >= EVAL_FROM) & (df["date"].dt.dayofweek == 4)].copy()
    ev = ev.dropna(subset=["pred", "vi", "rv_fwd"]).reset_index(drop=True)
    ev["spread"] = ev["pred"] ** 2 - ev["vi"] ** 2
    ev["long_pnl"] = np.where(ev["spread"] > TAU,
                              (ev["rv_fwd"] ** 2 - ev["vi"] ** 2) * (HORIZON / ANNUAL),
                              0.0)
    ev["is_long"] = ev["spread"] > TAU
    ev["half"] = np.where(ev["date"] < "2023-01-01", "first", "second")

    def block(d, label):
        x = d[d["is_long"]]
        if len(x) == 0:
            return f"| {label} | 0 | — | — | — | — | — |"
        ann = x["long_pnl"].mean() * (ANNUAL / HORIZON)
        rng = np.random.default_rng(7)
        pv = x["long_pnl"].values * (ANNUAL / HORIZON)
        boots = np.array([rng.choice(pv, len(pv)).mean() for _ in range(5000)])
        worst = x.loc[x["long_pnl"].idxmin()]
        return (f"| {label} | {len(x)} | {ann:+.0f} | "
                f"[{np.percentile(boots, 2.5):+.0f}, {np.percentile(boots, 97.5):+.0f}] | "
                f"{worst['long_pnl'] * (ANNUAL / HORIZON):+.0f} | "
                f"{(x['long_pnl'] > 0).mean():.0%} |")

    ann_always = float(((ev["rv_fwd"] ** 2 - ev["vi"] ** 2) * (HORIZON / ANNUAL)).mean()
                       * (ANNUAL / HORIZON))

    lines = [
        "# 「いつ買うか」分析 — long-vol週の検証 (予測モデルv2.1)",
        "",
        f"- 評価: {EVAL_FROM}〜 金曜エントリ {len(ev)}週 ／ long条件: 予測² > VI²+{TAU:.0f}"
        f" (τは既存検証で凍結済みの値を流用・再選択なし)",
        f"- 比較基準: 常にlong無条件 = 年率 {ann_always:+.0f} (バリアンススワップ近似・コスト未反映)",
        "",
        "## 買いシグナルの履歴実績",
        "",
        "| 区間 | 買い週数 | 年率キャプチャ | 95%CI | 最悪週(年率換算) | 勝率 |",
        "|---|---|---|---|---|---|",
        block(ev, "全期間"),
        block(ev[ev["half"] == "first"], "前半(〜2022)"),
        block(ev[ev["half"] == "second"], "後半(2023〜)"),
        "",
    ]

    longs = ev[ev["is_long"]]
    if len(longs):
        lines.append("## 買いシグナルが出た週 (直近10件)")
        lines.append("")
        lines.append("| 金曜日付 | 予測 | VI | 実績(5日) | long損益(週) |")
        lines.append("|---|---|---|---|---|")
        for _, x in longs.tail(10).iterrows():
            lines.append(f"| {x['date'].date()} | {x['pred']:.1f}% | {x['vi']:.1f}% | "
                         f"{x['rv_fwd']:.1f}% | {x['long_pnl']:+.0f} |")
        lines += ["",
                  f"- 買いシグナルの発生頻度: {len(longs)}/{len(ev)}週 "
                  f"({len(longs)/len(ev):.0%}) — 通常は flat が正解の市場",
                  f"- VI水準別: シグナル週のVI中央値 {longs['vi'].median():.1f}% "
                  f"(全週の中央値 {ev['vi'].median():.1f}%)"]

    lines += [
        "",
        "## 判定と限界",
        "",
        "- 採用基準は事前宣言なしの探索的分析である点に注意 (本分析は仮説生成)。",
        "- バリアンススワップ近似。実ストラドルではスキュー・ theta・売買コストで劣化する。",
        "- 結論が正でも負でも、次の一手 (事前登録バックテスト or 中止) はこの結果で決める。",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved: {REPORT}")


if __name__ == "__main__":
    main()
