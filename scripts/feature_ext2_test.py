#!/usr/bin/env python
"""優先順位2の特徴量拡張検証: 海外市場変数 (VIX・S&P500・USD/JPY)。

事前宣言 (results/feature_candidates.md セクション1):
  - vix      : 米VIX水準 (年率%) — 世界の含みボラが先行することがある
  - spx_ret5 : S&P500 5日リターン (%) — 直近の米国株の動き (下落は非対称に効く)
  - jpy_chg22: USD/JPY 22日変化率 (%) — 急な円高=キャリー巻き戻し型のボラ急騰
判定基準は優先順位1と同じ事前宣言ルール (後半改善 + ΔMSEの95%CIがゼロを含まない
+ 前半でも同方向)。データは FRED (fetch_external.py)、ffill limit 5。

結果: results/feature_ext2_report.md
使い方: .venv/bin/python scripts/feature_ext2_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from feature_ext_test import evaluate  # noqa: E402
from vol_backtest import DATA, build_features, load_data  # noqa: E402
from volmodel import COLS, target_h  # noqa: E402

REPORT = ROOT / "results" / "feature_ext2_report.md"


def add_market_features(df):
    ext = json.loads((DATA / "external_daily.json").read_text())
    m = pd.DataFrame([{"date": d, **vals} for d, vals in ext.items()])
    m["date"] = pd.to_datetime(m["date"])
    df = df.merge(m, on="date", how="left")
    for c in ("spx", "vix", "usdjpy"):
        df[c] = df[c].ffill(limit=5)  # FREDの更新遅れは直近既知情報で埋める(未来参照なし)
    df["spx_ret5"] = df["spx"].pct_change(5) * 100
    df["jpy_chg22"] = df["usdjpy"].pct_change(22) * 100
    return df


def main():
    df = build_features(load_data())
    df["y5"] = target_h(df, 5)
    df = add_market_features(df)

    variants = {
        "B1 基準 (v2.1)": COLS,
        "B2 +vix": COLS + ["vix"],
        "B3 +spx_ret5": COLS + ["spx_ret5"],
        "B4 +jpy_chg22": COLS + ["jpy_chg22"],
        "B5 海外3変数全部": COLS + ["vix", "spx_ret5", "jpy_chg22"],
    }
    res, detail = evaluate(df, variants, base_tag="B1 基準 (v2.1)")
    corr = df[COLS + ["vix", "spx_ret5", "jpy_chg22"]].dropna().corr()

    lines = [
        "# 特徴量拡張検証レポート2 (優先順位2: 海外市場変数)",
        "",
        f"実行: {pd.Timestamp.now():%Y-%m-%d %H:%M} ／ 評価: 2018-01-01 以降の金曜"
        "・非オーバーラップ・ベンチマーク=週次RV。基準は v2.1 (COLS+pk5)。",
        "",
        "## 結果",
        "",
        "| バリアント | 特徴量数 | n | OOS R² (全期間) | 前半 | 後半 |"
        " ΔMSE vs 基準 (95%CI) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, x in res.iterrows():
        d = x["dmse"]
        ci_s = (f"{d:+.1f} ± {x['ci']:.1f}"
                f"{' ✅基準より有意に小さい' if d - x['ci'] > 0 else ''}")
        lines.append(f"| {x['tag']} | {x['k']} | {x['n']} | {x['r2']:+.3f} | "
                     f"{x['r2_1']:+.3f} | {x['r2_2']:+.3f} | {ci_s} |")
    lines += [
        "",
        "## 海外変数と既存特徴量の相関",
        "",
        corr.round(2).to_string(),
        "",
        "## 採用判定 (事前宣言した基準: 後半改善/CI除外ゼロ/前半同方向)",
        "",
    ]
    adopted = []
    for _, x in res.iterrows():
        if x["tag"].startswith("B1"):
            continue
        ok1 = x["r2_2"] > res.loc[0, "r2_2"]
        ok2 = x["dmse"] - x["ci"] > 0
        ok3 = x["r2_1"] > res.loc[0, "r2_1"] * 0.98
        verdict = "採用候補" if (ok1 and ok2 and ok3) else "不採用"
        if ok1 and ok2 and ok3:
            adopted.append(x["tag"])
        lines.append(f"- **{x['tag']}**: 後半改善 {ok1} / CI除外ゼロ {ok2} / "
                     f"前半同方向 {ok3} → **{verdict}**")
    lines += ["", "## 制限", "",
              "- SP500はFREDが直近10年のみのため、B系の学習開始が基準より遅い (nが小さめ)。",
              "- FREDの更新遅れ (USD/JPYは1-5営業日) はffill(5)で既知情報として扱う。",
              "- 5バリアントの多重検定は厳密には補正していない (採用時は次月のOOS確認へ)。"]
    REPORT.write_text("\n".join(lines) + "\n")

    print(res.to_string(index=False,
                        formatters={"r2": "{:+.3f}".format, "r2_1": "{:+.3f}".format,
                                    "r2_2": "{:+.3f}".format,
                                    "dmse": "{:+.1f}".format, "ci": "±{:.1f}".format}))
    print(f"\n採用候補: {adopted if adopted else 'なし'}")
    print(f"saved: {REPORT}")


if __name__ == "__main__":
    main()
