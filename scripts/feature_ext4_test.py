#!/usr/bin/env python
"""優先順位3の残り: 薄商い期ダミーの検証 (お盆・年末・GW明け)。

事前宣言: 金曜エントリ T の次の5営業日窓が以下の暦窓に触れるかのダミー。
  - thin_summer : 8/11〜8/16 (お盆)
  - thin_yearend: 12/25〜12/31 (年末)
  - thin_gw     : 4/30〜5/6 (GW明け)
判定基準は検証1〜3と同じ事前宣言ルール。

結果: results/feature_ext4_report.md
使い方: .venv/bin/python scripts/feature_ext4_test.py
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from feature_ext_test import evaluate  # noqa: E402
from vol_backtest import build_features, load_data  # noqa: E402
from volmodel import COLS, target_h  # noqa: E402

REPORT = ROOT / "results" / "feature_ext4_report.md"
WINDOWS = [("thin_summer", (8, 11), (8, 16)),
           ("thin_yearend", (12, 25), (12, 31)),
           ("thin_gw", (4, 30), (5, 6))]


def in_window(d, start, end):
    key = (d.month, d.day)
    if start <= end:
        return start <= key <= end
    return key >= start or key <= end  # 年を跨ぐ窓用 (今回は未使用だが一般形)


def main():
    df = build_features(load_data())
    df["y5"] = target_h(df, 5)
    iso = df["date"].dt.strftime("%Y-%m-%d").to_numpy()

    for name, s, e in WINDOWS:
        flagged = np.array([in_window(date.fromisoformat(x), s, e) for x in iso])
        col = np.zeros(len(df))
        for i in range(len(df) - 5):
            if flagged[i + 1:i + 6].any():
                col[i] = 1.0
        df[f"{name}_ahead"] = col
    df["thin_all_ahead"] = ((df["thin_summer_ahead"] + df["thin_yearend_ahead"]
                             + df["thin_gw_ahead"]) > 0).astype(float)

    variants = {
        "D1 基準 (v2.1)": COLS,
        "D2 +thin_summer": COLS + ["thin_summer_ahead"],
        "D3 +thin_yearend": COLS + ["thin_yearend_ahead"],
        "D4 +thin_gw": COLS + ["thin_gw_ahead"],
        "D5 +薄商い3窓全部": COLS + ["thin_summer_ahead", "thin_yearend_ahead", "thin_gw_ahead"],
    }
    res, detail = evaluate(df, variants, base_tag="D1 基準 (v2.1)")

    lines = [
        "# 特徴量拡張検証レポート4 (優先順位3の残り: 薄商い期ダミー)",
        "",
        f"実行: {pd.Timestamp.now():%Y-%m-%d %H:%M} ／ 評価: 2018-01-01〜 金曜"
        "・ベンチマーク=週次RV。基準は v2.1 (ラベル補正後)。",
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
    lines += ["", "## 採用判定 (後半改善 / CI除外ゼロ / 前半同方向)", ""]
    adopted = []
    for _, x in res.iterrows():
        if x["tag"].startswith("D1"):
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
              "- 窓の定義は暦ベースの近似 (実際の祝日カレンダーとは数日ずれ得る)。", ""]
    REPORT.write_text("\n".join(lines) + "\n")

    print(res.to_string(index=False,
                        formatters={"r2": "{:+.3f}".format, "r2_1": "{:+.3f}".format,
                                    "r2_2": "{:+.3f}".format,
                                    "dmse": "{:+.1f}".format, "ci": "±{:.1f}".format}))
    print(f"\n採用候補: {adopted if adopted else 'なし'}")
    print(f"saved: {REPORT}")


if __name__ == "__main__":
    main()
