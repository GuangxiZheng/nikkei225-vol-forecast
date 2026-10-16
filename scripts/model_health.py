#!/usr/bin/env python
"""モデル健全性チェック (月次実行推奨)。

直近26週のウォークフォワードで「モデルが今も機能しているか」を検査する:
  - OOS R² (5日先・HAR+VI と GARCH)      → 予測力の枯渇検知
  - 80%予測区間の実カバレッジ            → 区間の信頼検証
  - 2ロジック乖離率の中央値              → ロジック間の整合

判定 (いずれか悪い側に倒れる):
  PASS / WARN / FAIL
結果は results/health_log.jsonl に追記 (日付・指標・判定)。

使い方: .venv/bin/python scripts/model_health.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vol_backtest import build_features, load_data  # noqa: E402
from volmodel import COLS, conformal_bounds, garch_walkforward, target_h, walkforward_direct

LOG = ROOT / "results" / "health_log.jsonl"
WEEKS = 26          # 検査窗口
R2_PASS, R2_WARN = 0.15, 0.05
COV_LO, COV_HI = 0.60, 0.95   # 名目80%に対する許容帯
GAP_WARN = 0.30


def main():
    df = build_features(load_data())
    df["y5"] = target_h(df, 5)
    har5 = walkforward_direct(df, 5)

    e = pd.DataFrame({
        "date": df["date"], "har": har5, "act": df["y5"], "n": df["rv_w"],
    })
    e = e[(e["date"] >= "2018-01-01") & (e["date"].dt.dayofweek == 4)].dropna()
    e = e.reset_index(drop=True).iloc[-WEEKS:]
    if len(e) < WEEKS - 4:
        print(f"[abort] 評価週が不足: {len(e)}")

    r2 = 1 - ((e["har"] - e["act"]) ** 2).mean() / ((e["n"] - e["act"]) ** 2).mean()

    # GARCH (キャッシュがあれば再推定なし)
    gw = garch_walkforward(df)
    g = {d.strftime("%Y%m%d"): v[3] for d, v in
         zip(df["date"], [gw.get(d.strftime("%Y%m%d")) for d in df["date"]])
         if gw.get(d.strftime("%Y%m%d"))}
    e["garch"] = e["date"].map(lambda d: g.get(d.strftime("%Y%m%d")))
    ok = e.dropna(subset=["garch"])
    r2_g = 1 - ((ok["garch"] - ok["act"]) ** 2).mean() / ((ok["n"] - ok["act"]) ** 2).mean() \
        if len(ok) > 10 else np.nan

    # 80%区間カバレッジ (全期間の残差ではなく、ローリングで逐次評価)
    full = pd.DataFrame({"p": har5, "a": df["y5"], "d": df["date"]}).dropna()
    full = full.reset_index(drop=True)
    cov = []
    for i in range(250, len(full)):
        lo, hi = conformal_bounds(full["p"].iloc[:i] - full["a"].iloc[:i])
        cov.append(full["p"].iloc[i] + lo <= full["a"].iloc[i] <= full["p"].iloc[i] + hi)
    coverage = float(np.mean(cov))

    # 乖離率
    if len(ok) > 10:
        gap = ((ok["har"] - ok["garch"]).abs() / ((ok["har"] + ok["garch"]) / 2)).median()
    else:
        gap = np.nan

    checks = {
        "r2_5d": (r2, "PASS" if r2 >= R2_PASS else ("WARN" if r2 >= R2_WARN else "FAIL")),
        "r2_garch": (r2_g, "PASS" if (np.isnan(r2_g) or r2_g >= R2_WARN) else "WARN"),
        "coverage80": (coverage, "PASS" if COV_LO <= coverage <= COV_HI else "FAIL"),
        "gap_median": (gap, "PASS" if (np.isnan(gap) or gap <= GAP_WARN) else "WARN"),
    }
    verdict = max((v[1] for v in checks.values()),
                  key=lambda s: ["PASS", "WARN", "FAIL"].index(s))

    entry = {
        "date": date.today().isoformat(),
        "window_weeks": int(len(e)),
        "verdict": verdict,
        "r2_5d_harvi": round(float(r2), 3),
        "r2_5d_garch": None if np.isnan(r2_g) else round(float(r2_g), 3),
        "coverage80": round(coverage, 3),
        "gap_median": None if np.isnan(gap) else round(float(gap), 3),
        "checks": {k: {"value": None if v[0] is None or (isinstance(v[0], float) and np.isnan(v[0]))
                       else round(float(v[0]), 3), "status": v[1]} for k, v in checks.items()},
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("╔════════════════════════════════════════════════╗")
    print(f"║ モデル健全性チェック (直近{WEEKS}週)  判定: {verdict:<4}      ║")
    print("╠════════════════════════════════════════════════╣")
    print(f"║ OOS R² (5日先):  HAR+VI {r2:+.3f} (閾値 {R2_PASS}/{R2_WARN}) "
          f"→ {checks['r2_5d'][1]}")
    if not np.isnan(r2_g):
        print(f"║                 GARCH  {r2_g:+.3f} → {checks['r2_garch'][1]}")
    print(f"║ 80%区間カバレッジ: {coverage:.1%} (許容 {COV_LO:.0%}〜{COV_HI:.0%}) "
          f"→ {checks['coverage80'][1]}")
    if not np.isnan(gap):
        print(f"║ 2ロジック乖離率(中央): {gap:.1%} (閾値 {GAP_WARN:.0%}) "
              f"→ {checks['gap_median'][1]}")
    print("╚════════════════════════════════════════════════╝")
    print(f"logged: {LOG}")


if __name__ == "__main__":
    main()
