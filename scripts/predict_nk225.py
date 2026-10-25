#!/usr/bin/env python
"""日経225ボラティリティ予測 v2.1 — 経路予測・予測区間・2ロジック合議。

- 経路(線)予測: 次の1/2/3/5営業日の実現ボラ(年率%)を直接多ホライズン
  HAR+VI+レンジ(Parkinson)で (v2.1: pk5追加・feature_ext_report.md)
- 80%予測区間: ウォークフォワード残差のコンフォーマル分位点
  (実カバレッジは検証で 65〜71% と名目を下回るため「目安」として扱う)
- 2ロジック: HAR+VI+レンジ と GARCH(1,1)(arch) を独立に予測し乖離を報告。
  乖離>30%の週は検証で精度が落ちる(6.77→7.73pt)ことが確認済み
- アンサンブル: 2ロジック平均(h=5でOOS R² +0.376と全手法最高・検証済み)

使い方:
  .venv/bin/python scripts/predict_nk225.py --refresh
出力: 標準出力 + results/model_latest.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vol_backtest import build_features, load_data  # noqa: E402
from volmodel import (COLS, HORIZONS, conformal_bounds, garch_path_full,  # noqa: E402
                      target_h, walkforward_direct)

TAU = 200
OUT = ROOT / "results" / "model_latest.json"


def ols_beta(tr, target):
    X = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in COLS])
    beta, *_ = np.linalg.lstsq(X, tr[target].values, rcond=None)
    return beta


def main():
    if "--refresh" in sys.argv:
        import fetch_nk225
        fetch_nk225.main()

    df = build_features(load_data())
    for h in HORIZONS:
        df[f"y{h}"] = target_h(df, h)
    last = df.iloc[-1]
    x = np.array([1.0] + [last[c] for c in COLS])

    # 経路: HAR+VI (全データ推定) + GARCH (全データ推定)
    har_path, garch_path, resid = {}, {}, {}
    for h in HORIZONS:
        beta = ols_beta(df[df[f"y{h}"].notna() & df[COLS].notna().all(axis=1)], f"y{h}")
        har_path[h] = float(x @ beta)
        # コンフォーマル区間用の OOS 残差 (ウォークフォワード)
        wf = walkforward_direct(df, h)
        resid[h] = (wf - df[f"y{h}"]).dropna()
    garch_path = dict(zip(HORIZONS, garch_path_full((df["ret"] * 100).dropna().values)))

    ens_path = {h: (har_path[h] + garch_path[h]) / 2 for h in HORIZONS}
    rel_gap = abs(har_path[5] - garch_path[5]) / ((har_path[5] + garch_path[5]) / 2)
    consensus = "合意" if rel_gap <= 0.30 else "乖離(注意: 乖離週は検証で精度低下)"
    vi = float(last["vi"])
    signal = vi ** 2 - ens_path[5] ** 2
    stance = ("short(VRP売り)" if signal > TAU else
              "long(ボラ買い)" if signal < -TAU else "flat(中立)")

    rows = []
    for h in HORIZONS:
        lo, hi = conformal_bounds(resid[h])
        rows.append({
            "h": h, "ensemble": round(ens_path[h], 2), "har_vi": round(har_path[h], 2),
            "garch": round(garch_path[h], 2),
            "pi80_lo": round(ens_path[h] + lo, 2), "pi80_hi": round(ens_path[h] + hi, 2),
        })

    out = {
        "model": "v2.1: path(HAR+VI+レンジ + GARCH ensemble) + conformal 80% PI + consensus",
        "data_asof": str(last["date"].date()),
        "features": {c: round(float(last[c]), 2) for c in COLS},
        "path_forecast": rows,
        "consensus": {"rel_gap": round(float(rel_gap), 3),
                      "verdict": consensus,
                      "verified": "乖離週は誤差 7.95pt vs 合意週 6.76pt (forecast_v2_report.md)"},
        "vrp": {"signal": round(signal, 0), "stance_tau200": stance,
                "caveat": "CIがゼロを含む未証明の戦略部分"},
        "verification": "results/forecast_v2_report.md / vol_report.md / validation_report.md",
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("╔═════════════════════════════════════════════════════════╗")
    print("║ 日経225 ボラティリティ経路予測 v2 (2ロジック+区間+合議)   ║")
    print("╠═════════════════════════════════════════════════════════╣")
    print(f"║ データ時点: {last['date'].date()}  VI: {vi:.1f}%")
    print("║")
    print("║   先読み  予測(アンサンブル)   80%区間        HAR   GARCH")
    for r in rows:
        print(f"║   {r['h']:>2}日先   {r['ensemble']:6.1f}%        "
              f"[{r['pi80_lo']:5.1f}, {r['pi80_hi']:5.1f}]   "
              f"{r['har_vi']:6.1f}  {r['garch']:6.1f}")
    print("║")
    print(f"║ 合議: HAR {har_path[5]:.1f}% vs GARCH {garch_path[5]:.1f}% "
          f"→ 乖離 {rel_gap:.1%} 【{consensus}】")
    print(f"║ VRP(h=5基準): シグナル {signal:+.0f} → スタンス {stance}")
    print("╠═════════════════════════════════════════════════════════╣")
    print("║ 検証: h=5 OOS R² HAR +0.370/GARCH +0.249/アンサンブル +0.376")
    print("║ 区間: 実カバレッジ 65〜71% (名目80%より狭い→目安として使用)")
    print("╚═════════════════════════════════════════════════════════╝")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
