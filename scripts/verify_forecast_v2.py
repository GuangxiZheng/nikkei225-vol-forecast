#!/usr/bin/env python
"""予測v2の検証: 経路(1..5日)×複数ロジック(HAR+VI/GARCH)×区間カバレッジ。

- 各ホライズンの OOS R² と HAR vs GARCH のDM比較
- 80%予測区間の実カバレッジ(名目80%に近いか)
- 2ロジックの乖離(consensus)と予測誤差の関係
出力: results/forecast_v2_report.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vol_backtest import build_features, load_data  # noqa: E402
from volmodel import (ANNUAL, COLS, HORIZONS, conformal_bounds,  # noqa: E402
                      garch_walkforward, target_h, walkforward_direct)

OUT = ROOT / "results" / "forecast_v2_report.md"


def main():
    df = build_features(load_data())
    for h in HORIZONS:
        df[f"y{h}"] = target_h(df, h)

    # ウォークフォワード (HAR+VI: 全h / GARCH: キャッシュ付き)
    preds = {f"har_h{h}": walkforward_direct(df, h) for h in HORIZONS}
    pdf = pd.DataFrame(preds)
    pdf["date"] = df["date"].values
    gw = garch_walkforward(df)
    garch = {d: gw[d.strftime("%Y%m%d")] for d in df.loc[df.index, "date"]
             if d.strftime("%Y%m%d") in gw}
    gdf = pd.DataFrame(
        [[d] + v for d, v in garch.items()], columns=["date"] + [f"g_h{h}" for h in (1, 2, 3, 5)]
    )
    gdf["date"] = pd.to_datetime(gdf["date"])
    ev = df[(df["date"] >= "2018-01-01") & (df["date"].dt.dayofweek == 4)].copy()
    ev = ev.merge(gdf, on="date", how="inner").merge(pdf, on="date", how="left")
    ev = ev.reset_index(drop=True)

    lines = [
        "# 予測v2 — 経路予測・区間・2ロジック合議の検証",
        "",
        f"- 評価: 金曜エントリ {len(ev)}週 (2018-01〜), ホライズン h=1..5営業日",
        "",
        "## 1. 経路予測の精度 (OOS R² vs ナイーブ=週次persistence)",
        "| h | HAR+VI | GARCH | アンサンブル(平均) |",
        "|---|---|---|---|",
    ]
    dm_note = []
    for h in HORIZONS:
        a = ev[f"y{h}"]
        n = ev["rv_w"]
        r2 = {}
        for name, col in [("HAR+VI", f"har_h{h}"), ("GARCH", f"g_h{h}")]:
            ok = a.notna() & ev[col].notna() & n.notna()
            r2[name] = 1 - ((ev.loc[ok, col] - a[ok]) ** 2).mean() / \
                ((n[ok] - a[ok]) ** 2).mean()
        ev[f"ens_h{h}"] = (ev[f"har_h{h}"] + ev[f"g_h{h}"]) / 2
        ok = a.notna() & ev[f"ens_h{h}"].notna() & n.notna()
        r2["ENS"] = 1 - ((ev.loc[ok, f"ens_h{h}"] - a[ok]) ** 2).mean() / \
            ((n[ok] - a[ok]) ** 2).mean()
        lines.append(f"| {h}日 | {r2['HAR+VI']:+.3f} | {r2['GARCH']:+.3f} | {r2['ENS']:+.3f} |")
        dm_note.append((h, r2))

    # DM比較 (h=5: HAR vs GARCH, アンサンブル vs HAR)
    def dm(a_col, b_col, y_col):
        ok = ev[y_col].notna() & ev[a_col].notna() & ev[b_col].notna()
        d = ((ev.loc[ok, a_col] - ev.loc[ok, y_col]) ** 2
             - (ev.loc[ok, b_col] - ev.loc[ok, y_col]) ** 2).values
        rng = np.random.default_rng(7)
        means = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    for a_col, b_col, y_col, lab in [
        ("har_h5", "g_h5", "y5", "HAR+VI vs GARCH (h=5)"),
        ("ens_h5", "har_h5", "y5", "アンサンブル vs HAR+VI (h=5)"),
    ]:
        lo, hi = dm(a_col, b_col, y_col)
        verdict = "アンサンブル/HAR優位" if hi < 0 else ("対戦相手優位" if lo > 0 else "優劣なし")
        lines.append(f"- {lab}: 二乗誤差差の95%CI [{lo:+.1f}, {hi:+.1f}] → {verdict}")

    # 区間カバレッジ (コンフォーマル: 直近250週の残差分位点)
    lines += ["", "## 2. 80%予測区間の実カバレッジ (コンフォーマル法・名目80%)",
              "| h | 実カバレッジ | 平均区間幅(年率%pt) |", "|---|---|---|"]
    for h in (1, 5):
        e = pd.DataFrame({
            "pred": ev[f"har_h{h}"], "act": ev[f"y{h}"], "dt": ev["date"]
        }).dropna().reset_index(drop=True)
        cov = []
        widths = []
        for i in range(250, len(e)):
            lo, hi = conformal_bounds(e["pred"].iloc[:i] - e["act"].iloc[:i])
            pl, ph = e["pred"].iloc[i] + lo, e["pred"].iloc[i] + hi
            cov.append(pl <= e["act"].iloc[i] <= ph)
            widths.append(ph - pl)
        lines.append(f"| {h}日 | {np.mean(cov):.1%} | {np.mean(widths):.1f} |")

    # 合意/乖離と誤差 (h=5)
    lines += ["", "## 3. 2ロジックの合意チェック (h=5)"]
    d5 = ev[["har_h5", "g_h5", "ens_h5", "y5"]].dropna()
    rel = (d5["har_h5"] - d5["g_h5"]).abs() / ((d5["har_h5"] + d5["g_h5"]) / 2)
    agree = rel <= 0.30
    err = (d5["ens_h5"] - d5["y5"]).abs()
    lines.append(f"- 乖離率の中央値: {rel.median():.1%} / 乖離>30% の週: {(~agree).mean():.1%}")
    lines.append(f"- 合意週の平均誤差: {err[agree].mean():.2f}pt / "
                 f"乖離週: {err[~agree].mean():.2f}pt → "
                 f"乖離時ほど精度が落ちる"
                 f"{'' if err[~agree].mean() > err[agree].mean() else '(逆転: 要注意)'}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
