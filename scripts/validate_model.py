#!/usr/bin/env python
"""自作ボラ予測モデル(HAR+VI)の妥当性検証バッテリー。

「自作で問題ないこと」を実証するための4+1点検査:

  1. OLS実装クロスチェック : 手実装lstsq と statsmodels.OLS が同一係数か
  2. ウォークフォワード整合性 : 月次再推定パイプラインが「その月開始時点までの
     データのみ」で計算されていることを直接再計算で検証(未来参照の欠如)
  3. プラセボ検定 : VIをランダム並べ替え(関係を破壊)したら予測力が消えるか。
     破壊してもR²が残るなら実装が漏れている
  4. 確立OSSとの比較 : arch 8.0 の GARCH(1,1) と同条件ウォークフォワードで対戦
  5. 係数の安定性 : 月次再推定の係数が暴走していないか

出力: results/validation_report.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vol_backtest import (ANNUAL, EVAL_FROM, HORIZON, build_features,  # noqa: E402
                          fit_har, load_data)

COLS = ["rv_d", "rv_w", "rv_m", "vi"]
OUT = ROOT / "results" / "validation_report.md"
LINES = []


def head(t):
    LINES.append(t)


def run():
    df = build_features(load_data())
    df["naive"] = df["rv_w"]
    ev = df[(df["date"] >= EVAL_FROM) & (df["date"].dt.dayofweek == 4)].copy()
    ev = ev[ev["rv_fwd"].notna() & ev["vi"].notna()]
    df_idx = ev.index.copy()          # df上の行位置(後でモデル予測を整合させる)

    # ---- 1. OLSクロスチェック (手実装 vs statsmodels) ----
    tr = df[df["rv_fwd"].notna() & df[COLS].notna().all(axis=1)]
    X = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in COLS])
    y = tr["rv_fwd"].values
    beta_hand, *_ = np.linalg.lstsq(X, y, rcond=None)
    res_sm = sm.OLS(y, X).fit()
    diff = np.abs(beta_hand - res_sm.params).max()
    head("## 1. OLS実装クロスチェック")
    head(f"- 手実装lstsq vs statsmodels.OLS: 係数最大差 = {diff:.2e} → "
         f"{'PASS' if diff < 1e-8 else 'FAIL'}")

    # ---- 2. ウォークフォワード整合性 (未来参照なし) ----
    m_har_vi = fit_har(df, COLS)
    ev["m_har_vi"] = m_har_vi.loc[df_idx].values
    ev = ev.reset_index(drop=True)
    max_diff = 0.0
    for month in pd.period_range("2018-01", "2026-10", freq="M"):
        cut = month.start_time
        te = df[(df["date"] >= cut) & (df["date"] <= month.end_time)]
        tr_m = df[(df["date"] < cut) & df["rv_fwd"].notna() & df[COLS].notna().all(axis=1)]
        if len(tr_m) < 120:
            continue
        Xm = np.column_stack([np.ones(len(tr_m))] + [tr_m[c].values for c in COLS])
        b, *_ = np.linalg.lstsq(Xm, tr_m["rv_fwd"].values, rcond=None)
        ok = te[COLS].notna().all(axis=1)
        Xv = np.column_stack([np.ones(ok.sum())] + [te.loc[ok, c].values for c in COLS])
        pred = Xv @ b
        max_diff = max(max_diff, float(np.abs(pred - m_har_vi.loc[te.loc[ok].index]).max()))
    head("")
    head("## 2. ウォークフォワード整合性 (未来参照なしの直接再計算)")
    head(f"- 月次パイプライン予測と「月開始までのデータで再計算」の最大差 = {max_diff:.2e} → "
         f"{'PASS (未来参照なし)' if max_diff < 1e-9 else 'FAIL'}")

    # ---- 3. プラセボ検定 (VIの限界貢献 ΔR² を統計量にする) ----
    # VIを並べ替えると「VIの持つ情報」だけが壊れ、HAR成分は残る。
    # → 統計量は R²(HAR+VI) − R²(HARのみ) とし、実データのΔが
    #   破壊版のΔ分布より有意に大きいかを検定する。
    def oos_r2(vi_series, cols):
        d = df.copy()
        d["vi"] = vi_series
        pred = fit_har(d, cols)
        e = pd.DataFrame({"p": pred, "a": df["rv_fwd"], "n": df["naive"],
                          "fri": df["date"].dt.dayofweek, "dt": df["date"]})
        e = e[(e["dt"] >= EVAL_FROM) & (e["fri"] == 4)].dropna()
        mse_p = ((e["p"] - e["a"]) ** 2).mean()
        mse_n = ((e["n"] - e["a"]) ** 2).mean()
        return 1 - mse_p / mse_n

    r2_har_only = oos_r2(df["vi"], ["rv_d", "rv_w", "rv_m"])   # VI無し=HARのみ
    real_r2_vi = oos_r2(df["vi"], COLS)
    real_dr2 = real_r2_vi - r2_har_only
    rng = np.random.default_rng(11)
    vi_base = df["vi"].values.copy()
    placebo_d = []
    for _ in range(200):
        rng.shuffle(vi_base)
        placebo_d.append(oos_r2(pd.Series(vi_base, index=df.index), COLS) - r2_har_only)
    placebo_d = np.array(placebo_d)
    pval = float((placebo_d >= real_dr2).mean())
    head("")
    head("## 3. プラセボ検定 (VIの限界貢献 ΔR² = R²(HAR+VI) − R²(HARのみ) ×200)")
    head(f"- R²(HARのみ) = {r2_har_only:+.3f} / R²(HAR+VI実データ) = {real_r2_vi:+.3f} "
         f"→ 実ΔR² = {real_dr2:+.3f}")
    head(f"- 破壊版ΔR² 分布: mean {placebo_d.mean():+.3f}, 95% "
         f"[{np.percentile(placebo_d, 2.5):+.3f}, {np.percentile(placebo_d, 97.5):+.3f}] "
         f"/ 経験p値 = {pval:.3f} → "
         f"{'PASS (VIの情報は実在し漏れ偽陽性ではない)' if pval <= 0.05 else 'FAIL'}")

    # ---- 4. GARCH(1,1) ベンチマーク (arch 8.0) ----
    from arch import arch_model
    rets = (df["ret"] * 100).values  # %
    dates = df["date"].values
    idx_of = {d: i for i, d in enumerate(dates)}
    garch_pred = []
    for d in ev["date"]:
        i = idx_of[np.datetime64(d.tz_localize(None).to_pydatetime(), "ns")]
        am = arch_model(rets[1:i + 1], mean="Constant", vol="GARCH", p=1, q=1,
                        rescale=False)
        res = am.fit(disp="off", show_warning=False)
        f = res.forecast(horizon=HORIZON, reindex=False)
        h = f.variance.values[-1]           # h1..h5 (日次分散, %²)
        garch_pred.append(np.sqrt(h.mean() * ANNUAL))
    ev["garch"] = garch_pred

    def mse(col):
        return ((ev[col] - ev["rv_fwd"]) ** 2).mean()

    def dm_ci(a, b, n_boot=5000):
        d_ = ((ev[a] - ev["rv_fwd"]) ** 2 - (ev[b] - ev["rv_fwd"]) ** 2).values
        rng2 = np.random.default_rng(7)
        means = np.array([rng2.choice(d_, len(d_)).mean() for _ in range(n_boot)])
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    r2_g = 1 - mse("garch") / mse("naive")
    r2_m = 1 - mse("m_har_vi") / mse("naive")
    lo, hi = dm_ci("m_har_vi", "garch")
    head("")
    head("## 4. 確立OSS (arch 8.0 GARCH(1,1)) との対戦 — 同一ウォークフォワード")
    head(f"- OOS R²: HAR+VI(自作) {r2_m:+.3f} / GARCH(1,1) {r2_g:+.3f}")
    head(f"- 二乗誤差差 (GARCH−自作) の95%CI: [{lo:+.2f}, {hi:+.2f}] → "
         f"{'自作が優位' if hi < 0 else ('GARCHが優位' if lo > 0 else '優劣つかず(僅差)')}")

    # ---- 5. 係数の安定性 ----
    betas = []
    for month in pd.period_range("2018-01", "2026-10", freq="M"):
        cut = month.start_time
        tr_m = df[(df["date"] < cut) & df["rv_fwd"].notna() & df[COLS].notna().all(axis=1)]
        if len(tr_m) < 120:
            continue
        Xm = np.column_stack([np.ones(len(tr_m))] + [tr_m[c].values for c in COLS])
        b, *_ = np.linalg.lstsq(Xm, tr_m["rv_fwd"].values, rcond=None)
        betas.append(b)
    B = np.array(betas)
    head("")
    head("## 5. 係数の安定性 (月次再推定 " + f"{len(B)}回)")
    head("| 係数 | 最小 | 中央 | 最大 |")
    head("|---|---|---|---|")
    for j, name in enumerate(["切片"] + COLS):
        head(f"| {name} | {B[:, j].min():.3f} | {np.median(B[:, j]):.3f} "
             f"| {B[:, j].max():.3f} |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    head("")
    OUT.write_text("\n".join(LINES), encoding="utf-8")
    print("\n".join(LINES))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    run()
