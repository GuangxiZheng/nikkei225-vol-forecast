#!/usr/bin/env python
"""ボラ予測 v2 共有ライブラリ: 経路予測・区間・複数ロジック合議。

- 直接多ホライズンHAR+VI: h=1..5 それぞれに独立の線形モデル(経路の「線」)
- GARCH(1,1): arch による第2ロジック(相互チェック用)
- コンフォーマル区間: ウォークフォワード残差のローリング分位点(分布フリーな
  予測区間・カバレッジを実データで検証)
- アンサンブル: 2ロジック平均 + 乖離(consensus)指標
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
COLS = ["rv_d", "rv_w", "rv_m", "vi", "pk5"]  # v2.1: Parkinsonレンジを追加 (feature_ext_report.md)
ANNUAL = 252
HORIZONS = (1, 2, 3, 5)


def target_h(df, h):
    """h営業日先までの実現ボラ(年率%)。h=1は|リターン|。"""
    r = df["ret"]
    if h == 1:
        return r.shift(-1).abs() * np.sqrt(ANNUAL) * 100
    fwd = pd.concat([r.shift(-i) for i in range(1, h + 1)], axis=1)
    return fwd.std(axis=1, ddof=1) * np.sqrt(ANNUAL) * 100


def ols_fit_predict(tr, te, target, cols):
    Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols])
    Xte = np.column_stack([np.ones(len(te))] + [te[c].values for c in cols])
    beta, *_ = np.linalg.lstsq(Xtr, tr[target].values, rcond=None)
    return Xte @ beta, beta


def walkforward_direct(df, h, cols=COLS):
    """月次再推定のウォークフォワードで h ホライズン予測列を返す。"""
    target = f"y{h}"
    out = pd.Series(np.nan, index=df.index)
    for m in df["date"].dt.to_period("M").unique():
        cut = m.start_time
        te_mask = (df["date"] >= cut) & (df["date"] <= m.end_time)
        tr = df[(df["date"] < cut) & df[target].notna() & df[cols].notna().all(axis=1)]
        te = df[te_mask]
        if len(tr) < 120:
            continue
        ok = te[cols].notna().all(axis=1) & te[target].notna()
        if ok.sum() == 0:
            continue
        pred, _ = ols_fit_predict(tr, te.loc[ok], target, cols)
        out.loc[te.loc[ok].index] = pred
    return out


def garch_path_full(rets_pct):
    """全データでGARCH(1,1)を推定し 1..5日先の経路予測(年率%)を返す。"""
    from arch import arch_model
    res = arch_model(rets_pct, mean="Constant", vol="GARCH", p=1, q=1,
                     rescale=False).fit(disp="off", show_warning=False)
    hvar = res.forecast(horizon=5, reindex=False).variance.values[-1]
    return [float(np.sqrt(hvar[:h].mean() * ANNUAL)) for h in (1, 2, 3, 5)]


def garch_walkforward(df, cache="garch_wf.json"):
    """各金曜時点でGARCH(1,1)を再推定し 5日経路を返す(キャッシュ付き)。

    キャッシュは日付キー式のため、ラベル補正などで日付がずれるとミスマップ
    する。行数の不一致を検知したら自動で無効化する (2026-10の教訓)。
    """
    cpath = DATA / cache
    rets = (df["ret"] * 100).values
    dates = df["date"].values
    idx_of = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    cache_map = {}
    if cpath.exists():
        raw = json.load(open(cpath))
        if raw.get("n_rows") == len(df):
            cache_map = raw.get("preds", {})
    out = {}
    fridays = df[(df["date"] >= "2018-01-01") & (df["date"].dt.dayofweek == 4)]
    from arch import arch_model
    for d in fridays["date"]:
        key = d.strftime("%Y%m%d")
        if key in cache_map:
            out[key] = cache_map[key]
            continue
        i = idx_of[d]
        res = arch_model(rets[1:i + 1], mean="Constant", vol="GARCH", p=1, q=1,
                         rescale=False).fit(disp="off", show_warning=False)
        hvar = res.forecast(horizon=5, reindex=False).variance.values[-1]
        out[key] = [float(np.sqrt(hvar[:h].mean() * ANNUAL)) for h in (1, 2, 3, 5)]
    cpath.write_text(json.dumps({"n_rows": len(df), "preds": cache_map | out}),
                     encoding="utf-8")
    return out


def conformal_bounds(resid, window=250, alpha=0.20):
    """直近 window 個のウォークフォワード残差の分位点で区間幅を作る。"""
    r = resid.dropna().iloc[-window:]
    lo = float(r.quantile(alpha / 2))
    hi = float(r.quantile(1 - alpha / 2))
    return lo, hi
