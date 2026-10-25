#!/usr/bin/env python
"""日経225ボラティリティ予測の時分割検証。

問い: 「日経VI(市場の含み波動度)より、統計モデルは将来の実現ボラを
      上手く予測できるか? それで変動プレミアム(VRP)戦略は億かるか?」

- 実現ボラ(RV): 日経225先物の日次対数リターンから計算(年率%)
- モデル: HAR-RV (日/週/月のRV成分 + 日経VI) をOLSで推定
- 検証: 拡張ウィンドウの月次再推定ウォークフォワード(未来参照ゼロ)
- 戦略: 週次ノンオーバーラップのバリアンス・スワップ近似
  short: predRV² < VI²−τ / long: predRV² > VI²+τ / flat: それ以外
  P&L = (VI² − RV_fwd²)×(5/252) (short時)。τは前半で選び後半で検証する。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORT = ROOT / "results" / "vol_report.md"

HORIZON = 5          # 予測ホライズン(営業日)
EVAL_FROM = "2018-01-01"   # 評価開始(それ以前は学習ウォームアップ)
ANNUAL = 252


def load_data():
    fut = json.load(open(DATA / "n225_fut_daily.json"))
    vi = json.load(open(DATA / "nkvi_daily.json"))
    df = pd.DataFrame(fut, columns=["ms", "open", "high", "low", "close", "_"]).copy()
    df["date"] = pd.to_datetime(df["ms"], unit="ms").dt.normalize()
    # このデータは金曜分の足が「日曜」ラベル(+2日)で保存されるため実日付へ復元
    sun = df["date"].dt.dayofweek == 6
    df.loc[sun, "date"] = df.loc[sun, "date"] - pd.Timedelta(days=2)
    assert not df["date"].duplicated().any(), "日付の重複(曜日補正に失敗)"
    df = df[["date", "high", "low", "close"]].drop_duplicates("date").sort_values("date")
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    v = pd.DataFrame(vi, columns=["ms", "vi"]).copy()
    v["date"] = pd.to_datetime(v["ms"], unit="ms").dt.normalize()
    df = df.merge(v, on="date", how="left")
    df["vi"] = df["vi"].ffill(limit=5)
    return df.reset_index(drop=True)


def build_features(df):
    r = df["ret"]
    df["rv_d"] = r.abs() * np.sqrt(ANNUAL) * 100   # 日次RV = |リターン|(年率%)
    df["rv_w"] = r.rolling(5).std() * np.sqrt(ANNUAL) * 100
    df["rv_m"] = r.rolling(22).std() * np.sqrt(ANNUAL) * 100
    # Parkinson (レンジ): 高値-安値の情報 (v2.1で追加・feature_ext_report.md)
    df["pk5"] = (np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2))
                 ).rolling(5).mean().pow(0.5) * np.sqrt(ANNUAL) * 100
    # 5営業日後までの実現ボラ(年率%) — 予測ターゲット
    fwd = pd.concat([r.shift(-i) for i in range(1, HORIZON + 1)], axis=1)
    df["rv_fwd"] = fwd.std(axis=1, ddof=1) * np.sqrt(ANNUAL) * 100
    df["vi2"] = df["vi"] ** 2
    return df


def fit_har(df, cols, use_vi=True):
    """OLSでHAR系モデルを推定し、全期間の予測値を返す(拡張ウィンドウ・月次再推定)。"""
    out = pd.Series(np.nan, index=df.index)
    X_all = df[cols]
    months = df["date"].dt.to_period("M").unique()
    for m in months:
        cut = m.to_timestamp()
        tr_end = df["date"] < cut
        te = (df["date"] >= cut) & (df["date"] < cut.to_period("M").end_time + pd.Timedelta(days=1))
        tr = df[tr_end & df["rv_fwd"].notna() & df[cols].notna().all(axis=1)]
        if len(tr) < 120:
            continue
        Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols])
        ytr = tr["rv_fwd"].values
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        Xte = df[te]
        ok = Xte[cols].notna().all(axis=1)
        Xv = np.column_stack([np.ones(ok.sum())] + [Xte.loc[ok, c].values for c in cols])
        out.loc[Xte.loc[ok].index] = Xv @ beta
    return out


def oos_r2(pred, actual, naive):
    ok = pred.notna() & actual.notna() & naive.notna()
    mse_p = ((pred[ok] - actual[ok]) ** 2).mean()
    mse_n = ((naive[ok] - actual[ok]) ** 2).mean()
    return 1 - mse_p / mse_n, ok.sum()


def main():
    df = load_data()
    df = build_features(df)
    df["naive"] = df["rv_w"]
    df["m_har"] = fit_har(df, ["rv_d", "rv_w", "rv_m"])
    df["m_vi"] = fit_har(df, ["vi"])
    df["m_har_vi"] = fit_har(df, ["rv_d", "rv_w", "rv_m", "vi"])

    ev = df[df["date"] >= EVAL_FROM].copy()
    ev = ev[ev["date"].dt.dayofweek == 4]  # 金曜エントリ(週末時点で判断)
    ok_all = ev["rv_fwd"].notna() & ev["vi"].notna()
    ev = ev[ok_all].reset_index(drop=True)

    lines = [
        "# 日経225 ボラティリティ予測 — 時分割検証報告",
        "",
        f"- データ: 日経225先物 日足 {df['date'].min().date()}〜{df['date'].max().date()}"
        f" ({len(df)}日) / 日経VI {df['vi'].first_valid_index() and df.loc[df['vi'].first_valid_index(), 'date'].date()}〜",
        f"- 評価: {EVAL_FROM}〜 金曜エントリの週次ノンオーバーラップ {len(ev)}週",
        f"- ターゲット: 5営業日先の実現ボラ(年率%) — 競艇の「抽選結果」に相当",
        "",
        "## 予測精度 (out-of-time, OOS R² vs ナイーブ=persistence)",
    ]
    actual, naive = ev["rv_fwd"], ev["naive"]
    for name, col in [("日経VI(市場)", "vi"), ("HAR-RV", "m_har"), ("VIのみOLS", "m_vi"),
                      ("HAR+VI", "m_har_vi")]:
        r2, n = oos_r2(ev[col], actual, naive)
        lines.append(f"- {name}: OOS R² = {r2:+.3f} (n={n})")

    okm = ev[["vi", "m_har", "m_har_vi", "rv_fwd"]].dropna()
    # モデル vs 市場 の二乗誤差差のブートストラップ
    e_vi = ((okm["vi"] - okm["rv_fwd"]) ** 2).values
    e_md = ((okm["m_har_vi"] - okm["rv_fwd"]) ** 2).values
    rng = np.random.default_rng(7)
    diffs = np.array([rng.choice(e_md - e_vi, len(e_md)).mean() for _ in range(5000)])
    lines += [
        f"- 二乗誤差: VI {e_vi.mean():.2f} / HAR+VI {e_md.mean():.2f} "
        f"(差の95%CI [{np.percentile(diffs, 2.5):+.2f}, {np.percentile(diffs, 97.5):+.2f}] "
        f"— 負ならモデル優位)",
        f"- 実測変動プレミアム: 平均(VI²−RV_fwd²) = "
        f"{(ev['vi']**2 - ev['rv_fwd']**2).mean():.0f} (年率ボラ^2ポイント)",
        "",
        "## VRP戦略シミュレーション (週次・バリアンススワップ近似・年率換算)",
        "",
        "τは前半(〜2022年)で選択し後半(2023年〜)を未知データとして検証:",
        "",
        "| 戦略 | 期間 | 保有週 | 年率キャプチャ | 95%CI | 最悪週 |",
        "|---|---|---|---|---|---|",
    ]

    ev["half"] = np.where(ev["date"] < "2023-01-01", "first", "second")

    def pnl_weeks(d, tau, use_model=True):
        if use_model:
            spread = d["m_har_vi"] ** 2 - d["vi"] ** 2
            pos = np.where(spread < -tau, "short",
                           np.where(spread > tau, "long", "flat"))
        else:
            pos = np.full(len(d), "short")   # 無条件VRP(基準戦略)
        raw = d["vi"] ** 2 - d["rv_fwd"] ** 2   # short で受け取る
        pnl = np.where(pos == "short", raw * (HORIZON / ANNUAL),
                       np.where(pos == "long", -raw * (HORIZON / ANNUAL), 0.0))
        return pd.Series(pnl, index=d.index), pd.Series(pos, index=d.index)

    def stats(pnl, pos):
        held = pos != "flat"
        if held.sum() == 0:
            return None
        ann = pnl[held].mean() * (ANNUAL / HORIZON)
        rng2 = np.random.default_rng(7)
        pv = pnl[held].values * (ANNUAL / HORIZON)
        boots = np.array([rng2.choice(pv, len(pv)).mean() for _ in range(5000)])
        worst = pnl[held].min() * (ANNUAL / HORIZON)
        return dict(n=int(held.sum()), ann=ann,
                    lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                    worst=float(worst))

    # τ選択(前半): 保有週が十分ある条件で年率キャプチャ最大のτを選ぶ
    half1 = ev[ev["half"] == "first"]
    best_tau, best_cap = None, -np.inf
    for tau in (0, 25, 50, 75, 100, 150, 200):
        p, pos = pnl_weeks(half1, tau)
        s = stats(p, pos)
        if s and s["n"] >= 30 and s["ann"] > best_cap:
            best_tau, best_cap = tau, s["ann"]
    lines.append(f"- (前半で選択されたτ: {best_tau}, 前半キャプチャ {best_cap:.0f})")

    for label, tau, use_model in [
        ("常にshort(無条件VRP)", None, False),
        (f"モデル timing τ={best_tau}", best_tau, True),
    ]:
        for half in ("first", "second"):
            d = ev[ev["half"] == half]
            p, pos = pnl_weeks(d, tau if tau is not None else 0, use_model)
            s = stats(p, pos)
            if s:
                lines.append(f"| {label} | {half} | {s['n']} | {s['ann']:.0f} | "
                             f"[{s['lo']:.0f}, {s['hi']:.0f}] | {s['worst']:.0f} |")

    # 最悪週の特定
    p, pos = pnl_weeks(ev, best_tau)
    wk = p[pos == "short"].idxmin()
    lines += [
        "",
        f"- 最悪週(short時): {ev.loc[wk, 'date'].date()} (VI {ev.loc[wk, 'vi']:.1f} → "
        f"実現 {ev.loc[wk, 'rv_fwd']:.1f})",
        "",
        "## 制約",
        "- 戦略P&Lはバリアンス・スワップ近似。実オプションではスキュー・金利・",
        "  売買コスト・証拠金が追加され、実績はこれより劣る",
        "- short-variance はテールリスクを持つ(最悪週の数字がそれ)",
        "- VIは日経225オプションから計算された指数。直接取引対象は先物/オプション",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
