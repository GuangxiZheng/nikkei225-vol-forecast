#!/usr/bin/env python
"""優先順位1の特徴量拡張検証: 既存データ (OHLC) から未使用情報を足して
v2.0基準 (HAR+VI の4特徴量) が改善するかを、本家と同じウォークフォワードで検証する。

事前宣言した候補 (results/feature_candidates.md セクション0):
  - pk5  : Parkinsonレンジ推定 (週次) — 高値-安値の情報 (build_features が提供)
  - dsr22: 下落半変動率の比率 (22日) — 非対称性 (レバレッジ効果)
  - maxz5: 直近5日の最大標準化日次変動 — ジャンプ成分
出来高はソースの欠落 (2007年まで) につき対象外。

判定基準 (すべて満たす場合のみ採用候補):
  1. 後半サンプルで基準より OOS R² が改善
  2. 二乗誤差差の95%CIがゼロを含まない (基準より有意に小さい)
  3. 前半サンプルでも同方向 (前半だけで効く過学習でない)

結果: results/feature_ext_report.md
使い方: .venv/bin/python scripts/feature_ext_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vol_backtest import build_features, load_data  # noqa: E402
from volmodel import target_h, walkforward_direct  # noqa: E402

REPORT = ROOT / "results" / "feature_ext_report.md"
EVAL_FROM = "2018-01-01"
BASE4 = ["rv_d", "rv_w", "rv_m", "vi"]  # v2.0の基準特徴量 (v2.1=COLSはpk5を含む)


def add_ext_features(df):
    """事前宣言した拡張特徴量を追加する (pk5はbuild_featuresが提供)。未来参照ゼロ。"""
    r = df["ret"]
    # 下落半変動率の比率: 22日の下落リターンstd ÷ 月次RV
    neg = r.where(r < 0, 0.0)
    dn22 = neg.pow(2).rolling(22).mean().pow(0.5) * np.sqrt(252) * 100
    df["dsr22"] = dn22 / df["rv_m"]
    # ジャンプ代理: 直近5日の最大標準化日次変動
    z = r.abs() / r.rolling(22).std()
    df["maxz5"] = z.rolling(5).max()
    return df


def friday_errors(df, cols, h=5):
    """金曜 (2018以降) 非オーバーラップの予測誤差列を返す。"""
    wf = walkforward_direct(df, h, cols=cols)
    e = pd.DataFrame({"date": df["date"], "p": wf, "a": df[f"y{h}"], "n": df["rv_w"]})
    e = e[(e["date"] >= EVAL_FROM) & (e["date"].dt.dayofweek == 4)].dropna()
    return e.reset_index(drop=True)


def r2_block(e):
    """OOS R² (ベンチマーク: 週次RV) とサンプル数。"""
    mse = ((e["p"] - e["a"]) ** 2).mean()
    return 1 - mse / ((e["n"] - e["a"]) ** 2).mean(), len(e)


def evaluate(df, variants, base_tag="V1 基準 (HAR+VI)"):
    """全バリアントを評価し、結果の (表, 詳細dict) を返す。"""
    errors = {tag: friday_errors(df, cols) for tag, cols in variants.items()}
    base = errors[base_tag]
    rows, detail = [], {}
    for tag, cols in variants.items():
        e = errors[tag]
        r2, n = r2_block(e)
        mid = e["date"].iloc[len(e) // 2]
        r2_1, _ = r2_block(e[e["date"] < mid])
        r2_2, _ = r2_block(e[e["date"] >= mid])
        # 基準と共通の日付でペア比較 (二乗誤差差の95%CI)
        m = base[["date"]].merge(e, on="date")
        mb = base[base["date"].isin(m["date"])]
        d = (mb["p"] - mb["a"]) ** 2 - (m["p"] - m["a"]) ** 2
        dmse, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        rows.append({"tag": tag, "k": len(cols), "n": n, "r2": r2,
                     "r2_1": r2_1, "r2_2": r2_2, "dmse": dmse, "ci": 1.96 * se})
        detail[tag] = e
    return pd.DataFrame(rows), detail


def main():
    df = build_features(load_data())
    df["y5"] = target_h(df, 5)
    df = add_ext_features(df)

    ext = ["pk5", "dsr22", "maxz5"]
    variants = {
        "V1 基準 (HAR+VI)": BASE4,
        "V2 +pk5": BASE4 + ["pk5"],
        "V3 +dsr22": BASE4 + ["dsr22"],
        "V4 +maxz5": BASE4 + ["maxz5"],
        "V5 全部乗 (+3)": BASE4 + ext,
    }
    res, detail = evaluate(df, variants)

    base_r2 = res.loc[0, "r2"]
    corr = df[BASE4 + ext].dropna().corr()

    lines = [
        "# 特徴量拡張検証レポート (優先順位1: 既存データ強化)",
        "",
        f"実行: {pd.Timestamp.now():%Y-%m-%d %H:%M} ／ 評価: {EVAL_FROM} 以降の金曜"
        "・非オーバーラップ・ベンチマーク=週次RV (model_health と同じ型)",
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
        f"基準 (V1) の再現値: OOS R² {base_r2:+.3f} — README の検証3 (+0.342) と一致すれば",
        "パイプラインの整合OK。",
        "",
        "## 拡張特徴量と既存特徴量の相関 (多重共線性の確認)",
        "",
        corr.round(2).to_string(),
        "",
        "## 採用判定 (事前宣言した基準)",
        "",
        "採用には (1) 後半でのR²改善、(2) ΔMSEの95%CIがゼロを含まない、(3) 前半でも同方向、",
        "の3点すべてが必要。1つでも欠ければ不採用とし、この結論は検証済み知見として残す。",
        "",
    ]
    adopted = []
    for _, x in res.iterrows():
        if x["tag"].startswith("V1"):
            continue
        ok1 = x["r2_2"] > res.loc[0, "r2_2"]
        ok2 = x["dmse"] - x["ci"] > 0
        ok3 = x["r2_1"] > res.loc[0, "r2_1"] * 0.98  # 前半は同方向(±2%許容)
        verdict = ("採用候補" if (ok1 and ok2 and ok3) else "不採用")
        if ok1 and ok2 and ok3:
            adopted.append(x["tag"])
        lines.append(f"- **{x['tag']}**: 後半改善 {ok1} / CI除外ゼロ {ok2} / "
                     f"前半同方向 {ok3} → **{verdict}**")
    lines += ["", "## 制限", "",
              "- 5バリアントの多重検定を厳密には補正していない (採用候補が出た場合は",
              "  次の月次データでアウトオブサンプル確認を義務付ける)。",
              "- 出来高はソース欠落 (2007年まで) のため対象外。"]
    REPORT.write_text("\n".join(lines) + "\n")

    print(res.to_string(index=False,
                        formatters={"r2": "{:+.3f}".format, "r2_1": "{:+.3f}".format,
                                    "r2_2": "{:+.3f}".format,
                                    "dmse": "{:+.1f}".format, "ci": "±{:.1f}".format}))
    print(f"\n基準再現: R²={base_r2:+.3f} (README検証3は+0.342)")
    print(f"採用候補: {adopted if adopted else 'なし'}")
    print(f"saved: {REPORT}")


if __name__ == "__main__":
    main()
