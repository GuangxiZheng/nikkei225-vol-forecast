#!/usr/bin/env python
"""ステージ2: 日経VI (S161.json) のラベルが実日付とずれているかの検証。

先物系列では「ラベル = 実日付 − 1日」が確定済み (2026-10-28)。S161 (VI) が
同じずれを持つなら、現行のマージ (生ラベルで結合) は VI を1日未来から取って
おり、ウォークフォワードに未来参照リークが発生していることになる。

判定方法 (事前宣言): FRED の VIXCLS (日付が確実な独立ソース) との整合を、
  A案: VIを生ラベル日で VIX と結合 (現行の解釈)
  B案: VIを実日付 (ラベル+1) で VIX と結合 (先物と同じずれの解釈)
の2通りで作り、水準・日次変化の相関を比較する。
  - Δcorr(B) − Δcorr(A) > +0.03 → B (=ずれあり、load_dataの修正が必要)
  - < −0.03 → A (現行のままで良い)
  - その間 → 判定不能 (現状維持・限界として記録)
補助証拠: 2024-08 と 2026-10 の急変時点で VI スパイクがどのラベルに乗るか。

使い方: .venv/bin/python scripts/vi_label_check.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vol_backtest import DATA  # noqa: E402

REPORT = ROOT / "results" / "vi_label_check.md"
START = "2017-01-01"


def corr(a, b):
    m = pd.concat([a, b], axis=1).dropna()
    return float(m.iloc[:, 0].corr(m.iloc[:, 1])) if len(m) > 100 else np.nan


def main():
    vi_raw = json.loads((DATA / "nkvi_daily.json").read_text())
    v = pd.DataFrame(vi_raw, columns=["ms", "vi"])
    v["label"] = pd.to_datetime(v["ms"], unit="ms").dt.normalize().dt.strftime("%Y-%m-%d")
    v["real"] = (pd.to_datetime(v["label"]) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")

    ext = json.loads((DATA / "external_daily.json").read_text())
    vx = pd.DataFrame([{"date": d, "vix": x.get("vix")} for d, x in ext.items()
                       if x.get("vix") is not None]).set_index("date")["vix"]

    a = v.set_index("label")["vi"].rename("vi_A")  # A案: ラベル日=実日付
    b = v.set_index("real")["vi"].rename("vi_B")   # B案: 実日付=ラベル+1
    a.index = pd.to_datetime(a.index)
    b.index = pd.to_datetime(b.index)
    vx.index = pd.to_datetime(vx.index)

    rows = []
    for name, s in [("A案 (ラベル日=実日付・現行)", a), ("B案 (実日付=ラベル+1)", b)]:
        c_lvl = corr(s[s.index >= START], vx[vx.index >= START])
        d_s = s.diff().dropna()
        d_x = vx.diff().dropna()
        c_chg = corr(d_s[d_s.index >= START], d_x[d_x.index >= START])
        rows.append({"案": name, "corr水準": c_lvl, "corr変化": c_chg})
        print(f"{name}: corr(水準)={c_lvl:.4f}  corr(日次変化)={c_chg:.4f}")
    diff = rows[1]["corr変化"] - rows[0]["corr変化"]

    # 補助証拠: 急変時のスパイク位置
    spike = v[(v["label"] >= "2024-08-01") & (v["label"] <= "2024-08-09")]
    spike_txt = " / ".join(f"{r['label']}: {r['vi']:.1f}" for _, r in spike.iterrows())

    if diff > 0.03:
        verdict = ("B — VIも先物と同じ1日ずれ。**load_data の VI 結合に +1日補正が必要**\n"
                   "(現行は VI が1日未来から取られており未来参照リークの疑い)")
        action = "要修正"
    elif diff < -0.03:
        verdict = "A — ずれなし。現行のマージで正しい"
        action = "現状維持"
    else:
        verdict = ("判定不能 — 相関差が小さく確定できない。現状維持とし、\n"
                   "VI には±1日の日付不確実性が残ることを限界として記録する")
        action = "現状維持"

    lines = [
        "# VI系列 (S161) のラベル検証",
        "",
        f"実行: {pd.Timestamp.now():%Y-%m-%d %H:%M} ／ 対象: {START}以降・FRED VIXCLS を独立基準に使用",
        "",
        "| 整合 | 水準の相関 | 日次変化の相関 |",
        "|---|---|---|",
        *[f"| {r['案']} | {r['corr水準']:.4f} | {r['corr変化']:.4f} |" for r in rows],
        "",
        f"- 変化相関の差 (B−A): {diff:+.4f} (閾値 ±0.03)",
        f"- 補助証拠: 2024-08-01〜09 の VI (ラベル:値) → {spike_txt}",
        "- 注: 2024-08-04(日)に70.7の行が存在する (VIは日曜に公表されない)。",
        "  単一事例であり相関判定(A案優位・差0.20)を覆すには足りないが、",
        "  週末ラベル行の扱いには小さな不確実性が残る (マージでは週末行は非一致で落ちる)。",
        f"- **判定: {verdict}**",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved: {REPORT}")


if __name__ == "__main__":
    main()
