#!/usr/bin/env python
"""特徴量拡張の月次アウトオブサンプル確認。

2026-10-16 の検証 (feature_ext_report.md) で採用候補になったバリアントが、
検証カットオフ以降の「未来データ」でも改善を保つかを毎月確認する。
仕様は results/feature_oos_spec.json に凍結済み (観測後の変更は禁止)。

使い方: .venv/bin/python scripts/feature_oos_check.py [--refresh] [--force] [--spec PATH]
  --refresh  先に最新データを取得する
  --force    月初営業日のガードを外して手動実行する
  --spec PATH 仕様ファイル (既定以外はログに書かない・テスト用)

挙動:
  - 月の最初の営業日以外は SKIP (毎日の自動実行から呼ばれることを想定)
  - カットオフ以降で y5 (5日先) が確定した金曜が2週未満でも SKIP
  - 実行時は全バリアントを同じウォークフォワード (月次再推定) で予測し、
    新規金曜のみで OOS R² と基準との二乗誤差差を評価、results/feature_oos_log.jsonl に追記
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from feature_ext_test import add_ext_features  # noqa: E402
from vol_backtest import build_features, load_data  # noqa: E402
from volmodel import target_h, walkforward_direct  # noqa: E402

SPEC = ROOT / "results" / "feature_oos_spec.json"
LOG = ROOT / "results" / "feature_oos_log.jsonl"
MIN_N = 2
CONFIRM_N = 8


def logged_months():
    """feature_oos_log.jsonl に記録済みの実行月 (YYYY-MM) の一覧。"""
    try:
        return [json.loads(l).get("run_at", "")[:7]
                for l in LOG.read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        return []


def main():
    args = sys.argv[1:]
    spec_path = Path(args[args.index("--spec") + 1]) if "--spec" in args else SPEC
    spec = json.loads(spec_path.read_text())
    cutoff = pd.Timestamp(spec["data_cutoff"])
    use_log = spec_path == SPEC

    if "--refresh" in args:
        import fetch_nk225
        fetch_nk225.main()
    # 実行ガード: 本番specなら「今月まだ実行していなければ」いつでも可
    # (ホスト停止で月初を逃しても、その月に最初に動いた日が担当する)
    if "--force" not in args:
        if spec_path != SPEC:
            print("SKIP: 月次OOS確認は月初営業日のみ (--force で手動実行可)")
            return
        cur = date.today().strftime("%Y-%m")
        if cur in logged_months():
            print("SKIP: 今月は実行済み")
            return

    df = build_features(load_data())
    df["y5"] = target_h(df, 5)
    df = add_ext_features(df)

    new = df[(df["date"] > cutoff) & (df["date"].dt.dayofweek == 4)].copy()
    wf_all = walkforward_direct(df, 5, cols=spec["base_features"])
    new["p_base"] = wf_all.reindex(new.index)
    for tag, cols in spec["variants"].items():
        new[f"p_{tag}"] = walkforward_direct(df, 5, cols=cols).reindex(new.index)
    need = ["p_base", "a", "n"] + [f"p_{t}" for t in spec["variants"]]
    new["a"], new["n"] = df["y5"].reindex(new.index), df["rv_w"].reindex(new.index)
    new = new.dropna(subset=need)
    n = len(new)
    if n < MIN_N:
        print(f"SKIP: カットオフ {spec['data_cutoff']} 以降で確定した金曜が不足 (n={n})")
        return

    mse = {v: ((new[c] - new["a"]) ** 2).mean() for v, c in
           [("base", "p_base")] + [(t, f"p_{t}") for t in spec["variants"]]}
    bench = ((new["n"] - new["a"]) ** 2).mean()
    r2 = {v: 1 - m / bench for v, m in mse.items()}

    print(f"OOS確認実行: カットオフ {spec['data_cutoff']} 以降の新規金曜 n={n} "
          f"({new['date'].iloc[0]:%Y-%m-%d} 〜 {new['date'].iloc[-1]:%Y-%m-%d})")
    for tag in spec["variants"]:
        d = mse["base"] - mse[tag]
        se = float(np.sqrt(((((new["p_base"] - new["a"]) ** 2)
                             - (new[f"p_{tag}"] - new["a"]) ** 2) ** 2).mean()) / np.sqrt(n))
        direction = ("検証と同方向 (改善)" if d > 0 else
                     "逆方向 (基準より悪化) — 要注意" if d < 0 else "変化なし")
        status = (f"確認中 (n<{CONFIRM_N})" if n < CONFIRM_N else
                  "採用確定の方向" if d > 0 else "採用見直し")
        print(f"  {tag}: OOS R² {r2[tag]:+.3f} (基準 {r2['base']:+.3f}) / "
              f"ΔMSE {d:+.2f}±{1.96 * se:.2f} → {direction} / 累積判定: {status}")

    if use_log:
        entry = {"run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                 "data_cutoff": spec["data_cutoff"], "n_oos": n,
                 "oos_first": str(new["date"].iloc[0].date()),
                 "oos_last": str(new["date"].iloc[-1].date()),
                 "r2": {k: round(v, 4) for k, v in r2.items()},
                 "mse": {k: round(v, 3) for k, v in mse.items()}}
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"logged: {LOG}")


if __name__ == "__main__":
    main()
