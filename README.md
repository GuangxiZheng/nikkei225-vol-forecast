# nikkei225-vol-forecast — 日経225ボラティリティ予測モデル

「日経225の**ボラティリティ(変動幅)は予測できるか？市場(日経VI)より上手く予測できるか？
それで儲かるか？**」を loto-analysis(予測不可能と実証) / boatrace-analysis(予測可能と実証)
と同じ**時分割検証の型**で検証し、検証を通過した予測モデルを提供する。

## モデル (v2.1 — 2026-10-16: Parkinsonレンジを追加 / 2026-10-28: データラベル補正で再ベースライン)

- **経路予測**: 次の1/2/3/5営業日の実現ボラ(年率%)を、ホライズン別の独立モデルで予測
- **2ロジック合議**:
  - ロジック1: HAR-RV + 日経VI + レンジ (自作・6パラメータ線形モデル)。
    v2.1で Parkinsonレンジ(pk5) を追加 — 高値-安値の情報で OOS R² +0.386→+0.397
    (検証: [`results/feature_ext_report.md`](results/feature_ext_report.md)・
    ウォークフォワードで事前宣言した採用基準を通過)
  - ロジック2: GARCH(1,1) (確立OSS [`arch`](https://github.com/bashtage/arch))
  - 乖離率を合議指標とし、乖離>30%の週は精度低下フラグ(検証済み)
- **80%予測区間**: コンフォーマル法(ウォークフォワード残差の分位点・分布フリー)
- **アンサンブル**: 2ロジック平均 — 検証で全ホライズンで個別モデルに劣らず、
  HAR単体と同水準(OOS R² +0.395 vs +0.397・劣らず)
- v2.1の採用はユーザー決定による先行採用。月次のOOS確認
  ([`feature_oos_check.py`](scripts/feature_oos_check.py)・仕様凍結済み) が
  改善の持続を監視し、逆転時はロールバック判断をする

## 検証結果サマリ (n=440週・2018-01〜2026-10・未来参照ゼロ)

※ 2026-10-28: ソースの日付ラベルが実セッションの前日だと判明し補正(実日付=ラベル+1日)。以下の数値はすべて補正後の再ベースライン値。

| 検証 | 結論 |
|------|------|
| 予測可能性 | **あり**。v2.1 HAR+VI+レンジ の OOS R² = **+0.397** (市場のVI単体 +0.332・v2.0仕様の市場比較) |
| モデル vs 市場 | モデルが**有意に正確** (二乗誤差差95%CI [-18.66, -1.45]・負・v2.0仕様) |
| アンサンブル vs 個別 | HAR単体と同水準(+0.395 vs +0.397)。いずれのホライズンでも劣らず |
| 自作実装の妥当性 | statsmodelsと機械精度一致・未来参照ゼロを直接再計算で検証・**GARCH(定番OSS)に劣らない** |
| VRP売買戦略 | **未証明**(後半CI [-852, 318]・キャプチャはマイナス方向)。テールリスク大 |
| プラセボ検定 | VIの限界貢献(+0.032)は n=440 では有意未満 — 限界として明記 |

詳細: [`results/vol_report.md`](results/vol_report.md) /
[`results/forecast_v2_report.md`](results/forecast_v2_report.md) /
[`results/validation_report.md`](results/validation_report.md)

## 使い方

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/fetch_nk225.py          # データ取得(nikkei225jp.com 公開データ)
.venv/bin/python scripts/vol_backtest.py         # 検証1: 市場比較+VRP戦略
.venv/bin/python scripts/validate_model.py       # 検証2: 自作モデル妥当性バッテリー
.venv/bin/python scripts/verify_forecast_v2.py   # 検証3: 経路・区間・合議の検証
.venv/bin/python scripts/predict_nk225.py --refresh   # 予測実行(最新データ)
.venv/bin/python scripts/render_dashboard.py    # ダッシュボード生成(results/dashboard.html)
.venv/bin/python scripts/model_health.py        # 月次: 健全性チェック(results/health_log.jsonl)
.venv/bin/python scripts/feature_ext_test.py    # 特徴量拡張のウォークフォワード検証
.venv/bin/python scripts/feature_oos_check.py   # 月次: 特徴量拡張のOOS確認(月初営業日のみ)
.venv/bin/python scripts/forecast_vs_actual.py  # 日次: 予測vs実績の突合・外れの原因分類
.venv/bin/python scripts/long_vol_analysis.py   # 買いシグナル(long-vol週)の履歴検証
.venv/bin/python scripts/straddle_backtest.py   # 買いシグナル週の実ストラドル検証(コスト込み)
.venv/bin/python scripts/fetch_external.py      # FREDからVIX/SP500/USDJPY取得(検証2用)
.venv/bin/python scripts/feature_ext2_test.py   # 特徴量拡張検証2: 海外市場変数
.venv/bin/python scripts/feature_ext3_test.py   # 特徴量拡張検証3: イベントダミー
```

推奨運用: 週1回 `predict_nk225.py --refresh` (月次再推定が検証と整合)。

## データ

| ファイル | 内容 | 期間 |
|---------|------|------|
| 日経225先物 日足OHLC | nikkei225jp.com 公開データ | 1990年〜 (9,025日) |
| 日経VI(日次) | 同上 | 2016年9月〜 |

※ 同サイトの金曜分の足は「日曜ラベル(+2日)」で保存されるため、実日付に復元して使用。

## 検証方法の特徴

- **未来参照ゼロ**: 月次再推定のウォークフォワード。未来参照なしは直接再計算で機械的に検証
- **リーク防止**: 特徴量は予測時点で既知の情報のみ
- **プラセボ検定**: 入力を破壊して予測力が消えるかを確認(モデルが偶然の相関を拾っていないか)
- **確立OSSとの対戦**: GARCH(1,1) と同一条件で対戦(自作が劣らないことを確認)

## 注意

- 予測対象は**確率(ボラティリティ)**であり、個別の価格変動ではない
- VRPスタンス判定は統計的に未証明の部分を含む。実オプション取引では
  スキュー・売買コスト・証拠金が別途効く
- 本プロジェクトは統計学・機械学習の**学習目的**であり、投資助言ではない

## 関連プロジェクト

- [loto-analysis](https://github.com/GuangxiZheng/loto-analysis) — ロト6/7は予測不可能であることの実証
- boatrace-analysis — ボートレースは予測可能であり市場を僅かに上回ることの検証(ローカルのみ)
