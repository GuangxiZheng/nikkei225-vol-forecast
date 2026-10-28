#!/usr/bin/env python
"""日次ダッシュボード生成: results/dashboard.html (自己完結HTML・外部依存ゼロ)。

「売買判断」ではなく「リスク環境と予測の信頼度」を一目で見るための板。
  - ファンチャート: 経路予測(アンサンブル/HAR/GARCH) + 80%区間の帯 + VI/RV基準線
  - 履歴チャート: 直近の rv_w / rv_m / VI の推移
  - VRPギャップ: VI(含み) vs 予測実現ボラ — スタンスは「未証明」の参考表示
  - 合議: HAR vs GARCH 乖離率と30%閾値
  - 健全性: results/health_log.jsonl の履歴(未実行なら次回予定日を表示)

使い方: .venv/bin/python scripts/render_dashboard.py
(先に predict_nk225.py --refresh で results/model_latest.json を更新しておくこと)
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vol_backtest import build_features, load_data  # noqa: E402
from volmodel import target_h, walkforward_direct  # noqa: E402

LATEST = ROOT / "results" / "model_latest.json"
HEALTH_LOG = ROOT / "results" / "health_log.jsonl"
OUT = ROOT / "results" / "dashboard.html"

C_ENS, C_HAR, C_GAR = "#1565c0", "#6a1b9a", "#ef6c00"
C_VI, C_GRID, C_TXT = "#2e7d32", "#e0e0e0", "#333"


class Scale:
    def __init__(self, dmin, dmax, rmin, rmax):
        self.dmin, self.dmax, self.rmin, self.rmax = dmin, dmax, rmin, rmax

    def __call__(self, v):
        return self.rmin + (v - self.dmin) / (self.dmax - self.dmin) * (self.rmax - self.rmin)


def polyline(pts, stroke, width=2.0, dash=None):
    segs, run = [], []
    for p in pts:  # NaN 区切りで連続区間ごとに分ける
        if p is None:
            if len(run) > 1:
                segs.append(run)
            run = []
        else:
            run.append(p)
    if len(run) > 1:
        segs.append(run)
    dashattr = f' stroke-dasharray="{dash}"' if dash else ""
    return "".join(
        f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in seg)}" fill="none" '
        f'stroke="{stroke}" stroke-width="{width}"{dashattr} '
        'stroke-linejoin="round" stroke-linecap="round"/>'
        for seg in segs
    )


def dot(x, y, color, label):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}">'
            f'<title>{label}</title></circle>')


def fan_chart(path, feat):
    W, H, L, R, T, B = 880, 330, 64, 130, 24, 46
    hi_max = max(p["pi80_hi"] for p in path)
    ymax = max(hi_max, feat["vi"], feat["rv_w"]) + 4
    sx, sy = Scale(0, 5, L, W - R), Scale(0, ymax, H - B, T)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             'aria-label="経路予測ファンチャート">']
    # 横グリッド+目盛
    for v in range(0, int(ymax) + 1, 10):
        y = sy(v)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{C_GRID}"/>')
        parts.append(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" '
                     f'fill="{C_TXT}">{v}%</text>')
    # 80%区間の帯
    band = [(sx(p["h"]), sy(p["pi80_hi"])) for p in path] + \
           [(sx(p["h"]), sy(p["pi80_lo"])) for p in reversed(path)]
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in band)
    parts.append(f'<polygon points="{d}" fill="{C_ENS}" opacity="0.14">'
                 '<title>80%予測区間 (実カバレッジ64-73%で名目より狭い・目安)</title></polygon>')
    # 基準線 (現在の水準)
    for val, label, color in [(feat["vi"], f'日経VI {feat["vi"]:.1f}%', C_VI),
                              (feat["rv_w"], f'週次RV {feat["rv_w"]:.1f}%', "#777"),
                              (feat["rv_m"], f'月次RV {feat["rv_m"]:.1f}%', "#aaa")]:
        y = sy(val)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" '
                     f'stroke="{color}" stroke-dasharray="5,4" opacity="0.8"/>')
        parts.append(f'<text x="{W-R+6}" y="{y+4:.1f}" font-size="12" fill="{color}">{label}</text>')
    # 3ロジックの経路
    for key, color in [("ensemble", C_ENS), ("garch", C_GAR), ("har_vi", C_HAR)]:
        pts = [(sx(p["h"]), sy(p[key])) for p in path]
        parts.append(polyline(pts, color, 2.5 if key == "ensemble" else 1.6))
        parts.extend(dot(sx(p["h"]), sy(p[key]), color,
                         f'{p["h"]}日先 {key}: {p[key]:.1f}%') for p in path)
    # x軸
    for h in (1, 2, 3, 5):
        x = sx(h)
        parts.append(f'<line x1="{x:.1f}" y1="{H-B}" x2="{x:.1f}" y2="{H-B+5}" stroke="{C_TXT}"/>')
        parts.append(f'<text x="{x:.1f}" y="{H-B+20}" text-anchor="middle" font-size="12" '
                     f'fill="{C_TXT}">{h}日先</text>')
    parts.append('</svg>')
    return "".join(parts)


def history_chart(hist):
    W, H, L, R, T, B = 880, 260, 64, 20, 14, 34
    vals = hist[["rv_w", "rv_m", "vi"]].to_numpy()
    ymax = float(np.nanmax(vals) + 4)
    sx, sy = Scale(0, len(hist) - 1, L, W - R), Scale(0, ymax, H - B, T)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             'aria-label="ボラティリティ履歴">']
    for v in range(0, int(ymax) + 1, 10):
        y = sy(v)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{C_GRID}"/>')
        parts.append(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" '
                     f'fill="{C_TXT}">{v}%</text>')
    for col, color in [("vi", C_VI), ("rv_w", C_ENS), ("rv_m", "#999")]:
        pts = [None if pd.isna(v) else (sx(i), sy(v))
               for i, v in enumerate(hist[col].to_numpy())]
        parts.append(polyline(pts, color, 1.6))
    last = hist.iloc[-1]
    parts.append(dot(sx(len(hist) - 1), sy(last["vi"]), C_VI,
                     f'最新VI {last["vi"]:.1f}% ({last["date"]:%m/%d})'))
    for i in (0, len(hist) // 2, len(hist) - 1):
        x = sx(i)
        anchor = "end" if i == len(hist) - 1 else "middle"
        parts.append(f'<text x="{x:.1f}" y="{H-B+20}" text-anchor="{anchor}" font-size="12" '
                     f'fill="{C_TXT}">{hist.iloc[i]["date"]:%Y-%m}</text>')
    parts.append('</svg>')
    return "".join(parts)


def hbar(label, value, vmax, color, suffix="%"):
    w = max(2, min(100, value / vmax * 100))
    return (f'<div class="hbar"><span class="hlabel">{label}</span>'
            f'<span class="htrack"><span class="hfill" style="width:{w:.0f}%;'
            f'background:{color}"></span></span>'
            f'<span class="hval">{value:.1f}{suffix}</span></div>')


def badge(text, color):
    return (f'<span class="badge" style="background:{color}1a;color:{color};'
            f'border:1px solid {color}">{text}</span>')


def gap_chart(wk):
    """買い筋条件 (OOS予測h5 − VI) の週次推移。0の線を上回る週が理論上の買い筋。"""
    W, H, L, R, T, B = 880, 170, 64, 20, 12, 30
    gaps = (wk["wf5"] - wk["vi"]).to_numpy()
    amax = max(abs(float(gaps.min())), abs(float(gaps.max()))) * 1.15 + 1
    sx, sy = Scale(0, len(gaps) - 1, L, W - R), Scale(-amax, amax, H - B, T)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             'aria-label="買い筋条件の推移">']
    y0 = sy(0)
    parts.append(f'<line x1="{L}" y1="{y0:.1f}" x2="{W-R}" y2="{y0:.1f}" '
                 'stroke="#c62828" stroke-dasharray="6,4"/>')
    parts.append(f'<text x="{W-R-4}" y="{y0-6:.1f}" text-anchor="end" font-size="12" '
                 'fill="#c62828">この線より上=買い筋条件(参考)</text>')
    for v in (-10, 10):
        y = sy(v)
        if T < y < H - B:
            parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{C_GRID}"/>')
            parts.append(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" font-size="12" '
                         f'fill="{C_TXT}">{v:+d}pt</text>')
    pts = [(sx(i), sy(g)) for i, g in enumerate(gaps)]
    parts.append(polyline(pts, C_ENS, 1.8))
    last_d = wk.iloc[-1]["date"]
    parts.append(dot(sx(len(gaps) - 1), sy(gaps[-1]), C_ENS,
                     f'直近 {gaps[-1]:+.1f}pt ({last_d:%m/%d})'))
    for i in (0, len(gaps) // 2, len(gaps) - 1):
        anchor = "end" if i == len(gaps) - 1 else "middle"
        parts.append(f'<text x="{sx(i):.1f}" y="{H-B+20}" text-anchor="{anchor}" '
                     f'font-size="12" fill="{C_TXT}">{wk.iloc[i]["date"]:%Y-%m}</text>')
    parts.append('</svg>')
    return "".join(parts)


def practical_panel(model, wk):
    """最上部の実務パネル: 「どこで・いつ買うか」を先頭に置く。"""
    feat = model["features"]
    h5 = model["path_forecast"][-1]
    diff = h5["ensemble"] - feat["vi"]
    ok = diff > 0
    color = "#2e7d32" if ok else "#78909c"
    status = ("予測が含みを上回る — 理論上の買い筋(参考)" if ok
              else "含みが予測を上回る — 買い筋の条件未達")
    gaps = (wk["wf5"] - wk["vi"]).to_numpy()
    n_pos = int((gaps > 0).sum())
    cur = float(gaps[-1])
    remain = (f"達成まで あと{abs(cur):.1f}pt (VIの低下か予測の上昇で達成)"
              if cur <= 0 else "現在 達成中")
    strict = h5["ensemble"] ** 2 - feat["vi"] ** 2
    sig_ok = strict > 200
    sig_color = "#2e7d32" if sig_ok else "#90a4ae"
    return f"""<div class="card wide">
  <h2>実務: ボラティリティは「どこで・いつ」買うか (SBI証券の場合)</h2>
  <div class="two">
    <div>
      <h3>どこで買うか</h3>
      <ul>
        <li><b>日経225オプション</b>(大阪取引所): コール+プットの同時買い(ストラドル)が
          ボラ買いの基本形。日経VIはこのオプション市場から算出される。
          SBIでは別途「先物・オプション取引口座」の開設が必要。</li>
        <li><b>日経225ミニオプション</b>(1/10サイズ): 小口向け。</li>
        <li>日経VIそのものに直接連動する上場商品は個人口座にはない
          (VI先物は個人取引不可・VIX系ETNは米国VIX連動の別物)。</li>
        <li>実現ボラを直接狙う(先物でデルタヘッジするガンマスキャルプ)は上級者向け。</li>
      </ul>
    </div>
    <div>
      <h3>いつ買うか — モデルの参考条件</h3>
      <div class="note">理論上の買い筋: <b>5日先予測実現ボラ &gt; 日経VI</b>
        (オプションが理論的に割安な時)</div>
      <p style="color:{color};font-weight:700;margin:10px 0 4px">
        現在地: 予測 {h5["ensemble"]:.1f}% vs VI {feat["vi"]:.1f}%
        (差 {diff:+.1f}pt) → {status}</p>
      <div class="note">下の監視ラインが0を上回る週が買い筋の目安。
        直近{len(gaps)}回の週次観測(祝日等で金曜が欠落する週あり)中、
        条件達成は <b>{n_pos}回 ({n_pos/len(gaps):.0%})</b>。</div>
      <div class="note" style="margin-top:8px">厳格シグナル (参照:
        <b>results/long_vol_report.md</b>): 予測²−VI² = <b>{strict:+.0f}</b>
        (閾値 +200) → <b style="color:{sig_color}">{'条件達成 — 過去の点灯は損失が多く要精査' if sig_ok else '未点灯'}</b>。
        歴史上の点灯は2回のみ(2018-02・2020-02)で、コスト込みでは損失 — 現時点で買い根拠は未検証。</div>
    </div>
  </div>
  <h3 style="margin:14px 0 8px">買い筋条件の監視ライン: 予測実現ボラ(h=5) − 日経VI
    (週次・OOS予測は本番モデルv2.1版)</h3>
  {gap_chart(wk)}
  <div class="legend">現在地 <b>{cur:+.1f}pt</b> — {remain}。線が0を上回った週に
    オプション(ストラドル)を検討する、がモデル整合な待ち方。</div>
  <div class="warn">⚠ この条件自体は検証で利益未証明(VRP戦略のCIがゼロを含む)。
  投資助言ではありません。取引するならサイズを予測ボラに反比例させて小さく。</div>
</div>"""


def main():
    model = json.loads(LATEST.read_text())
    feat = model["features"]
    path = model["path_forecast"]
    df = build_features(load_data())
    hist = df.tail(180)

    # 買い筋条件の週次推移 (OOS予測h5 − VI、金曜時点・直近104週)
    df["y5"] = target_h(df, 5)
    wf5 = walkforward_direct(df, 5)
    wk = df[(df["date"] >= "2018-01-01") & (df["date"].dt.dayofweek == 4)].copy()
    wk["wf5"] = wf5.reindex(wk.index)
    wk = wk.dropna(subset=["wf5", "vi"]).tail(104)

    next_health = date.today() + timedelta(days=1)
    for _ in range(45):
        if next_health.weekday() == 5 and next_health.day <= 7:
            break
        next_health += timedelta(days=1)

    h5 = path[-1]
    gap = model["consensus"]["rel_gap"]
    verdict = model["consensus"]["verdict"]
    v_col = "#2e7d32" if verdict == "合意" else "#c62828"
    stance = model["vrp"]["stance_tau200"]

    try:
        health = [json.loads(l) for l in HEALTH_LOG.read_text().splitlines() if l.strip()]
    except FileNotFoundError:
        health = []
    if health:
        v_color = {"PASS": "#2e7d32", "WARN": "#b26a00", "FAIL": "#c62828"}
        rows = "".join(
            f'<tr><td>{e["date"]}</td>'
            f'<td>{e["r2_5d_harvi"]:+.3f}</td>'
            f'<td>{"—" if e.get("r2_5d_garch") is None else f"{e["r2_5d_garch"]:+.3f}"}</td>'
            f'<td>{e["coverage80"]:.1%}</td>'
            f'<td>{"—" if e.get("gap_median") is None else f"{e["gap_median"]:.1%}"}</td>'
            f'<td style="color:{v_color.get(e.get("verdict"), C_TXT)};font-weight:600">'
            f'{e.get("verdict", "—")}</td></tr>'
            for e in health[-6:])
        health_html = (f'<table><tr><th>日付</th><th>OOS R² (HAR+VI)</th><th>OOS R² (GARCH)</th>'
                       f'<th>80%カバレッジ</th><th>乖離中央値</th><th>判定</th></tr>{rows}</table>')
    else:
        health_html = (f'<p class="note">初回の月次健全性チェックは未実行です。'
                       f'次回予定: <b>{next_health:%Y-%m-%d} (土) 12:30</b> — '
                       f'直近26週の OOS R²・区間カバレッジ・乖離率を検査し '
                       f'PASS/WARN/FAIL を results/health_log.jsonl に記録します。</p>')

    html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NK225 ボラティリティ予測ダッシュボード</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, "Hiragino Sans", "Noto Sans JP", sans-serif;
         margin: 24px auto; max-width: 960px; padding: 0 16px; color: {C_TXT}; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .sub {{ color: #777; font-size: 13px; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .card {{ border: 1px solid #e5e5e5; border-radius: 10px; padding: 16px 18px; background: #fff; }}
  .wide {{ grid-column: 1 / -1; }}
  h2 {{ font-size: 14px; margin: 0 0 12px; color: #555; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 12px; font-weight: 600; margin-left: 8px; }}
  .kpis {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0 20px; }}
  .kpi {{ border: 1px solid #e5e5e5; border-radius: 10px; padding: 10px 16px; min-width: 130px; }}
  .kpi .v {{ font-size: 22px; font-weight: 700; }}
  .kpi .l {{ font-size: 12px; color: #777; }}
  .hbar {{ display: flex; align-items: center; gap: 10px; margin: 8px 0; font-size: 13px; }}
  .hlabel {{ width: 150px; text-align: right; color: #555; }}
  .htrack {{ flex: 1; height: 14px; background: #f0f0f0; border-radius: 7px; overflow: hidden; }}
  .hfill {{ display: block; height: 100%; border-radius: 7px; }}
  .hval {{ width: 70px; font-weight: 600; }}
  .thresh {{ border-left: 3px solid #c62828; padding-left: 6px; margin-left: 160px; }}
  .note {{ font-size: 13px; color: #666; }}
  .warn {{ font-size: 12px; color: #b26a00; background: #fff7e6; border: 1px solid #f0d9a8;
           border-radius: 8px; padding: 8px 12px; margin-top: 10px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #eee; padding: 6px 8px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: #777; font-weight: 600; }}
  .legend {{ font-size: 12px; color: #666; margin-top: 6px; }}
  .legend i {{ display: inline-block; width: 18px; height: 3px; border-radius: 2px;
               vertical-align: middle; margin: 0 4px 0 12px; }}
  .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  .card h3 {{ font-size: 13px; margin: 0 0 8px; color: #555; }}
  ul {{ margin: 0; padding-left: 18px; font-size: 13px; line-height: 1.7; }}
</style></head><body>
<h1>日経225 ボラティリティ予測ダッシュボード</h1>
<div class="sub">データ時点 {model["data_asof"]} ／ 生成 {date.today():%Y-%m-%d} ／
リスク環境と予測の信頼度を見るための板(売買シグナルではありません)</div>
<div class="kpis">
  <div class="kpi"><div class="v">{feat["vi"]:.1f}%</div><div class="l">日経VI (含み波動度)</div></div>
  <div class="kpi"><div class="v">{h5["ensemble"]:.1f}%</div><div class="l">5日先予測(アンサンブル)</div></div>
  <div class="kpi"><div class="v">{gap*100:.1f}%</div><div class="l">2ロジック乖離率</div>
    {badge("合意" if verdict == "合意" else "乖離大(精度低下注意)", v_col)}</div>
  <div class="kpi"><div class="v">{stance}</div><div class="l">VRPスタンス(未証明・参考)</div></div>
</div>
{practical_panel(model, wk)}
<div class="grid">
  <div class="card wide">
    <h2>経路予測ファンチャート (実現ボラ 年率%・80%区間)</h2>
    {fan_chart(path, feat)}
    <div class="legend"><i style="background:{C_ENS}"></i>アンサンブル
      <i style="background:{C_HAR}"></i>HAR+VI <i style="background:{C_GAR}"></i>GARCH
      <i style="background:{C_VI};height:1px"></i>日経VI(基準)
      ／帯=80%予測区間(実カバレッジ64〜73%と名目より狭い)</div>
  </div>
  <div class="card wide">
    <h2>ボラティリティ履歴 (直近180営業日)</h2>
    {history_chart(hist)}
    <div class="legend"><i style="background:{C_ENS}"></i>週次RV
      <i style="background:#999"></i>月次RV <i style="background:{C_VI}"></i>日経VI</div>
  </div>
  <div class="card">
    <h2>VRPギャップ: 含み vs 予測 (h=5基準)</h2>
    {hbar("日経VI (市場の含み)", feat["vi"], 50, C_VI)}
    {hbar("予測実現ボラ (h=5)", h5["ensemble"], 50, C_ENS)}
    <div class="note">VRPシグナル <b>{model["vrp"]["signal"]:+.0f}</b> → スタンス <b>{stance}</b></div>
    <div class="warn">⚠ VRP売買戦略は検証で未証明(CIがゼロを含む・テールリスク大)。
    この欄は売買判断として使用しないこと。</div>
  </div>
  <div class="card">
    <h2>2ロジック合議 (乖離&gt;30%で精度低下フラグ)</h2>
    {hbar("HAR+VI (h=5)", h5["har_vi"], 50, C_HAR)}
    {hbar("GARCH (h=5)", h5["garch"], 50, C_GAR)}
    {hbar("乖離率", gap*100, 50, v_col if gap > 0.30 else "#90a4ae", suffix="%")}
    <div class="thresh" style="margin-left:0;padding-left:0;border:none;color:#c62828;
         font-size:12px">| 閾値 30% — 乖離週は誤差が約1.2pt拡大(検証済み)</div>
    <div class="note">判定: <b style="color:{v_col}">{verdict}</b></div>
  </div>
  <div class="card wide">
    <h2>モデル健全性 (毎月第1土曜 12:30 自動チェック)</h2>
    {health_html}
  </div>
</div>
</body></html>"""
    OUT.write_text(html)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
