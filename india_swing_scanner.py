from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from universe_lists import UNIVERSE_CONFIG


LOOKBACK_BARS = 220
FETCH_PERIOD = "500d"
MIN_PRICE_INR = 50.0
MIN_TRADED_VALUE_20D = 250_000_000.0
OUTPUT_FILENAME = "india_swing_report.html"
PAGES_FILENAME = "index.html"


def resolve_universe(universe: str) -> tuple[str, dict]:
    key = universe.lower()
    if key not in UNIVERSE_CONFIG:
        key = "india-largecap"
    return key, UNIVERSE_CONFIG[key]


def download_history(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=tickers,
        period=FETCH_PERIOD,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
        actions=False,
    )
    if data.empty:
        raise RuntimeError("No data returned from Yahoo Finance.")

    history: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in data.columns.get_level_values(0):
                continue
            history[ticker] = normalize_history_frame(data[ticker].copy())
    else:
        history[tickers[0]] = normalize_history_frame(data.copy())
    return history


def normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    keep = [column for column in ["open", "high", "low", "close", "volume"] if column in renamed.columns]
    cleaned = renamed[keep].dropna().copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    if getattr(cleaned.index, "tz", None) is not None:
        cleaned.index = cleaned.index.tz_convert(None)
    return cleaned


def sma(values: list[float], period: int) -> float:
    return sum(values[-period:]) / period


def ema_series(values: list[float], period: int) -> list[float]:
    seed = sum(values[:period]) / period
    multiplier = 2 / (period + 1)
    result = [seed]
    for value in values[period:]:
        result.append((value - result[-1]) * multiplier + result[-1])
    return result


def rsi(values: list[float], period: int = 14) -> float:
    gains: list[float] = []
    losses: list[float] = []
    for idx in range(1, period + 1):
        change = values[idx] - values[idx - 1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for idx in range(period + 1, len(values)):
        change = values[idx] - values[idx - 1]
        gain = max(change, 0)
        loss = abs(min(change, 0))
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(values: list[float]) -> tuple[float, float]:
    ema12 = ema_series(values, 12)
    ema26 = ema_series(values, 26)
    offset = len(ema12) - len(ema26)
    line = [ema12[idx + offset] - ema26[idx] for idx in range(len(ema26))]
    signal = ema_series(line, 9)
    return line[-1], signal[-1]


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    true_ranges: list[float] = []
    for idx, _close in enumerate(closes):
        if idx == 0:
            true_ranges.append(highs[idx] - lows[idx])
            continue
        prev_close = closes[idx - 1]
        true_ranges.append(max(highs[idx] - lows[idx], abs(highs[idx] - prev_close), abs(lows[idx] - prev_close)))
    value = sum(true_ranges[:period]) / period
    for current in true_ranges[period:]:
        value = ((value * (period - 1)) + current) / period
    return value


def pct_change(new: float, old: float) -> float:
    return 0.0 if old == 0 else ((new - old) / old) * 100


def format_display_ticker(ticker: str) -> str:
    return ticker.removesuffix(".NS")


def hold_window(setup: str, atr_pct: float, rs_value: float) -> str:
    if setup == "Avoid":
        return "No swing setup"
    low, high = {"Breakout": (5, 12), "Pullback": (4, 10), "Trend Continuation": (7, 15)}.get(setup, (3, 8))
    if atr_pct > 4.5:
        high -= 2
    if rs_value > 4:
        high += 2
    if rs_value < 0:
        low = max(2, low - 1)
        high = max(low + 2, high - 2)
    return f"{low}-{high} trading days"


def classify_stock(
    ticker: str,
    history: pd.DataFrame,
    benchmark_closes: list[float],
    latest_date: str,
) -> dict | None:
    if len(history) < LOOKBACK_BARS or len(benchmark_closes) < LOOKBACK_BARS:
        return None

    closes = history["close"].tolist()[-LOOKBACK_BARS:]
    highs = history["high"].tolist()[-LOOKBACK_BARS:]
    lows = history["low"].tolist()[-LOOKBACK_BARS:]
    volumes = history["volume"].tolist()[-LOOKBACK_BARS:]

    latest_close = float(closes[-1])
    avg_traded_value = sum(c * v for c, v in zip(closes[-20:], volumes[-20:])) / 20
    if latest_close < MIN_PRICE_INR or avg_traded_value < MIN_TRADED_VALUE_20D:
        return None

    ema20 = ema_series(closes, 20)[-1]
    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200)
    rsi14 = rsi(closes, 14)
    macd_line, macd_signal = macd(closes)
    atr14 = atr(highs, lows, closes, 14)
    atr_pct = (atr14 / latest_close) * 100 if latest_close else 0.0
    avg_volume20 = sum(volumes[-20:]) / 20
    volume_ratio = volumes[-1] / avg_volume20 if avg_volume20 else 0.0
    prev_20_high = max(highs[-21:-1])
    prev_20_low = min(lows[-21:-1])
    rs_vs_benchmark = pct_change(closes[-1], closes[-21]) - pct_change(benchmark_closes[-1], benchmark_closes[-21])

    trend_aligned = ema20 > sma50 > sma200
    above50 = latest_close > sma50
    above200 = latest_close > sma200
    near_breakout = latest_close >= prev_20_high * 0.985
    pullback_hold = latest_close > ema20 and min(closes[-5:]) <= ema20 * 1.01
    bullish_rsi = 50 <= rsi14 <= 68
    extended_rsi = rsi14 > 72
    bullish_macd = macd_line > macd_signal
    healthy_volume = volume_ratio >= 1.0
    strong_rs = rs_vs_benchmark > 0

    score = 0
    score += 15 if above50 else 0
    score += 10 if above200 else 0
    score += 20 if trend_aligned else 0
    score += 15 if bullish_rsi else 0
    score += 10 if bullish_macd else 0
    score += 10 if healthy_volume else 0
    score += 10 if near_breakout else 0
    score += 10 if pullback_hold else 0
    score += 10 if strong_rs else 0
    score += 5 if atr_pct <= 4.5 else 0
    score -= 10 if extended_rsi else 0
    score -= 15 if latest_close < prev_20_low else 0

    setup = "Watch"
    if score >= 75 and near_breakout:
        setup = "Breakout"
    elif score >= 65 and pullback_hold:
        setup = "Pullback"
    elif score >= 55:
        setup = "Trend Continuation"
    elif latest_close < sma50 or latest_close < sma200:
        setup = "Avoid"

    notes: list[str] = []
    if trend_aligned:
        notes.append("20 EMA > 50 SMA > 200 SMA")
    if near_breakout:
        notes.append("near 20-day high")
    if pullback_hold:
        notes.append("held 20 EMA on pullback")
    if bullish_macd:
        notes.append("MACD above signal")
    if bullish_rsi:
        notes.append("RSI in bullish zone")
    if healthy_volume:
        notes.append("volume above 20-day average")
    if strong_rs:
        notes.append("beating benchmark over 20 days")
    if extended_rsi:
        notes.append("RSI extended")

    return {
        "ticker": format_display_ticker(ticker),
        "symbol": ticker,
        "exchange": "NSE",
        "date": latest_date,
        "setup": setup,
        "score": score,
        "close": round(latest_close, 2),
        "ema20": round(ema20, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "rsi14": round(rsi14, 1),
        "macd": round(macd_line, 2),
        "macdSignal": round(macd_signal, 2),
        "atrPct": round(atr_pct, 2),
        "volumeRatio": round(volume_ratio, 2),
        "rsVsBenchmark20d": round(rs_vs_benchmark, 2),
        "avgTradedValue20dCr": round(avg_traded_value / 10_000_000, 1),
        "stopLoss": round(max(ema20 - (0.5 * atr14), latest_close - (1.5 * atr14)), 2),
        "target1": round(latest_close + (1.5 * atr14), 2),
        "target2": round(latest_close + (3.0 * atr14), 2),
        "holdWindow": hold_window(setup, atr_pct, rs_vs_benchmark),
        "goodForSwing": setup != "Avoid" and score >= 55,
        "notes": "; ".join(notes[:6]),
    }


def build_html(rows: list[dict], generated_at: datetime, coverage_note: str, benchmark_label: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>India Swing Scanner</title>
  <style>
    :root {{ --bg:#f5efe2; --panel:rgba(255,252,246,.93); --ink:#1b2d3a; --muted:#5e6f79; --line:#dacfbf; --green:#0f7b64; --amber:#b97824; --red:#a43f45; --blue:#254f74; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); font-family:Georgia, "Times New Roman", serif; background:radial-gradient(circle at 12% 0%, rgba(15,123,100,.12), transparent 28%), radial-gradient(circle at 88% 10%, rgba(185,120,36,.12), transparent 25%), linear-gradient(180deg,#fbf6ec 0%,var(--bg) 100%); }}
    .wrap {{ max-width:1440px; margin:0 auto; padding:28px 18px 42px; }}
    .hero {{ background:linear-gradient(135deg, rgba(16,48,64,.98), rgba(19,95,104,.92)); color:#fff8ef; border-radius:24px; padding:30px; box-shadow:0 20px 45px rgba(27,45,58,.18); }}
    .hero h1 {{ margin:0 0 10px; font-size:clamp(2rem,4vw,3.4rem); }}
    .hero p {{ margin:0; line-height:1.55; max-width:980px; color:rgba(255,248,239,.88); }}
    .hero-tools {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:18px; align-items:center; }}
    .stamp {{ display:inline-flex; align-items:center; gap:8px; padding:10px 14px; border-radius:999px; background:rgba(255,248,239,.14); color:#fff8ef; font-size:.95rem; }}
    .refresh-btn {{ appearance:none; border:none; border-radius:999px; padding:10px 16px; background:#fff8ef; color:#103040; font-weight:700; cursor:pointer; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin-top:20px; }}
    .metric,.controls,.table-wrap {{ background:var(--panel); border:1px solid var(--line); border-radius:20px; box-shadow:0 12px 30px rgba(27,45,58,.08); }}
    .metric {{ padding:18px; }}
    .label {{ color:var(--muted); text-transform:uppercase; letter-spacing:.08em; font-size:.78rem; }}
    .value {{ font-size:1.8rem; margin-top:8px; }}
    .controls {{ margin-top:20px; padding:18px; }}
    .control-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
    label {{ font-size:.84rem; color:var(--muted); display:block; margin-bottom:6px; text-transform:uppercase; letter-spacing:.07em; }}
    input,select {{ width:100%; padding:11px 12px; border-radius:12px; border:1px solid var(--line); background:#fffdf8; color:var(--ink); font-size:.96rem; }}
    .table-wrap {{ margin-top:20px; overflow:hidden; }}
    .table-scroll {{ overflow:auto; max-height:70vh; }}
    table {{ width:100%; border-collapse:collapse; min-width:1500px; }}
    th,td {{ padding:11px 10px; border-bottom:1px solid var(--line); text-align:left; font-size:.93rem; vertical-align:top; }}
    th {{ position:sticky; top:0; background:#ece4d3; cursor:pointer; z-index:1; }}
    tbody tr:hover {{ background:rgba(15,123,100,.06); }}
    .tag {{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:.78rem; font-weight:700; }}
    .Breakout {{ background:rgba(15,123,100,.15); color:var(--green); }}
    .Pullback {{ background:rgba(185,120,36,.16); color:#8d5916; }}
    .Trend {{ background:rgba(37,79,116,.14); color:var(--blue); }}
    .Avoid {{ background:rgba(164,63,69,.15); color:var(--red); }}
    .yes {{ color:var(--green); font-weight:700; }}
    .no {{ color:var(--red); font-weight:700; }}
    .foot {{ margin-top:14px; color:var(--muted); line-height:1.5; font-size:.95rem; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>India Swing Scanner</h1>
      <p>Daily swing-trade scanner for Indian equities using open Yahoo Finance data. It ranks NSE names by trend, momentum, confirmation, volatility, and relative strength versus {benchmark_label}. Generated {generated_at.strftime("%Y-%m-%d %H:%M")}.</p>
      <div class="hero-tools">
        <div class="stamp">Last Updated: {generated_at.strftime("%Y-%m-%d %H:%M")}</div>
        <button class="refresh-btn" type="button" onclick="window.location.reload()">Refresh View</button>
      </div>
    </section>
    <section class="metrics">
      <article class="metric"><div class="label">Coverage</div><div class="value" id="coverageCount">-</div><div>{coverage_note}</div></article>
      <article class="metric"><div class="label">Good Swing Setups</div><div class="value" id="goodCount">-</div><div>Score 55+ and not marked Avoid.</div></article>
      <article class="metric"><div class="label">Breakouts</div><div class="value" id="breakoutCount">-</div><div>Near 20-day highs with strong alignment.</div></article>
      <article class="metric"><div class="label">Pullbacks</div><div class="value" id="pullbackCount">-</div><div>Held the 20 EMA inside a larger uptrend.</div></article>
    </section>
    <section class="controls">
      <div class="control-grid">
        <div><label for="search">Search</label><input id="search" type="text" placeholder="Ticker"></div>
        <div><label for="setupFilter">Setup</label><select id="setupFilter"><option value="All">All</option><option value="Breakout">Breakout</option><option value="Pullback">Pullback</option><option value="Trend Continuation">Trend Continuation</option><option value="Avoid">Avoid</option></select></div>
        <div><label for="qualityFilter">Swing Quality</label><select id="qualityFilter"><option value="All">All</option><option value="Yes">Good for swing trade</option><option value="No">Not good right now</option></select></div>
        <div><label for="minScore">Minimum Score</label><input id="minScore" type="number" value="55" min="0" max="100" step="5"></div>
        <div><label for="maxAtr">Max ATR %</label><input id="maxAtr" type="number" value="6" min="1" max="20" step="0.5"></div>
        <div><label for="sortBy">Sort</label><select id="sortBy"><option value="score">Score</option><option value="rsVsBenchmark20d">Relative Strength</option><option value="volumeRatio">Volume Ratio</option><option value="atrPct">ATR %</option><option value="avgTradedValue20dCr">20D Value (Cr)</option><option value="close">Price</option></select></div>
      </div>
    </section>
    <section class="table-wrap"><div class="table-scroll"><table><thead><tr><th data-sort="ticker">Ticker</th><th data-sort="setup">Setup</th><th data-sort="goodForSwing">Good For Swing?</th><th data-sort="holdWindow">Hold Window</th><th data-sort="score">Score</th><th data-sort="close">Close</th><th data-sort="ema20">20 EMA</th><th data-sort="sma50">50 SMA</th><th data-sort="sma200">200 SMA</th><th data-sort="rsi14">RSI</th><th data-sort="macd">MACD</th><th data-sort="atrPct">ATR %</th><th data-sort="volumeRatio">Vol Ratio</th><th data-sort="avgTradedValue20dCr">20D Value (Cr)</th><th data-sort="rsVsBenchmark20d">RS vs Benchmark</th><th data-sort="stopLoss">Stop</th><th data-sort="target1">T1</th><th data-sort="target2">T2</th><th data-sort="notes">Notes</th></tr></thead><tbody id="reportBody"></tbody></table></div></section>
    <p class="foot">This report uses the latest completed daily bar from Yahoo Finance. Hold windows are heuristic ranges based on setup type, volatility, and relative strength, not guarantees. Please confirm earnings, corporate actions, and sector news before trading.</p>
  </div>
  <script>
    const rows = {json.dumps(rows)};
    const state = {{ sortBy: "score", sortDir: "desc" }};
    const els = {{
      body: document.getElementById("reportBody"),
      search: document.getElementById("search"),
      setupFilter: document.getElementById("setupFilter"),
      qualityFilter: document.getElementById("qualityFilter"),
      minScore: document.getElementById("minScore"),
      maxAtr: document.getElementById("maxAtr"),
      sortBy: document.getElementById("sortBy"),
      coverageCount: document.getElementById("coverageCount"),
      goodCount: document.getElementById("goodCount"),
      breakoutCount: document.getElementById("breakoutCount"),
      pullbackCount: document.getElementById("pullbackCount")
    }};
    function compare(a,b,key) {{ const av=a[key], bv=b[key]; if (typeof av === "number" && typeof bv === "number") return av-bv; return String(av).localeCompare(String(bv)); }}
    function setupClass(setup) {{ return setup === "Trend Continuation" ? "Trend" : setup; }}
    function render(filtered) {{
      els.body.innerHTML = filtered.map(row => `<tr><td><strong>${{row.ticker}}</strong><br><span style="color:var(--muted)">${{row.exchange}}</span></td><td><span class="tag ${{setupClass(row.setup)}}">${{row.setup}}</span></td><td class="${{row.goodForSwing ? "yes" : "no"}}">${{row.goodForSwing ? "Yes" : "No"}}</td><td>${{row.holdWindow}}</td><td>${{row.score}}</td><td>${{row.close.toFixed(2)}}</td><td>${{row.ema20.toFixed(2)}}</td><td>${{row.sma50.toFixed(2)}}</td><td>${{row.sma200.toFixed(2)}}</td><td>${{row.rsi14.toFixed(1)}}</td><td>${{row.macd.toFixed(2)}} / ${{row.macdSignal.toFixed(2)}}</td><td>${{row.atrPct.toFixed(2)}}%</td><td>${{row.volumeRatio.toFixed(2)}}x</td><td>${{row.avgTradedValue20dCr.toFixed(1)}}</td><td>${{row.rsVsBenchmark20d.toFixed(2)}}%</td><td>${{row.stopLoss.toFixed(2)}}</td><td>${{row.target1.toFixed(2)}}</td><td>${{row.target2.toFixed(2)}}</td><td>${{row.notes}}</td></tr>`).join("");
      els.coverageCount.textContent = `${filtered.length}`;
      els.goodCount.textContent = `${filtered.filter(row => row.goodForSwing).length}`;
      els.breakoutCount.textContent = `${filtered.filter(row => row.setup === "Breakout").length}`;
      els.pullbackCount.textContent = `${filtered.filter(row => row.setup === "Pullback").length}`;
    }}
    function applyFilters() {{
      const term = els.search.value.trim().toLowerCase();
      const setup = els.setupFilter.value;
      const quality = els.qualityFilter.value;
      const minScoreValue = Number(els.minScore.value || 0);
      const maxAtrValue = Number(els.maxAtr.value || 99);
      const filtered = rows.filter(row => (!term || row.ticker.toLowerCase().includes(term)) && (setup === "All" || row.setup === setup) && !(quality === "Yes" && !row.goodForSwing) && !(quality === "No" && row.goodForSwing) && row.score >= minScoreValue && row.atrPct <= maxAtrValue).sort((a,b) => state.sortDir === "asc" ? compare(a,b,state.sortBy) : compare(b,a,state.sortBy));
      render(filtered);
    }}
    document.querySelectorAll("th[data-sort]").forEach(th => th.addEventListener("click", () => {{ const key = th.dataset.sort; if (state.sortBy === key) state.sortDir = state.sortDir === "asc" ? "desc" : "asc"; else {{ state.sortBy = key; state.sortDir = "desc"; els.sortBy.value = key; }} applyFilters(); }}));
    [els.search, els.setupFilter, els.qualityFilter, els.minScore, els.maxAtr].forEach(el => {{ el.addEventListener("input", applyFilters); el.addEventListener("change", applyFilters); }});
    els.sortBy.addEventListener("change", () => {{ state.sortBy = els.sortBy.value; state.sortDir = "desc"; applyFilters(); }});
    applyFilters();
  </script>
</body>
</html>"""


def generate_report(universe: str = "india-largecap") -> dict:
    universe_key, config = resolve_universe(universe)
    tickers = list(config["members"])
    benchmark = config["benchmark"]
    history_map = download_history(tickers + [benchmark])
    benchmark_history = history_map.get(benchmark)
    if benchmark_history is None or len(benchmark_history) < LOOKBACK_BARS:
        raise RuntimeError(f"Benchmark data missing for {benchmark}.")

    benchmark_closes = benchmark_history["close"].tolist()[-LOOKBACK_BARS:]
    latest_date = benchmark_history.index[-1].strftime("%Y-%m-%d")
    rows: list[dict] = []

    for ticker in tickers:
        history = history_map.get(ticker)
        if history is None:
            continue
        row = classify_stock(ticker, history, benchmark_closes, latest_date)
        if row:
            rows.append(row)

    rows.sort(key=lambda row: (row["score"], row["rsVsBenchmark20d"]), reverse=True)
    generated_at = datetime.now()
    coverage_note = f"{config['label']} scanned on {latest_date}. {config['description']}"
    html = build_html(rows, generated_at, coverage_note, config["benchmark_label"])
    output_path = Path(__file__).with_name(OUTPUT_FILENAME)
    output_path.write_text(html, encoding="utf-8")
    Path(__file__).with_name(PAGES_FILENAME).write_text(html, encoding="utf-8")

    return {
        "output_path": str(output_path),
        "latest_date": latest_date,
        "universe_key": universe_key,
        "universe_label": config["label"],
        "universe_size": len(rows),
        "good_setups": sum(1 for row in rows if row["goodForSwing"]),
    }


def main() -> int:
    try:
        universe = sys.argv[1] if len(sys.argv) > 1 else "india-largecap"
        result = generate_report(universe)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Saved report to {result['output_path']}")
    print(f"Universe: {result['universe_label']}")
    print(f"Universe size: {result['universe_size']}")
    print(f"Good setups: {result['good_setups']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
