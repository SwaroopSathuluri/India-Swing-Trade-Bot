from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from pnsea import NSE


TICKERS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT",
    "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "PIDILITIND", "POWERGRID", "RELIANCE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

LOOKBACK_BARS = 220
MIN_PRICE = 50.0
MIN_TRADED_VALUE_20D = 250_000_000.0
CHUNK_DAYS = 65
TOTAL_CALENDAR_DAYS = 430


def fetch_symbol_history(nse: NSE, symbol: str, end_date: date) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    current_end = end_date
    start_limit = end_date - timedelta(days=TOTAL_CALENDAR_DAYS)
    while current_end > start_limit:
        current_start = max(start_limit, current_end - timedelta(days=CHUNK_DAYS - 1))
        df = nse.equity.delivery_history(
            symbol,
            current_start.strftime("%d-%m-%Y"),
            current_end.strftime("%d-%m-%Y"),
        )
        if not df.empty:
            frames.append(df)
        current_end = current_start - timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True)
    history = history.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return history


def pct_change(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100.0


def hold_window(setup: str, atr_pct: float, relative_strength: float) -> str:
    if setup == "Avoid":
        return "No swing setup"
    windows = {
        "Breakout": (5, 12),
        "Pullback": (4, 10),
        "Trend Continuation": (7, 15),
    }
    low, high = windows.get(setup, (3, 8))
    if atr_pct > 4.5:
        high -= 2
    if relative_strength > 4:
        high += 2
    if relative_strength < 0:
        low = max(2, low - 1)
        high = max(low + 2, high - 2)
    return f"{low}-{high} trading days"


def classify_symbol(symbol: str, history: pd.DataFrame, basket_return_20: float, latest_date: str) -> dict | None:
    if len(history) < LOOKBACK_BARS:
        return None

    working = history.tail(LOOKBACK_BARS).copy()
    closes = working["Close"].astype(float)
    highs = working["High"].astype(float)
    lows = working["Low"].astype(float)
    volumes = working["Volume"].astype(float)
    latest_close = float(closes.iloc[-1])
    avg_traded_value = float((closes.tail(20) * volumes.tail(20)).mean())

    if latest_close < MIN_PRICE or avg_traded_value < MIN_TRADED_VALUE_20D:
        return None

    ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    sma50 = float(closes.tail(50).mean())
    sma200 = float(closes.tail(200).mean())

    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi14 = float((100 - (100 / (1 + rs))).fillna(100).iloc[-1])

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_value = float(macd_line.iloc[-1])
    macd_signal_value = float(macd_signal.iloc[-1])

    tr_components = pd.concat(
        [
            highs - lows,
            (highs - closes.shift(1)).abs(),
            (lows - closes.shift(1)).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr14 = float(true_range.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    atr_pct = (atr14 / latest_close) * 100 if latest_close else 0.0

    avg_volume20 = float(volumes.tail(20).mean())
    volume_ratio = float(volumes.iloc[-1] / avg_volume20) if avg_volume20 else 0.0
    prev_20_high = float(highs.iloc[-21:-1].max())
    prev_20_low = float(lows.iloc[-21:-1].min())
    stock_return_20 = pct_change(float(closes.iloc[-1]), float(closes.iloc[-21]))
    rs_vs_basket = stock_return_20 - basket_return_20

    trend_aligned = ema20 > sma50 > sma200
    above50 = latest_close > sma50
    above200 = latest_close > sma200
    near_breakout = latest_close >= prev_20_high * 0.985
    pullback_hold = latest_close > ema20 and float(closes.tail(5).min()) <= ema20 * 1.01
    bullish_rsi = 50 <= rsi14 <= 68
    extended_rsi = rsi14 > 72
    bullish_macd = macd_value > macd_signal_value
    healthy_volume = volume_ratio >= 1.0
    strong_rs = rs_vs_basket > 0

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
        notes.append("beating India large-cap basket over 20 days")
    if extended_rsi:
        notes.append("RSI extended")

    stop_loss = max(ema20 - (0.5 * atr14), latest_close - (1.5 * atr14))
    target1 = latest_close + (1.5 * atr14)
    target2 = latest_close + (3.0 * atr14)

    return {
        "ticker": symbol,
        "exchange": "NSE",
        "date": latest_date,
        "setup": setup,
        "score": score,
        "close": round(latest_close, 2),
        "ema20": round(ema20, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "rsi14": round(rsi14, 1),
        "macd": round(macd_value, 2),
        "macdSignal": round(macd_signal_value, 2),
        "atrPct": round(atr_pct, 2),
        "volumeRatio": round(volume_ratio, 2),
        "avgTradedValue20dCr": round(avg_traded_value / 10_000_000, 1),
        "rsVsBenchmark20d": round(rs_vs_basket, 2),
        "stopLoss": round(stop_loss, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "holdWindow": hold_window(setup, atr_pct, rs_vs_basket),
        "goodForSwing": setup != "Avoid" and score >= 55,
        "notes": "; ".join(notes[:6]),
    }


def build_html(rows: list[dict], generated_at: datetime, latest_date: str) -> str:
    data_json = json.dumps(rows, separators=(",", ":"))
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
      <p>Real-data swing-trade scanner for Indian large-cap equities using NSE historical data. It ranks names by trend, momentum, volatility, and relative strength versus the India large-cap basket. Generated {generated_at.strftime("%Y-%m-%d %H:%M")}.</p>
      <div class="hero-tools">
        <div class="stamp">Last Updated: {generated_at.strftime("%Y-%m-%d %H:%M")}</div>
        <div class="stamp">Benchmark: India large-cap basket</div>
      </div>
    </section>
    <section class="metrics">
      <article class="metric"><div class="label">Coverage</div><div class="value" id="coverageCount">-</div><div>India large-cap leaders scanned on {latest_date}.</div></article>
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
    <section class="table-wrap"><div class="table-scroll"><table><thead><tr><th data-sort="ticker">Ticker</th><th data-sort="setup">Setup</th><th data-sort="goodForSwing">Good For Swing?</th><th data-sort="holdWindow">Hold Window</th><th data-sort="score">Score</th><th data-sort="close">Close</th><th data-sort="ema20">20 EMA</th><th data-sort="sma50">50 SMA</th><th data-sort="sma200">200 SMA</th><th data-sort="rsi14">RSI</th><th data-sort="macd">MACD</th><th data-sort="atrPct">ATR %</th><th data-sort="volumeRatio">Vol Ratio</th><th data-sort="avgTradedValue20dCr">20D Value (Cr)</th><th data-sort="rsVsBenchmark20d">RS vs Basket</th><th data-sort="stopLoss">Stop</th><th data-sort="target1">T1</th><th data-sort="target2">T2</th><th data-sort="notes">Notes</th></tr></thead><tbody id="reportBody"></tbody></table></div></section>
    <p class="foot">This report uses daily NSE historical data. Hold windows are heuristic ranges based on setup type, volatility, and relative strength, not guarantees. Please confirm earnings, corporate actions, and sector news before trading.</p>
  </div>
  <script>
    const rows = {data_json};
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
      els.coverageCount.textContent = `${{filtered.length}}`;
      els.goodCount.textContent = `${{filtered.filter(row => row.goodForSwing).length}}`;
      els.breakoutCount.textContent = `${{filtered.filter(row => row.setup === "Breakout").length}}`;
      els.pullbackCount.textContent = `${{filtered.filter(row => row.setup === "Pullback").length}}`;
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


def main() -> None:
    nse = NSE()
    end_date = date.today()
    histories: dict[str, pd.DataFrame] = {}

    for symbol in TICKERS:
        print(f"Fetching {symbol} ...")
        df = fetch_symbol_history(nse, symbol, end_date)
        if not df.empty and len(df) >= LOOKBACK_BARS:
            histories[symbol] = df

    if not histories:
        raise RuntimeError("No usable NSE history was returned.")

    basket_returns = []
    for df in histories.values():
        closes = df["Close"].astype(float).reset_index(drop=True)
        basket_returns.append(pct_change(float(closes.iloc[-1]), float(closes.iloc[-21])))
    basket_return_20 = sum(basket_returns) / len(basket_returns)

    first_history = next(iter(histories.values()))
    latest_date = pd.to_datetime(first_history["Date"]).max().strftime("%Y-%m-%d")

    rows: list[dict] = []
    for symbol, df in histories.items():
        row = classify_symbol(symbol, df, basket_return_20, latest_date)
        if row:
            rows.append(row)

    rows.sort(key=lambda item: (item["score"], item["rsVsBenchmark20d"]), reverse=True)
    html = build_html(rows, datetime.now(), latest_date)
    root = Path(__file__).resolve().parent
    (root / "india_swing_report.html").write_text(html, encoding="utf-8")
    (root / "index.html").write_text(html, encoding="utf-8")
    print(f"Saved report with {len(rows)} rows")


if __name__ == "__main__":
    main()
