import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import urllib.parse
import xml.etree.ElementTree as ET
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
from PIL import Image
import json

# WAJIB: panggil set_page_config di awal
st.set_page_config(page_title="Hyper-Hybrid Engine V12", layout="wide", page_icon="📊")

# ===============================================================================
# AUTO-SAVE HELPERS
# ===============================================================================
def save_key_to_file(key):
    with open("api_key.txt", "w") as f:
        f.write(key)

def load_key_from_file():
    if os.path.exists("api_key.txt"):
        with open("api_key.txt", "r") as f:
            return f.read().strip()
    return ""

def auto_save_broker_data(df: pd.DataFrame):
    df.to_csv("broker_data.csv", index=False)

# ===============================================================================
# DYNAMIC OCR INITIALIZATION (Auto-detect & Safe Load)
# ===============================================================================
HAS_OCR = False
try:
    import easyocr
    @st.cache_resource
    def load_ocr_reader():
        return easyocr.Reader(['en'], gpu=False, verbose=False)
    reader = load_ocr_reader()
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ===============================================================================
# CONFIG & KONSTANTA GLOBAL (V12 Anti-Overfitting Engine)
# ===============================================================================
WEIGHT_MIN    = 0.08
WEIGHT_MAX    = 0.40
SOFTMAX_TEMP  = 2.5
AI_SIGNAL_CAP = 0.30
MC_PESSIMISM  = 0.82

@dataclass
class BrokerEntry:
    broker_code: str
    buy_lot: int
    buy_freq: int
    sell_lot: int
    sell_freq: int
    avg_buy_price: float = 0.0
    avg_sell_price: float = 0.0

    @property
    def net_lot(self) -> int:   return self.buy_lot - self.sell_lot
    @property
    def total_vol(self) -> int: return self.buy_lot + self.sell_lot
    @property
    def avg_buy_lot(self) -> float:
        return self.buy_lot / self.buy_freq if self.buy_freq > 0 else 0.0
    @property
    def avg_sell_lot(self) -> float:
        return self.sell_lot / self.sell_freq if self.sell_freq > 0 else 0.0

def tick(p: int) -> int:
    if p < 200:  return 1
    if p < 500:  return 2
    if p < 2000: return 5
    if p < 5000: return 10
    return 25

def softmax_weights(scores: List[float], temp: float = 2.5) -> np.ndarray:
    arr     = np.array(scores) * temp
    exp_arr = np.exp(arr - np.max(arr))
    raw_w   = exp_arr / np.sum(exp_arr)
    return WEIGHT_MIN + (raw_w * (WEIGHT_MAX - WEIGHT_MIN))

# ===============================================================================
# BROKER SUMMARY OCR ENGINE  —  V2 (Ported from Android ML Kit BrokerOCREngine)
#
#  Pipeline (mirrors Kotlin exactly):
#   1. crop        → buang status bar (~10% atas) & nav bar (~12% bawah)
#   2. OCR         → EasyOCR → raw results
#   3. extract     → normalisasi + pisah multi-kata proporsional
#   4. findHeader  → cocokkan teks ke nama kolom; dominant-Y clustering
#   5. buildCols   → batas kolom via midpoint antar header
#   6. groupRows   → kluster elemen per baris (toleransi dinamis = 55% avg-H)
#   7. parseRows   → pilih sel terdekat (in-range dulu, fallback nearest)
#   8. parseKMB / parsePrice → format Stockbit ID (koma = desimal jika ada suffix)
#   9. merge       → gabung multi-screenshot (non-zero priority)
# ===============================================================================

# ── 1. Normalise column name ────────────────────────────────────────────────────
def normalize_col_name(raw: str) -> Optional[str]:
    t = (raw.upper().strip()
           .replace("·", ".").replace(",", ".").replace(";", ".")
           .replace(" ", "").rstrip("."))
    return {
        "BY": "BY",   "B.Y": "BY",
        "SL": "SL",   "S.L": "SL",  "5L": "SL",
        "B.VAL": "B.VAL", "BVAL": "B.VAL",
        "S.VAL": "S.VAL", "SVAL": "S.VAL",
        "B.LOT": "B.LOT", "BLOT": "B.LOT",
        "S.LOT": "S.LOT", "SLOT": "S.LOT",
        "B.FREQ": "B.FREQ", "BFREQ": "B.FREQ",
        "S.FREQ": "S.FREQ", "SFREQ": "S.FREQ",
        "B.AVG": "B.AVG", "BAVG": "B.AVG",
        "S.AVG": "S.AVG", "SAVG": "S.AVG",
    }.get(t)

# ── 2. Number parsers (Stockbit Indonesian locale) ──────────────────────────────
def parse_kmb(raw: str) -> float:
    """
    Format Stockbit ID:
      • Ada suffix K/M/B/T → koma = desimal  ("32,5K" → 32 500)
      • Tanpa suffix        → koma = ribuan  ("1,478" → 1 478)
    """
    s = str(raw).strip().upper().replace(" ", "")
    if not s or s in ["-", ".", ","]: return 0.0
    try:
        last = s[-1]
        if last in ('K', 'M', 'B', 'T'):
            v = float(s[:-1].replace(",", "."))   # koma → desimal
            return v * {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}[last]
        else:
            cleaned = s.replace(",", "")           # buang koma (ribuan)
            return float(cleaned) if cleaned.replace(".", "").isdigit() \
                   else float(s.replace(",", ".")) # last resort
    except:
        return 0.0

def parse_price(raw: str) -> float:
    """
    Harga IDX selalu integer 3-5 digit (IDR).
    "1,478" → 1478  (koma = separator ribuan, BUKAN desimal).
    """
    s = str(raw).strip().upper().replace(" ", "")
    if not s or s == "-": return 0.0
    if s[-1] in ('K', 'M', 'B', 'T'): return parse_kmb(s)
    try:
        return float(s.replace(",", "").replace(".", ""))
    except:
        return 0.0

# ── 3. Extract elements (split multi-word proportionally) ───────────────────────
def _extract_elements(ocr_results, img_width: int) -> List[Dict]:
    """
    EasyOCR returns (bbox, text, conf).
    bbox = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].
    Multi-word tokens are split and distributed proportionally across the bbox
    so that each number becomes its own TextEl (mirrors Kotlin extractElements).
    """
    elements = []
    for bbox, text, _conf in ocr_results:
        text = str(text).strip()
        if not text: continue
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        cy = (y1 + y2) // 2
        h  = max(1, y2 - y1)

        parts = text.split()
        if len(parts) > 1:
            part_w = (x2 - x1) / len(parts)
            for i, p in enumerate(parts):
                px1 = int(x1 + i * part_w)
                px2 = int(px1 + part_w)
                elements.append({"text": p,
                                  "cx": (px1 + px2) // 2, "cy": cy,
                                  "x1": px1, "x2": px2,
                                  "y1": y1,  "y2": y2, "h": h})
        else:
            elements.append({"text": text,
                              "cx": (x1 + x2) // 2, "cy": cy,
                              "x1": x1, "x2": x2,
                              "y1": y1, "y2": y2, "h": h})
    return elements

# ── 4. Find header row (dominant-Y clustering) ──────────────────────────────────
def _find_dominant_y(candidates: List[Dict], tol: int = 25) -> Optional[int]:
    """Temukan Y paling banyak kandidat dalam toleransi (sliding window)."""
    if not candidates: return None
    srt = sorted(candidates, key=lambda e: e["cy"])
    best_y, best_n = srt[0]["cy"], 1
    cur_y,  cur_n  = srt[0]["cy"], 1
    for i in range(1, len(srt)):
        if abs(srt[i]["cy"] - cur_y) <= tol:
            cur_n += 1
            if cur_n > best_n:
                best_n, best_y = cur_n, cur_y
        else:
            cur_y, cur_n = srt[i]["cy"], 1
    return best_y if best_n >= 3 else None

def _find_header_elements(elements: List[Dict]) -> Optional[List[Dict]]:
    candidates = [el for el in elements if normalize_col_name(el["text"]) is not None]
    if len(candidates) < 4: return None
    best_y = _find_dominant_y(candidates, tol=25)
    if best_y is None: return None
    return [el for el in candidates if abs(el["cy"] - best_y) <= 25]

# ── 5. Build column map (midpoint boundaries) ───────────────────────────────────
def _build_column_map(header_els: List[Dict], img_width: int) -> List[Dict]:
    """
    Batas kiri/kanan kolom = midpoint antara header yang bersebelahan.
    Lebih akurat dari lebar header itu sendiri.
    """
    srt = sorted(header_els, key=lambda e: e["cx"])
    n   = len(srt)
    cols = []
    for i, el in enumerate(srt):
        name = normalize_col_name(el["text"])
        if not name: continue
        x1 = 0           if i == 0     else (srt[i-1]["cx"] + el["cx"]) // 2
        x2 = img_width-1 if i == n - 1 else (el["cx"] + srt[i+1]["cx"]) // 2
        cols.append({"name": name, "cx": el["cx"], "x1": x1, "x2": x2})
    return cols

# ── 6. Group elements into rows (dynamic Y tolerance) ───────────────────────────
def _group_into_rows(elements: List[Dict]) -> List[List[Dict]]:
    """
    Toleransi dinamis = 55% rata-rata tinggi elemen,
    diklem ke [8, 28] px (sama dengan Kotlin coerceIn).
    """
    if not elements: return []
    avg_h   = max(10, int(np.mean([el["h"] for el in elements])))
    row_tol = max(8, min(28, int(avg_h * 0.55)))
    srt     = sorted(elements, key=lambda e: e["cy"])

    rows    = []
    cur_row = [srt[0]]
    cur_y   = srt[0]["cy"]
    for el in srt[1:]:
        if abs(el["cy"] - cur_y) <= row_tol:
            cur_row.append(el)
        else:
            if len(cur_row) >= 2: rows.append(cur_row)
            cur_row, cur_y = [el], el["cy"]
    if len(cur_row) >= 2: rows.append(cur_row)
    return rows

# ── 7. Parse rows → broker dict ─────────────────────────────────────────────────
_CODE_RE = re.compile(r"^[A-Z]{2,3}$")

def _cell(col_info: Optional[Dict], row: List[Dict]) -> str:
    """
    Pilih teks sel untuk kolom ini dari elemen baris:
      Priority 1 → elemen yang X-nya dalam rentang [x1, x2] kolom
      Priority 2 → elemen terdekat secara mutlak (fallback)
    """
    if not col_info: return "-"
    in_range = [el for el in row if col_info["x1"] <= el["cx"] <= col_info["x2"]]
    if in_range:
        return min(in_range, key=lambda e: abs(e["cx"] - col_info["cx"]))["text"]
    return (min(row, key=lambda e: abs(e["cx"] - col_info["cx"]))["text"]
            if row else "-")

def _parse_rows(rows: List[List[Dict]], cols: List[Dict]) -> Dict[str, Dict]:
    def gc(name): return next((c for c in cols if c["name"] == name), None)

    by_col    = gc("BY");    sl_col    = gc("SL")
    blot_col  = gc("B.LOT"); bfreq_col = gc("B.FREQ"); bavg_col = gc("B.AVG")
    slot_col  = gc("S.LOT"); sfreq_col = gc("S.FREQ"); savg_col = gc("S.AVG")

    buy_map: Dict[str, tuple] = {}
    sell_map: Dict[str, tuple] = {}

    for row in rows:
        by_text = "".join(filter(str.isalpha, _cell(by_col, row).upper()))
        sl_text = "".join(filter(str.isalpha, _cell(sl_col, row).upper()))

        if _CODE_RE.match(by_text):
            lot  = max(0, int(parse_kmb(_cell(blot_col,  row))))
            freq = max(0, int(parse_kmb(_cell(bfreq_col, row))))
            avg  = parse_price(_cell(bavg_col, row))
            if lot > 0: buy_map[by_text] = (lot, freq, avg)

        if _CODE_RE.match(sl_text):
            lot  = max(0, int(parse_kmb(_cell(slot_col,  row))))
            freq = max(0, int(parse_kmb(_cell(sfreq_col, row))))
            avg  = parse_price(_cell(savg_col, row))
            if lot > 0: sell_map[sl_text] = (lot, freq, avg)

    result = {}
    for code in set(list(buy_map) + list(sell_map)):
        b = buy_map.get(code)
        s = sell_map.get(code)
        result[code] = {
            "Broker":      code,
            "Buy Lot":     b[0] if b else 0,
            "Buy Freq":    b[1] if b else 0,
            "Sell Lot":    s[0] if s else 0,
            "Sell Freq":   s[1] if s else 0,
            "Avg Buy Px":  b[2] if b else 0.0,
            "Avg Sell Px": s[2] if s else 0.0,
        }
    return result

# ── 8. Merge two entries (non-zero priority, mirrors Kotlin merge) ───────────────
def _merge_entries(a: Dict, b: Dict) -> Dict:
    def pick(key): return a[key] if a[key] > 0 else b[key]
    return {
        "Broker":      a["Broker"],
        "Buy Lot":     pick("Buy Lot"),
        "Buy Freq":    pick("Buy Freq"),
        "Sell Lot":    pick("Sell Lot"),
        "Sell Freq":   pick("Sell Freq"),
        "Avg Buy Px":  pick("Avg Buy Px"),
        "Avg Sell Px": pick("Avg Sell Px"),
    }

# ── 9. Public entry point ────────────────────────────────────────────────────────
def process_broker_ocr(pil_images: List[Image.Image]) -> List[Dict[str, Any]]:
    """
    Proses satu atau lebih screenshot Stockbit → List[BrokerDict].
    Hasil multi-gambar di-merge otomatis (tangkapan scroll kiri & kanan).
    """
    if not HAS_OCR or not pil_images: return []
    master_map: Dict[str, Dict] = {}

    for img in pil_images:
        w, h = img.size
        crop_top    = int(h * 0.10)
        crop_bottom = int(h * 0.12)
        usable_h    = h - crop_top - crop_bottom
        if usable_h <= 0: continue

        cropped     = img.crop((0, crop_top, w, h - crop_bottom))
        raw_results = reader.readtext(np.array(cropped))
        elements    = _extract_elements(raw_results, w)
        if len(elements) < 8: continue

        header_els = _find_header_elements(elements)
        if not header_els: continue

        header_y = int(np.mean([el["cy"] for el in header_els]))
        cols     = _build_column_map(header_els, w)
        if len(cols) < 4: continue

        data_els     = [el for el in elements if el["cy"] > header_y + 15]
        rows         = _group_into_rows(data_els)
        page_results = _parse_rows(rows, cols)

        for code, entry in page_results.items():
            master_map[code] = (_merge_entries(master_map[code], entry)
                                if code in master_map else entry)

    result = [v for v in master_map.values()
              if v["Buy Lot"] > 0 or v["Sell Lot"] > 0]
    return sorted(result, key=lambda x: x["Buy Lot"] + x["Sell Lot"], reverse=True)

# ===============================================================================
# CORE SCRAPER & HISTORICAL DATA MODULE
# ===============================================================================
def fetch_yahoo_raw_data(symbol: str) -> dict:
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=3mo"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).json()
        result = res["chart"]["result"][0]
        df = pd.DataFrame({
            "timestamp": [datetime.datetime.fromtimestamp(ts) for ts in result["timestamp"]],
            "open":      result["indicators"]["quote"][0]["open"],
            "high":      result["indicators"]["quote"][0]["high"],
            "low":       result["indicators"]["quote"][0]["low"],
            "close":     result["indicators"]["quote"][0]["close"],
            "adj_close": result["indicators"]["adjclose"][0]["adjclose"],
            "volume":    result["indicators"]["quote"][0]["volume"]
        }).dropna()
        return {"status": "SUCCESS", "data": df,
                "latest_price": float(df["adj_close"].iloc[-1])}
    except Exception as e:
        return {"status": "FALLBACK", "data": pd.DataFrame(),
                "latest_price": 5000.0, "error": str(e)}

def fetch_rss_news(ticker: str) -> List[str]:
    queries = [f"{ticker} site:cnbcindonesia.com", ticker]
    titles  = set()
    for q in queries:
        try:
            res = requests.get(
                f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            for item in ET.fromstring(res.text).findall(".//item")[:4]:
                titles.add(item.find("title").text)
        except:
            continue
    return list(titles)

# ===============================================================================
# QUANTITATIVE ALGORITHMS (V12)
# ===============================================================================
def analyze_broker_summary_ai(entries: List[BrokerEntry], current_price: float) -> dict:
    if not entries:
        return {"score": 0.0, "signal": "NEUTRAL", "net_lot": 0,
                "whale_present": False, "desc": "No Data"}
    total_vol  = sum(e.total_vol for e in entries)
    net_lot    = sum(e.net_lot   for e in entries)
    flow_ratio = net_lot / max(1, total_vol)

    valid_lots = ([e.avg_buy_lot  for e in entries if e.buy_freq  > 0] +
                  [e.avg_sell_lot for e in entries if e.sell_freq > 0])
    median_lot = np.median(valid_lots) if valid_lots else 1.0
    whale = any(e.avg_buy_lot  > median_lot * 4 or
                e.avg_sell_lot > median_lot * 4 for e in entries)

    w_buy_px  = (sum(e.avg_buy_price  * e.buy_lot  for e in entries) /
                 max(1, sum(e.buy_lot  for e in entries)))
    w_sell_px = (sum(e.avg_sell_price * e.sell_lot for e in entries) /
                 max(1, sum(e.sell_lot for e in entries)))
    pressure  = (((current_price - w_buy_px)  / max(1.0, w_buy_px)) -
                 ((w_sell_px - current_price)  / max(1.0, w_sell_px)))
    score     = max(-1.0, min(1.0, (flow_ratio * 0.6) + (pressure * 0.4)))

    if   score >  0.2: signal = "STRONG ACCUM"
    elif score >  0.05: signal = "ACCUM"
    elif score < -0.2:  signal = "STRONG DIST"
    elif score < -0.05: signal = "DISTRIBUTION"
    else:               signal = "NEUTRAL"

    return {"score": score, "signal": signal, "net_lot": net_lot,
            "whale_present": whale,
            "desc": f"Flow Ratio: {flow_ratio:.2%}, Price Pressure: {pressure:.2f}"}

def analyze_with_gemini(ticker: str, headlines: List[str],
                        api_key: str, model_name: str) -> dict:
    if not api_key:
        return {"stock_score": 0.0, "market_score": 0.0,
                "label": "Neutral", "reason": "No Key"}
    url    = (f"https://generativelanguage.googleapis.com/v1beta/models/"
              f"{model_name}:generateContent")
    prompt = (f'Analyze sentiment for {ticker}. Return ONLY valid JSON:\n'
              f'{{"stock_score":0.15,"market_score":0.02,'
              f'"label":"Bullish","reason":"text"}}')
    try:
        res = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt + "\nContext: " +
                                            ". ".join(headlines)}]}]},
            headers={"x-goog-api-key": api_key}, timeout=20)
        raw = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except:
        return {"stock_score": 0.0, "market_score": 0.0,
                "label": "Neutral", "reason": "API Error"}

def compute_quantitative_matrix(df: pd.DataFrame, current_price: float) -> dict:
    closes      = df["adj_close"].to_numpy()
    log_returns = np.diff(np.log(closes))
    volatility  = np.std(log_returns) * np.sqrt(252) if len(log_returns) > 1 else 0.20

    gains    = np.where(log_returns > 0, log_returns, 0)
    losses   = np.where(log_returns < 0, -log_returns, 0)
    avg_gain = np.mean(gains[-14:])  if len(gains)  >= 14 else 0.01
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.01
    rsi      = 100 - (100 / (1 + (avg_gain / max(0.00001, avg_loss))))

    last_h = df["high"].iloc[-1]
    last_l = df["low"].iloc[-1]
    last_c = closes[-1]
    pivot  = (last_h + last_l + last_c) / 3.0
    return {
        "volatility":     volatility,
        "rsi":            rsi,
        "momentum_score": (1.0 if rsi < 30 else -1.0 if rsi > 70
                           else (50 - rsi) / 20),
        "pivot": pivot,
        "r1": (2.0 * pivot) - last_l, "s1": (2.0 * pivot) - last_h,
        "r2": pivot + (last_h - last_l), "s2": pivot - (last_h - last_l),
    }

def run_monte_carlo_v12(current_price: float, volatility: float,
                        bias: float, tp: float, sl: float) -> dict:
    np.random.seed(42)
    simulations, days, dt = 1000, 20, 1 / 252
    drift       = (bias * AI_SIGNAL_CAP) * (1.0 - MC_PESSIMISM)
    price_paths = np.zeros((days, simulations))
    price_paths[0] = current_price
    for t in range(1, days):
        price_paths[t] = (price_paths[t-1] *
                          np.exp((drift - 0.5 * volatility**2) * dt +
                                 volatility * np.sqrt(dt) *
                                 np.random.standard_normal(simulations)))
    return {
        "mean_target": float(np.mean(price_paths[-1])),
        "p_tp":       np.sum(price_paths[-1] >= tp) / simulations,
        "p_sl":       np.sum(price_paths[-1] <= sl) / simulations,
        "p_bullish":  np.sum(price_paths[-1] > current_price) / simulations,
    }

def compute_advanced_risk_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"var_95": 0.05, "max_drawdown": 0.1, "sharpe": 1.0, "allocation": 0.05}
    log_returns  = np.diff(np.log(df["adj_close"].to_numpy()))
    cum_returns  = np.cumprod(1 + log_returns)
    running_max  = np.maximum.accumulate(cum_returns)
    running_max  = np.where(running_max == 0, 1.0, running_max)
    max_dd       = (np.min((cum_returns - running_max) / running_max)
                    if len(cum_returns) > 0 else -0.10)
    sharpe       = ((np.mean(log_returns - (0.06 / 252)) /
                     np.std(log_returns)) * np.sqrt(252)
                    if np.std(log_returns) > 0 else 0.0)
    return {
        "var_95":      abs(np.percentile(log_returns, 5)),
        "max_drawdown": abs(max_dd),
        "sharpe":      sharpe,
        "allocation":  0.05 if sharpe < 1.0 else 0.10 if sharpe < 2.0 else 0.15,
    }

# ===============================================================================
# CUSTOM CSS (tema-aware — pakai CSS variables Streamlit)
# ===============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-color);
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-color);
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.2rem;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .card {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(128,128,128,0.2);
        color: var(--text-color);
    }
    .card p, .card b { color: var(--text-color); }
    .metric-big   { font-size: 2rem; font-weight: bold; color: var(--text-color); }
    .metric-label { font-size: 0.9rem; color: var(--text-color); opacity: 0.6; }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        box-shadow: 0 2px 8px rgba(52,152,219,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ===============================================================================
# MAIN LAYOUT APPLICATION BUILDER
# ===============================================================================
st.markdown('<div class="main-header">📊 HYPER-HYBRID QUANTITATIVE ENGINE V12</div>',
            unsafe_allow_html=True)

# --- SESSION STATE INIT ---
if "v12_memory_runs"    not in st.session_state: st.session_state["v12_memory_runs"]    = 0
if "v12_cumulative_bias" not in st.session_state: st.session_state["v12_cumulative_bias"] = 0.0
if "prediction_history" not in st.session_state: st.session_state["prediction_history"] = []
if "gemini_api_key"     not in st.session_state:
    st.session_state["gemini_api_key"] = load_key_from_file()
if "table_data" not in st.session_state:
    if os.path.exists("broker_data.csv"):
        try:
            st.session_state["table_data"] = pd.read_csv("broker_data.csv")
        except:
            st.session_state["table_data"] = pd.DataFrame([{
                "Broker": "YP", "Buy Lot": 8500, "Buy Freq": 420,
                "Sell Lot": 1200, "Sell Freq": 95,
                "Avg Buy Px": 4500.0, "Avg Sell Px": 4480.0}])
    else:
        st.session_state["table_data"] = pd.DataFrame([{
            "Broker": "YP", "Buy Lot": 8500, "Buy Freq": 420,
            "Sell Lot": 1200, "Sell Freq": 95,
            "Avg Buy Px": 4500.0, "Avg Sell Px": 4480.0}])

# ===============================================================================
# SIDEBAR
# ===============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration Center")

    with st.expander("🔑 Gemini AI API Key", expanded=True):
        key_input = st.text_input("Masukkan API Key", type="password",
                                  value=st.session_state["gemini_api_key"])
        if key_input != st.session_state["gemini_api_key"]:
            st.session_state["gemini_api_key"] = key_input
            save_key_to_file(key_input)

    with st.expander("🧠 Model & Engine", expanded=False):
        selected_model = st.selectbox("Pilih Model Gemini",
                                      ["gemini-1.5-flash", "gemini-2.5-flash"])

    with st.expander("📸 Broker Summary OCR", expanded=HAS_OCR):
        if HAS_OCR:
            st.success("✅ OCR Engine V2 (ML Kit port) Aktif")
            st.caption("Multi-screenshot merge • KMB locale-aware • Dynamic row clustering")
        else:
            st.warning("⚠️ Install easyocr untuk fitur OCR")

        uploaded_files = st.file_uploader(
            "Upload Screenshot Broker (bisa beberapa untuk scroll kiri/kanan)",
            type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        if uploaded_files and HAS_OCR:
            if st.button("Ekstrak Data Dari Gambar", use_container_width=True):
                with st.spinner(f"Memproses {len(uploaded_files)} screenshot..."):
                    ocr_results = process_broker_ocr(
                        [Image.open(f) for f in uploaded_files])
                    if ocr_results:
                        new_df = pd.DataFrame(ocr_results)
                        st.session_state["table_data"] = new_df
                        auto_save_broker_data(new_df)
                        st.success(f"✅ Berhasil mengekstrak {len(ocr_results)} broker!")
                    else:
                        st.error("❌ Gagal mendeteksi tabel — pastikan screenshot "
                                 "memuat kolom BY/SL dengan jelas.")

    with st.expander("🧠 V12 Adaptive Memory", expanded=False):
        st.progress(min(1.0, st.session_state["v12_memory_runs"] / 10.0))
        st.caption(f"Long-term Bias Anchoring: "
                   f"{st.session_state['v12_cumulative_bias']:.4f}")

    st.markdown("---")
    st.caption("Versi 12.5 – Anti-Overfitting Engine + ML Kit OCR Port")

# ===============================================================================
# MAIN CONTENT AREA
# ===============================================================================
tab1, tab2 = st.tabs(["🎯 Live Trading Engine", "📈 Historis & Risiko"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="section-title">▼ INPUT ASET</div>',
                    unsafe_allow_html=True)
        ticker = st.text_input("Kode Saham (IDX)", value="BBRI").upper().strip()
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_btn = st.button("▶ RUN ENGINE", use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">▼ REAL-TIME BROKER DATA MATRIX</div>',
                    unsafe_allow_html=True)
        try:
            edited_df = st.data_editor(st.session_state["table_data"],
                                       num_rows="dynamic", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memuat editor: {e}")
            edited_df = st.session_state["table_data"]
        col_save, _ = st.columns([1, 3])
        with col_save:
            if st.button("💾 Simpan Tabel", use_container_width=True):
                auto_save_broker_data(edited_df)
                st.session_state["table_data"] = edited_df
                st.success("Data broker tersimpan!")

    if run_btn:
        with st.spinner("Mengambil data harga & berita..."):
            yahoo_data    = fetch_yahoo_raw_data(ticker)
            current_price = yahoo_data["latest_price"]
            df            = yahoo_data["data"]
            headlines     = fetch_rss_news(ticker)

            broker_entries = []
            for _, row in st.session_state["table_data"].iterrows():
                try:
                    broker_entries.append(BrokerEntry(
                        broker_code    = str(row["Broker"]),
                        buy_lot        = int(row["Buy Lot"]),
                        buy_freq       = int(row["Buy Freq"]),
                        sell_lot       = int(row["Sell Lot"]),
                        sell_freq      = int(row["Sell Freq"]),
                        avg_buy_price  = float(row.get("Avg Buy Px",  0)),
                        avg_sell_price = float(row.get("Avg Sell Px", 0))
                    ))
                except:
                    pass

            broker_analysis = analyze_broker_summary_ai(broker_entries, current_price)
            gemini_res      = analyze_with_gemini(ticker, headlines,
                                                  st.session_state["gemini_api_key"],
                                                  selected_model)
            quant = (compute_quantitative_matrix(df, current_price)
                     if not df.empty else {
                         "volatility": 0.2, "rsi": 50.0, "momentum_score": 0.0,
                         "pivot": current_price,
                         "r1": current_price*1.01, "s1": current_price*0.99,
                         "r2": current_price*1.02, "s2": current_price*0.98,
                     })

        st.markdown('<div class="section-title">📊 SINTESIS SINYAL & METRIK</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        SIGNAL_ICON = {"STRONG ACCUM": "🟢", "ACCUM": "🟢",
                       "DISTRIBUTION": "🔴", "STRONG DIST": "🔴", "NEUTRAL": "⚪"}
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Harga Terakhir", f"Rp {current_price:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            icon = SIGNAL_ICON.get(broker_analysis["signal"], "⚪")
            st.metric("Sinyal Broker",
                      f"{icon} {broker_analysis['signal']}")
            st.caption(broker_analysis["desc"])
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Sentimen Gemini",
                      f"{gemini_res.get('label','N/A')}",
                      delta=f"{gemini_res.get('stock_score',0):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("RSI (14)", f"{quant['rsi']:.1f}",
                      delta=f"{quant['momentum_score']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        colA, colB = st.columns([2, 1])
        with colA:
            st.markdown('<div class="section-title">🔮 PROYEKSI MONTE CARLO V12</div>',
                        unsafe_allow_html=True)
            tp     = current_price * 1.05
            sl     = current_price * 0.95
            mc_res = run_monte_carlo_v12(
                current_price, quant["volatility"],
                broker_analysis["score"] + gemini_res["stock_score"], tp, sl)
            st.metric("Ekspektasi Harga 20 Hari",
                      f"Rp {mc_res['mean_target']:,.0f}",
                      delta=f"{mc_res['mean_target']/current_price-1:.2%}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Prob Bullish",  f"{mc_res['p_bullish']:.0%}")
            m2.metric("Prob TP (+5%)", f"{mc_res['p_tp']:.0%}")
            m3.metric("Prob SL (-5%)", f"{mc_res['p_sl']:.0%}")
        with colB:
            st.markdown('<div class="section-title">⚡ LEVEL PIVOT</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card">
                <p><b>Pivot:</b> {quant['pivot']:,.0f}</p>
                <p><b>R1:</b> {quant['r1']:,.0f}  |  <b>S1:</b> {quant['s1']:,.0f}</p>
                <p><b>R2:</b> {quant['r2']:,.0f}  |  <b>S2:</b> {quant['s2']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.session_state["v12_memory_runs"] += 1
        st.session_state["v12_cumulative_bias"] += (
            (broker_analysis["score"] + gemini_res["stock_score"]) / 2)

        st.markdown("---")
        st.caption(f"✅ Engine berhasil dijalankan. "
                   f"Berita terkini: {', '.join(headlines[:3])}")

with tab2:
    st.markdown('<div class="section-title">📉 METRIK RISIKO & HISTORIS</div>',
                unsafe_allow_html=True)
    if "df" not in locals() and not os.path.exists("broker_data.csv"):
        st.info("Silakan jalankan 'RUN ENGINE' terlebih dahulu.")
    else:
        df_hist = df if "df" in locals() and not df.empty else pd.DataFrame()
        risk    = compute_advanced_risk_metrics(df_hist)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Value at Risk (95%)",      f"{risk['var_95']:.2%}")
        c2.metric("Max Drawdown",              f"{risk['max_drawdown']:.2%}")
        c3.metric("Sharpe Ratio",              f"{risk['sharpe']:.2f}")
        c4.metric("Alokasi Modal Disarankan",  f"{risk['allocation']:.1%}")
        if not df_hist.empty:
            st.line_chart(df_hist.set_index("timestamp")["adj_close"],
                          use_container_width=True)
