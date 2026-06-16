import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from PIL import Image

# Coba import EasyOCR untuk mengaktifkan pemrosesan screenshot otomatis
HAS_OCR = False
try:
    import easyocr
    # Inisialisasi recognizer untuk bahasa inggris/latin (Stockbit menggunakan teks latin)
    @st.cache_resource
    def load_ocr_reader():
        return easyocr.Reader(['en'], gpu=False)
    reader = load_ocr_reader()
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ===============================================================================
# CONFIG & KONSTANTA GLOBAL (V12 Anti-Overfitting Engine - Korelasi Kotlin)
# ===============================================================================
WEIGHT_MIN = 0.08
WEIGHT_MAX = 0.40
SOFTMAX_TEMP = 2.5
AI_SIGNAL_CAP = 0.30
MC_PESSIMISM = 0.82

# ===============================================================================
# DATA STRUCTURES & HELPERS
# ===============================================================================
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
    def net_lot(self) -> int: return self.buy_lot - self.sell_lot
    @property
    def total_vol(self) -> int: return self.buy_lot + self.sell_lot
    @property
    def avg_buy_lot(self) -> float: return self.buy_lot / self.buy_freq if self.buy_freq > 0 else 0.0
    @property
    def avg_sell_lot(self) -> float: return self.sell_lot / self.sell_freq if self.sell_freq > 0 else 0.0

def tick(p: int) -> int:
    if p < 200: return 1
    if p < 500: return 2
    if p < 2000: return 5
    if p < 5000: return 10
    return 25

def softmax_weights(scores: List[float], temp: float = 2.5) -> np.ndarray:
    arr = np.array(scores) * temp
    exp_arr = np.exp(arr - np.max(arr))
    raw_w = exp_arr / np.sum(exp_arr)
    return WEIGHT_MIN + (raw_w * (WEIGHT_MAX - WEIGHT_MIN))

# ===============================================================================
# PYTHON REPLICATION OF BROKER OCR ENGINE (Konversi Eksak dari Kode Kotlin Anda)
# ===============================================================================
def normalize_col_name(raw: str) -> str:
    t = raw.upper().strip().replace("·", ".").replace(",", ".").replace(";", ".").replace(" ", "")
    if t.endswith("."): t = t[:-1]
    
    if t in ["BY", "B.Y", "BY."]: return "BY"
    if t in ["SL", "S.L", "SL.", "5L"]: return "SL"
    if t in ["B.VAL", "BVAL"]: return "B.VAL"
    if t in ["S.VAL", "SVAL"]: return "S.VAL"
    if t in ["B.LOT", "BLOT"]: return "B.LOT"
    if t in ["S.LOT", "SLOT"]: return "S.LOT"
    if t in ["B.FREQ", "BFREQ"]: return "B.FREQ"
    if t in ["S.FREQ", "SFREQ"]: return "S.FREQ"
    if t in ["B.AVG", "BAVG"]: return "B.AVG"
    if t in ["S.AVG", "SAVG"]: return "S.AVG"
    return ""

def parse_kmb(raw: str) -> float:
    s = raw.trim().upper().replace(" ", "") if hasattr(raw, 'trim') else str(raw).strip().upper().replace(" ", "")
    if not s or s in ["-", ".", ","]: return 0.0
    try:
        last_char = s[-1]
        if last_char in ['K', 'M', 'B', 'T']:
            num_str = s[:-1].replace(",", ".")
            v = float(num_str)
            if last_char == 'K': return v * 1000.0
            if last_char == 'M': return v * 1000000.0
            if last_char == 'B': return v * 1000000000.0
            if last_char == 'T': return v * 1000000000000.0
        else:
            cleaned = s.replace(",", "")
            return float(cleaned) if cleaned.isdigit() else float(s.replace(",", "."))
    except:
        return 0.0

def parse_price(raw: str) -> float:
    s = str(raw).strip().upper().replace(" ", "")
    if not s or s == "-": return 0.0
    if s[-1] in ['K', 'M', 'B', 'T']: return parse_kmb(s)
    try:
        return float(s.replace(",", "").replace(".", ""))
    except:
        return 0.0

def process_broker_ocr(pil_images: List[Image.Image]) -> List[Dict[str, Any]]:
    """ Fungsi Utama Pemrosesan OCR Screenshot Emulasi Kode Kotlin """
    if not HAS_OCR or not pil_images:
        return []
        
    master_map = {}
    
    for img in pil_images:
        w, h = img.size
        # Step 1: Crop buang noise status bar (~10% atas) & nav bar (~12% bawah)
        crop_top = int(h * 0.10)
        crop_bottom = int(h * 0.12)
        cropped_img = img.crop((0, crop_top, w, h - crop_bottom))
        
        # Jalankan Engine OCR EasyOCR
        # Format output: [([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], text, confidence), ...]
        raw_results = reader.readtext(np.array(cropped_img))
        
        elements = []
        for bbox, text, conf in raw_results:
            text_str = str(text).strip()
            if not text_str: continue
            
            # Hitung bounding box center points
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            box_w = max(xs) - min(xs)
            box_h = max(ys) - min(ys)
            
            elements.append({
                "text": text_str, "cx": cx, "cy": cy, 
                "w": box_w, "h": box_h, "x1": min(xs), "x2": max(xs)
            })
            
        if len(elements) < 4: continue
        
        # Step 3: Cari Row Header
        candidates = [el for el in elements if normalize_col_name(el["text"]) != ""]
        if len(candidates) < 3: continue
        
        # Find Dominant Y Rata-rata koordinat header band
        c_ys = [el["cy"] for el in candidates]
        header_y = int(np.mean(c_ys))
        
        header_els = [el for el in candidates if abs(el["cy"] - header_y) <= 25]
        header_els.sort(key=lambda x: x["cx"])
        
        # Step 4: Build Column Map Bounds Boundary
        cols = []
        for i, el in enumerate(header_els):
            name = normalize_col_name(el["text"])
            x1_bound = 0 if i == 0 else (header_els[i-1]["cx"] + el["cx"]) // 2
            x2_bound = w if i == len(header_els) - 1 else (el["cx"] + header_els[i+1]["cx"]) // 2
            cols.append({"name": name, "cx": el["cx"], "x1": x1_bound, "x2": x2_bound})
            
        # Step 5: Ambil elemen data di bawah header dan Group into Rows
        data_els = [el for el in elements if el["cy"] > header_y + 15]
        if not data_els: continue
        
        avg_h = np.mean([el["h"] for el in data_els]) if data_els else 15
        row_tol = max(8, min(28, int(avg_h * 0.55)))
        
        data_els.sort(key=lambda x: x["cy"])
        rows = []
        current_row = [data_els[0]]
        current_y = data_els[0]["cy"]
        
        for el in data_els[1:]:
            if abs(el["cy"] - current_y) <= row_tol:
                current_row.append(el)
            else:
                if len(current_row) >= 2: rows.append(current_row)
                current_row = [el]
                current_y = el["cy"]
        if len(current_row) >= 2: rows.append(current_row)
        
        # Step 6: Parse baris data menjadi BrokerEntry structured data
        import re
        code_re = re.compile(r"^[A-Z]{2,3}$")
        
        col_by = next((c for c in cols if c["name"] == "BY"), None)
        col_sl = next((c for c in cols if c["name"] == "SL"), None)
        col_blot = next((c for c in cols if c["name"] == "B.LOT"), None)
        col_bfreq = next((c for c in cols if c["name"] == "B.FREQ"), None)
        col_bavg = next((c for c in cols if c["name"] == "B.AVG"), None)
        col_slot = next((c for c in cols if c["name"] == "S.LOT"), None)
        col_sfreq = next((c for c in cols if c["name"] == "S.FREQ"), None)
        col_savg = next((c for c in cols if c["name"] == "S.AVG"), None)
        
        for r in rows:
            def get_cell_text(col_info):
                if not col_info: return "-"
                in_range = [el for el in r if el["cx"] >= col_info["x1"] and el["cx"] <= col_info["x2"]]
                if in_range:
                    return min(in_range, key=lambda x: abs(x["cx"] - col_info["cx"]))["text"]
                return min(r, key=lambda x: abs(x["cx"] - col_info["cx"]))["text"]
                
            by_text = "".join(filter(str.isalpha, get_cell_text(col_by).upper()))
            sl_text = "".join(filter(str.isalpha, get_cell_text(col_sl).upper()))
            
            if code_re.match(by_text):
                lot = int(parse_kmb(get_cell_text(col_blot)))
                freq = int(parse_kmb(get_cell_text(col_bfreq)))
                avg = parse_price(get_cell_text(col_bavg))
                if lot > 0:
                    if by_text not in master_map: master_map[by_text] = {"broker": by_text, "b_lot": 0, "b_freq": 0, "b_avg": 0.0, "s_lot": 0, "s_freq": 0, "s_avg": 0.0}
                    master_map[by_text]["b_lot"] = lot
                    master_map[by_text]["b_freq"] = freq
                    master_map[by_text]["b_avg"] = avg
                    
            if code_re.match(sl_text):
                lot = int(parse_kmb(get_cell_text(col_slot)))
                freq = int(parse_kmb(get_cell_text(col_sfreq)))
                avg = parse_price(get_cell_text(col_savg))
                if lot > 0:
                    if sl_text not in master_map: master_map[sl_text] = {"broker": sl_text, "b_lot": 0, "b_freq": 0, "b_avg": 0.0, "s_lot": 0, "s_freq": 0, "s_avg": 0.0}
                    master_map[sl_text]["s_lot"] = lot
                    master_map[sl_text]["s_freq"] = freq
                    master_map[sl_text]["s_avg"] = avg

    # Format akhir untuk dikembalikan ke komponen data editor UI
    output_list = []
    for k, v in master_map.items():
        output_list.append({
            "Broker": v["broker"],
            "Buy Lot": v["b_lot"],
            "Buy Freq": v["b_freq"],
            "Sell Lot": v["s_lot"],
            "Sell Freq": v["s_freq"],
            "Avg Buy Px": v["b_avg"],
            "Avg Sell Px": v["s_avg"]
        })
    return output_list

# ===============================================================================
# CORE SCRAPER & HISTORICAL RAW DATA MODULE
# ===============================================================================
def fetch_yahoo_raw_data(symbol: str) -> dict:
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=3mo"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=15).json()
        result = res["chart"]["result"][0]
        timestamps = result["timestamp"]
        indicators = result["indicators"]["quote"][0]
        adj_close = result["indicators"]["adjclose"][0]["adjclose"]
        
        df = pd.DataFrame({
            "timestamp": [datetime.datetime.fromtimestamp(ts) for ts in timestamps],
            "open": indicators["open"], "high": indicators["high"], "low": indicators["low"],
            "close": indicators["close"], "adj_close": adj_close, "volume": indicators["volume"]
        }).dropna()
        return {"status": "SUCCESS", "data": df, "latest_price": float(df["adj_close"].iloc[-1])}
    except Exception as e:
        return {"status": "FALLBACK", "data": pd.DataFrame(), "latest_price": 5000.0, "error": str(e)}

def fetch_rss_news(ticker: str) -> List[str]:
    queries = [f"{ticker} site:cnbcindonesia.com", ticker, "IHSG Makro"]
    news_titles = set()
    for q in queries:
        try:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}"
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            root = ET.fromstring(res.text)
            for item in root.findall(".//item")[:6]: news_titles.add(item.find("title").text)
        except: continue
    return list(news_titles)

def analyze_broker_summary_ai(entries: List[BrokerEntry], current_price: float) -> dict:
    if not entries: return {"score": 0.0, "signal": "NEUTRAL", "net_lot": 0, "whale_present": False, "desc": "No Data"}
    total_vol = sum(e.total_vol for e in entries)
    net_lot = sum(e.net_lot for e in entries)
    flow_ratio = net_lot / max(1, total_vol)
    
    valid_lots = [e.avg_buy_lot for e in entries if e.buy_freq > 0] + [e.avg_sell_lot for e in entries if e.sell_freq > 0]
    median_lot = np.median(valid_lots) if valid_lots else 1.0
    whale_present = any(e.avg_buy_lot > median_lot * 4 or e.avg_sell_lot > median_lot * 4 for e in entries)
    
    w_buy_px = sum(e.avg_buy_price * e.buy_lot for e in entries if e.buy_lot > 0) / max(1, sum(e.buy_lot for e in entries))
    w_sell_px = sum(e.avg_sell_price * e.sell_lot for e in entries if e.sell_lot > 0) / max(1, sum(e.sell_lot for e in entries))
    
    pressure = 0.0
    if w_buy_px > 0 and w_sell_px > 0:
        pressure = ((current_price - w_buy_px) / w_buy_px) - ((w_sell_px - current_price) / w_sell_px)
        
    inst_score = max(-1.0, min(1.0, (flow_ratio * 0.6) + (pressure * 0.4)))
    signal = "STRONG ACCUMULATION" if inst_score > 0.2 else "ACCUMULATION" if inst_score > 0.05 else "DISTRIBUTION" if inst_score < -0.05 else "STRONG DISTRIBUTION" if inst_score < -0.2 else "NEUTRAL"
    return {"score": inst_score, "signal": signal, "net_lot": net_lot, "whale_present": whale_present, "desc": f"Flow: {flow_ratio:.2%}"}

def analyze_with_gemini(ticker: str, headlines: List[str], api_key: str, model_name: str) -> dict:
    if not api_key: return {"stock_score": 0.0, "market_score": 0.0, "reason": "No Key"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    text_news = ". ".join(headlines[:12]) if headlines else "Netral market setup."
    prompt = f"Analyze sentiment for {ticker}. Return ONLY valid JSON: {{\n\"stock_score\":0.15,\n\"market_score\":0.02,\n\"label\":\"Bullish\",\n\"reason\":\"text\"\n}}"
    body = {"contents": [{"parts": [{"text": prompt + "\nContext: " + text_news}]}]}
    try:
        res = requests.post(url, json=body, headers={"x-goog-api-key": api_key}, timeout=20)
        raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        import json
        return json.loads(raw_text[raw_text.find("{"):raw_text.rfind("}")+1])
    except:
        return {"stock_score": 0.0, "market_score": 0.0, "label": "Neutral", "reason": "API Error"}

def compute_quantitative_matrix(df: pd.DataFrame, current_price: float) -> dict:
    closes = df["adj_close"].to_numpy()
    log_returns = np.diff(np.log(closes))
    volatility = np.std(log_returns) * np.sqrt(252) if len(log_returns) > 1 else 0.20
    
    gains = np.where(log_returns > 0, log_returns, 0)
    losses = np.where(log_returns < 0, -log_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.01
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.01
    rsi = 100 - (100 / (1 + (avg_gain / max(0.00001, avg_loss))))
    momentum_score = 1.0 if rsi < 30 else -1.0 if rsi > 70 else (50 - rsi) / 20
    
    last_h, last_l, last_c = df["high"].iloc[-1], df["low"].iloc[-1], closes[-1]
    pivot = (last_h + last_l + last_c) / 3.0
    return {"volatility": volatility, "rsi": rsi, "momentum_score": momentum_score, "pivot": pivot, "r1": (2.0*pivot)-last_l, "s1": (2.0*pivot)-last_h, "r2": pivot+(last_h-last_l), "s2": pivot-(last_h-last_l)}

def run_monte_carlo_v12(current_price: float, volatility: float, bias: float, tp: float, sl: float) -> dict:
    np.random.seed(42)
    simulations, days, dt = 1000, 20, 1 / 252
    adjusted_drift = (bias * AI_SIGNAL_CAP) * (1.0 - MC_PESSIMISM)
    price_paths = np.zeros((days, simulations))
    price_paths[0] = current_price
    for t in range(1, days):
        rand = np.random.standard_normal(simulations)
        price_paths[t] = price_paths[t-1] * np.exp((adjusted_drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rand)
    return {"mean_target": float(np.mean(price_paths[-1])), "p_tp": np.sum(price_paths[-1] >= tp) / simulations, "p_sl": np.sum(price_paths[-1] <= sl) / simulations, "p_bullish": np.sum(price_paths[-1] > current_price) / simulations}

def compute_advanced_risk_metrics(df: pd.DataFrame, sl_price: float, current_price: float) -> dict:
    if df.empty: return {"var_95": 0.05, "max_drawdown": 0.1, "sharpe": 1.0, "allocation": 0.05}
    closes = df["adj_close"].to_numpy()
    log_returns = np.diff(np.log(closes))
    var_95 = np.percentile(log_returns, 5) if len(log_returns) > 5 else -0.03
    cum_returns = np.cumprod(1 + log_returns)
    running_max = np.maximum.accumulate(cum_returns)
    running_max = np.where(running_max == 0, 1.0, running_max)
    max_dd = np.min((cum_returns - running_max) / running_max) if len(cum_returns) > 0 else -0.10
    sharpe = (np.mean(log_returns - (0.06/252)) / np.std(log_returns)) * np.sqrt(252) if np.std(log_returns) > 0 else 0.0
    return {"var_95": abs(var_95), "max_drawdown": abs(max_dd), "sharpe": sharpe, "allocation": 0.05 if sharpe < 1.0 else 0.10 if sharpe < 2.0 else 0.15}

# ===============================================================================
# MAIN STREAMLIT UI LAYOUT
# ===============================================================================
st.set_page_config(page_title="Hyper-Hybrid Engine V12 + OCR", layout="wide")
st.title("📊 HYPER-HYBRID QUANTITATIVE ENGINE V12 + OCR")

if "v12_memory_runs" not in st.session_state: st.session_state["v12_memory_runs"] = 0
if "v12_cumulative_bias" not in st.session_state: st.session_state["v12_cumulative_bias"] = 0.0
if "v12_last_tickers" not in st.session_state: st.session_state["v12_last_tickers"] = []
if "prediction_history" not in st.session_state: st.session_state["prediction_history"] = []
if "table_data" not in st.session_state: 
    st.session_state["table_data"] = pd.DataFrame([{"Broker": "YP", "Buy Lot": 8500, "Buy Freq": 420, "Sell Lot": 1200, "Sell Freq": 95, "Avg Buy Px": 4500.0, "Avg Sell Px": 4480.0}])

with st.sidebar:
    st.header("⚙ CORE V12 CONFIG")
    saved_key = st.text_input("Gemini API Key", type="password", value=st.session_state.get("gemini_api_key", ""))
    if saved_key: st.session_state["gemini_api_key"] = saved_key
    selected_model = st.selectbox("Model Engine", ["gemini-1.5-flash", "gemini-3-flash-preview"])
    
    st.markdown("---")
    st.subheader("📸 Broker Summary OCR Engine")
    if HAS_OCR:
        st.success("✅ ML Kit Python (EasyOCR) Active")
    else:
        st.warning("⚠️ EasyOCR belum terinstall. Menjalankan mode manual. Untuk mengaktifkan OCR, ketik di terminal: `pip install easyocr`")
        
    uploaded_files = st.file_uploader("Upload Broker Summary Screenshots (Bisa Multi-file)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files and HAS_OCR:
        if st.button("Extract Data Dari Gambar"):
            pil_imgs = [Image.open(f) for f in uploaded_files]
            with st.spinner("Processing OCR Table Matrix Elements..."):
                ocr_results = process_broker_ocr(pil_imgs)
                if ocr_results:
                    st.session_state["table_data"] = pd.DataFrame(ocr_results)
                    st.success(f"Berhasil mengekstrak {len(ocr_results)} baris broker!")
                else:
                    st.error("Gagal mendeteksi tabel. Pastikan screenshot menampilkan kolom BY/SL Stockbit dengan jelas.")

    st.markdown("---")
    st.subheader("🧠 Adaptive Memory Status")
    st.progress(min(1.0, st.session_state["v12_memory_runs"] / 10.0))
    st.caption(f"Long-term Bias Anchoring: {st.session_state['v12_cumulative_bias']:.4f}")

col_inp, col_grid = st.columns([1, 2])
with col_inp:
    st.subheader("▼ INPUT ASSET")
    ticker = st.text_input("Ticker Code (IDX)", value="BBRI").upper().strip()
    run_btn = st.button("▶ RUN FULL QUANT ENGINE DEPLOYMENT", use_container_width=True)

with col_grid:
    st.subheader("▼ REAL-TIME BROKER DATA MATRIX")
    edited_df = st.data_editor(st.session_state["table_data"], num_rows="dynamic", use_container_width=True)

if run_btn:
    if not st.session_state.get("gemini_api_key"):
        st.error("❌ Masukkan API Key untuk mengaktifkan AI Sentiment Consensus!")
    else:
        with st.spinner("Executing V12 Quantitative Matrix Pipelines..."):
            y_res = fetch_yahoo_raw_data(f"{ticker}.JK")
            df_raw = y_res["data"]
            price_current = y_res["latest_price"]
            
            news_headlines = fetch_rss_news(ticker)
            entries = [BrokerEntry(str(r["Broker"]).upper(), int(r["Buy Lot"]), int(r["Buy Freq"]), int(r["Sell Lot"]), int(r["Sell Freq"]), float(r["Avg Buy Px"]), float(r["Avg Sell Px"])) for _, r in edited_df.iterrows() if pd.notna(r["Broker"])]
            
            brk_out = analyze_broker_summary_ai(entries, price_current)
            ai_out = analyze_with_gemini(ticker, news_headlines, st.session_state["gemini_api_key"], selected_model)
            q_out = compute_quantitative_matrix(df_raw, price_current)
            
            # 5. V12 CONSENSUS MULTI-FACTOR ENGINE
            factors = ["Bandarmology", "AI News", "Momentum", "Mean Reversion"]
            raw_scores = [brk_out["score"], float(ai_out.get("stock_score", 0.0)), q_out["momentum_score"], (50 - q_out["rsi"])/50]
            weights = softmax_weights(raw_scores, temp=SOFTMAX_TEMP)
            combined_consensus_bias = float(np.sum(np.array(raw_scores) * weights))
            
            # Memory state update
            st.session_state["v12_memory_runs"] += 1
            st.session_state["v12_cumulative_bias"] = (st.session_state["v12_cumulative_bias"] * 0.7) + (combined_consensus_bias * 0.3)
            
            t_size = tick(int(price_current))
            sl_price = round((price_current * (1.0 - (0.04 + abs(combined_consensus_bias)*0.05))) / t_size) * t_size
            tp1_price = round((price_current * (1.0 + (0.05 + combined_consensus_bias*0.1))) / t_size) * t_size
            
            mc_out = run_monte_carlo_v12(price_current, q_out["volatility"], combined_consensus_bias, tp1_price, sl_price)
            r_out = compute_advanced_risk_metrics(df_raw, sl_price, price_current)
            
            # Simpan Riwayat
            st.session_state["prediction_history"].insert(0, {"ticker": ticker, "time": (datetime.datetime.utcnow() + datetime.timedelta(hours=7)).strftime("%d %b %Y, %H:%M WIB"), "price": price_current, "bias": combined_consensus_bias, "tp1": tp1_price, "sl": sl_price, "action": "BUY / ACCUM" if combined_consensus_bias > 0.05 else "REDUCE / AVOID" if combined_consensus_bias < -0.05 else "HOLD"})

            # RENDERING INTERFACE OUTPUT
            st.success(f"### 🎯 V12 QUANTUM ENGINE SUMMARY: {ticker}")
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Yahoo Current Price", f"Rp {price_current:,.0f}")
            cm2.metric("Consensus Bias", f"{combined_consensus_bias:.4f}")
            cm3.metric("MC Mean Target", f"Rp {mc_out['mean_target']:,.0f}")
            cm4.metric("Recommendation", st.session_state["prediction_history"][0]["action"])
            
            st.dataframe(pd.DataFrame({"Factor": factors, "Raw Score": raw_scores, "Dynamic Weight": weights}))

st.markdown("---")
st.subheader("📋 History Log")
if st.session_state["prediction_history"]:
    st.dataframe(pd.DataFrame(st.session_state["prediction_history"]), use_container_width=True)
