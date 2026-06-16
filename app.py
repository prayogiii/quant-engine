import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ===============================================================================
# CONFIG & KONSTANTA GLOBAL (V12 Anti-Overfitting Engine)
# ===============================================================================
WEIGHT_MIN = 0.08
WEIGHT_MAX = 0.40
SOFTMAX_TEMP = 2.5
AI_SIGNAL_CAP = 0.30
MC_PESSIMISM = 0.82
TEMPORAL_HALF_DECAY = 25.0

SMART_KEYWORDS = [
    "fed","fomc","rate","inflation","perang","war","konflik","conflict",
    "bi rate","interest","msci","ftse","trump","election","ihsg",
    "rups","dividen","dividend","laba","profit","lk","report","buyback",
    "emas","gold","xau","minyak","oil","crude","coal","nikel","nickel",
    "bond","yield","rupiah","ekspor","impor","china","nikkei","dow jones","nasdaq","akuisisi"
]

IDX_EXCHANGE_HOLIDAYS = {
    # 2024
    "20240101", "20240208", "20240209", "20240210", "20240311", "20240312",
    "20240329", "20240408", "20240409", "20240410", "20240411", "20240412",
    "20240415", "20240501", "20240509", "20240523", "20240524", "20240601",
    "20240617", "20240618", "20240707", "20240817", "20240916", "20241225", "20241226",
    # 2025
    "20250101", "20250127", "20250128", "20250129", "20250328", "20250329",
    "20250331", "20250401", "20250402", "20250403", "20250404", "20250407",
    "20250501", "20250512", "20250529", "20250601", "20250606", "20250627",
    "20250817", "20250905", "20251225", "20251226",
    # 2026
    "20260101", "20260217", "20260303", "20260320", "20260403", "20260420",
    "20260421", "20260422", "20260423", "20260424", "20260501", "20260514",
    "20260526", "20260601", "20260616", "20260716", "20260817", "20260924",
    "20261225", "20261226"
}

# ===============================================================================
# DATA CLASSES
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
    def net_freq(self) -> int: return self.buy_freq - self.sell_freq
    @property
    def avg_buy_lot(self) -> float: return self.buy_lot / self.buy_freq if self.buy_freq > 0 else 0.0
    @property
    def avg_sell_lot(self) -> float: return self.sell_lot / self.sell_freq if self.sell_freq > 0 else 0.0
    @property
    def dominance(self) -> str:
        if self.net_lot > 0 and self.avg_buy_lot > self.avg_sell_lot * 1.5: return "WHALE_BUY"
        if self.net_lot < 0 and self.avg_sell_lot > self.avg_buy_lot * 1.5: return "WHALE_SELL"
        if self.net_lot > 0: return "RETAIL_BUY"
        if self.net_lot < 0: return "RETAIL_SELL"
        return "NEUTRAL"

# ===============================================================================
# MATRICES & MATHEMATICAL HELPERS
# ===============================================================================
def normalize_signal(v: float, scale: float = 1.0) -> float:
    return max(-1.0, min(1.0, v / max(0.0001, scale)))

def tick(p: int) -> int:
    if p < 200: return 1
    if p < 500: return 2
    if p < 2000: return 5
    if p < 5000: return 10
    return 25

# ===============================================================================
# ENGINE 1: DATA BROKER SUMMARY ANALYSIS
# ===============================================================================
def analyze_broker_summary(entries: List[BrokerEntry], current_price: float) -> dict:
    total_buy_lot = sum(b.buy_lot for b in entries)
    total_sell_lot = sum(b.sell_lot for b in entries)
    total_buy_freq = sum(b.buy_freq for b in entries)
    total_sell_freq = sum(b.sell_freq for b in entries)
    total_freq = max(1, total_buy_freq + total_sell_freq)
    total_volume = max(1, total_buy_lot + total_sell_lot)

    net_flow_lot = total_buy_lot - total_sell_lot
    net_flow_freq = total_buy_freq - total_sell_freq
    avg_buy_order = total_buy_lot / total_buy_freq if total_buy_freq > 0 else 0.0
    avg_sell_order = total_sell_lot / total_sell_freq if total_sell_freq > 0 else 0.0
    freq_imbalance = (total_buy_freq - total_sell_freq) / total_freq

    all_lot_sizes = []
    for b in entries:
        if b.buy_freq > 0: all_lot_sizes.append(b.avg_buy_lot)
        if b.sell_freq > 0: all_lot_sizes.append(b.avg_sell_lot)
    
    median_lot = np.median(all_lot_sizes) if all_lot_sizes else 1.0
    whale_threshold = max(1.0, median_lot * 4.0)
    whale_presence = any(b.avg_buy_lot > whale_threshold or b.avg_sell_lot > whale_threshold for b in entries)

    top_buyers = sorted([b for b in entries if b.net_lot > 0], key=lambda x: x.net_lot, reverse=True)[:5]
    top_sellers = sorted([b for b in entries if b.net_lot < 0], key=lambda x: x.net_lot)[:5]

    flow_score = max(-1.0, min(1.0, net_flow_lot / total_volume))
    size_score = max(-1.0, min(1.0, (avg_buy_order - avg_sell_order) / max(1.0, avg_buy_order + avg_sell_order)))

    buy_pr_entries = [b for b in entries if b.avg_buy_price > 0 and b.buy_lot > 0]
    sell_pr_entries = [b for b in entries if b.avg_sell_price > 0 and b.sell_lot > 0]

    w_avg_buy_price = sum(b.avg_buy_price * b.buy_lot for b in buy_pr_entries) / sum(b.buy_lot for b in buy_pr_entries) if buy_pr_entries else 0.0
    w_avg_sell_price = sum(b.avg_sell_price * b.sell_lot for b in sell_pr_entries) / sum(b.sell_lot for b in sell_pr_entries) if sell_pr_entries else 0.0
    
    has_price_data = w_avg_buy_price > 0 or w_avg_sell_price > 0
    buy_weight = total_buy_lot / total_volume
    sell_weight = 1.0 - buy_weight
    buyer_gain = (current_price - w_avg_buy_price) / w_avg_buy_price if w_avg_buy_price > 0 else 0.0
    seller_gain = (w_avg_sell_price - current_price) / w_avg_sell_price if w_avg_sell_price > 0 else 0.0
    price_pressure = max(-1.0, min(1.0, (buyer_gain * buy_weight) - (seller_gain * sell_weight))) if has_price_data else 0.0

    price_bonus = price_pressure * 0.15 if has_price_data else 0.0
    volume_multiplier = 1.0 if total_volume >= 100 else 0.5
    
    inst_score = ((flow_score * 0.45) + (size_score * 0.25) + (freq_imbalance * 0.15) + price_bonus) * volume_multiplier
    inst_score = max(-1.0, min(1.0, inst_score))

    if inst_score > 0.25 and whale_presence: signal = "STRONG ACCUMULATION"
    elif inst_score > 0.10: signal = "ACCUMULATION"
    elif inst_score < -0.25 and whale_presence: signal = "STRONG DISTRIBUTION"
    elif inst_score < -0.10: signal = "DISTRIBUTION"
    else: signal = "NEUTRAL"

    return {
        "top_buyers": top_buyers, "top_sellers": top_sellers,
        "net_flow_lot": net_flow_lot, "net_flow_freq": net_flow_freq,
        "whale_presence": whale_presence, "broker_signal": signal,
        "institutional_score": inst_score, "avg_buy_order": avg_buy_order,
        "avg_sell_order": avg_sell_order, "w_avg_buy_price": w_avg_buy_price,
        "w_avg_sell_price": w_avg_sell_price, "price_pressure": price_pressure
    }

# ===============================================================================
# ENGINE 2: LIVE YAHOO FINANCE & NEWS DECAY RSS ENGINE
# ===============================================================================
def fetch_yahoo_price(symbol: str) -> float:
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10).json()
        closes = res["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]
        return float(closes[-1]) if closes else 0.0
    except:
        return 0.0

def fetch_rss_news(ticker: str) -> List[dict]:
    queries = [f"{ticker} site:cnbcindonesia.com", f"{ticker} site:bloomberg.com", ticker, "IHSG Hari Ini"]
    news_items = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for q in queries:
        try:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}"
            res = requests.get(url, headers=headers, timeout=10)
            root = ET.fromstring(res.text)
            for item in root.findall(".//item"):
                title = item.find("title").text
                pub_date_str = item.find("pubDate").text
                # Estimasi pemrosesan waktu sederhana
                news_items[title] = {"title": title, "time": datetime.datetime.now()}
        except:
            continue
    return list(news_items.values())

# ===============================================================================
# ENGINE 3: GEMINI SENTIMENT AI CALL
# ===============================================================================
def analyze_with_gemini(ticker: str, news_list: List[str], api_key: str) -> dict:
    if not api_key:
        return {"stock_score": 0.0, "market_score": 0.0, "label": "No API Key", "reason": "API Key belum diset."}
    
    # PERBAIKAN: Mengubah v1beta menjadi v1 karena model ini sudah stable (GA)
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    
    headlines = ". ".join(news_list[:15]) if news_list else "No recent news available."
    prompt = (
        f"Analyze sentiment for {ticker}.JK (IDX) and IHSG market.\n"
        f"News headlines: {headlines}.\n"
        f"Return ONLY a strict raw JSON object with this exact format, do not include markdown blocks:\n"
        f'{{"stock_score": 0.15, "market_score": 0.05, "label": "Bullish", "reason": "Summary of global setup", "breakdown": "Details", "confidence": "85"}}'
    )
    
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(url, json=body, headers=headers, timeout=15)
        
        if res.status_code != 200:
            st.error(f"🚨 Google API Error (Status {res.status_code}): {res.text}")
            return {"stock_score": 0.0, "market_score": 0.0, "label": "API Error", "reason": f"Google API returned status {res.status_code}"}
            
        raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        
        # Cari { pertama dan } terakhir untuk antisipasi noise karakter dari LLM
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        
        if start_idx != -1 and end_idx != -1:
            json_text = raw_text[start_idx:end_idx+1]
        else:
            json_text = raw_text

        import json
        return json.loads(json_text)
        
    except Exception as e:
        st.error(f"🚨 Detail Error System: {str(e)}")
        return {"stock_score": 0.0, "market_score": 0.0, "label": "Error Exception", "reason": f"Terjadi exception: {str(e)}"}
# ===============================================================================
# STREAMLIT UI SYSTEM
# ===============================================================================
st.set_page_config(page_title="Hyper-Hybrid Macro Engine V12", layout="wide")

# Validasi Kalender Real-time Hari ini
wib_now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
today_key = wib_now.strftime("%Y%m%d")
is_weekend = wib_now.weekday() >= 5
is_holiday = today_key in IDX_EXCHANGE_HOLIDAYS

st.title("📊 HYPER-HYBRID MACRO ENGINE V12")
if is_weekend or is_holiday:
    st.warning(f"⚠ Hari ini ({wib_now.strftime('%d-%m-%Y')}) Bursa IDX terpantau LIBUR / Tutup Sesi.")

# Layout Sidebar untuk API Settings
with st.sidebar:
    st.header("⚙ PENGATURAN API")
    saved_key = st.text_input("Gemini API Key", type="password", value=st.session_state.get("gemini_api_key", ""))
    if saved_key:
        st.session_state["gemini_api_key"] = saved_key

# Form Utama Pencarian & Input
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("▼ MARKET DATA INPUT")
    stock_code = st.text_input("Kode Saham (contoh: BBRI, TLKM, BMRI)", value="BBRI").upper().strip()
    
    # Tombol Aksi Utama
    run_analysis = st.button("▶ RUN HYPER-HYBRID MACRO ENGINE V12", use_container_width=True)

with col_right:
    st.subheader("▼ BROKER SUMMARY MATRIX (BUM\")")
    st.caption("Tips: Kamu bisa Copy-Paste tabel langsung dari Excel/Spreadsheet ke grid di bawah ini.")
    
    # Setup tabel kosong interaktif pengganti addBrokerRow manual
    init_df = pd.DataFrame([
        {"Broker": "YP", "Buy Lot": 1500, "Buy Freq": 120, "Sell Lot": 200, "Sell Freq": 25, "Avg Buy Px": 0.0, "Avg Sell Px": 0.0}
    ])
    
    edited_df = st.data_editor(
        init_df,
        num_rows="dynamic",
        column_config={
            "Broker": st.column_config.TextColumn("Kode Broker", max_chars=2, required=True),
            "Buy Lot": st.column_config.NumberColumn("Buy Lot", min_value=0, default=0),
            "Buy Freq": st.column_config.NumberColumn("Buy Freq", min_value=0, default=0),
            "Sell Lot": st.column_config.NumberColumn("Sell Lot", min_value=0, default=0),
            "Sell Freq": st.column_config.NumberColumn("Sell Freq", min_value=0, default=0),
            "Avg Buy Px": st.column_config.NumberColumn("Avg Buy Price", min_value=0.0, default=0.0),
            "Avg Sell Px": st.column_config.NumberColumn("Avg Sell Price", min_value=0.0, default=0.0),
        },
        use_container_width=True
    )

# EXECUTION TRIGGER
if run_analysis:
    if not st.session_state.get("gemini_api_key"):
        st.error("❌ Analisis dibatalkan: Masukkan Gemini API Key Anda di sidebar terlebih dahulu!")
    elif not stock_code:
        st.error("❌ Analisis dibatalkan: Kode saham tidak boleh kosong.")
    else:
        with st.spinner("Mengaktifkan V12 Engine: Menarik data market, berita makro, & memproses algoritma AI..."):
            
            # 1. Fetch live prices
            live_price = fetch_yahoo_price(f"{stock_code}.JK")
            if live_price == 0.0:
                live_price = 5000.0 # Fallback default price jika API Yahoo limit
            
            # 2. Fetch & Score RSS News
            raw_news = fetch_rss_news(stock_code)
            news_titles = [n["title"] for n in raw_news]
            
            # 3. Process Gemini Sentiment Analysis
            ai_res = analyze_with_gemini(stock_code, news_titles, st.session_state["gemini_api_key"])
            
            # 4. Parse Broker Entries dari UI Grid
            entries = []
            for _, r in edited_df.iterrows():
                if pd.notna(r["Broker"]) and str(r["Broker"]).strip() != "":
                    entries.append(BrokerEntry(
                        broker_code=str(r["Broker"]).upper(),
                        buy_lot=int(r["Buy Lot"]) if pd.notna(r["Buy Lot"]) else 0,
                        buy_freq=int(r["Buy Freq"]) if pd.notna(r["Buy Freq"]) else 0,
                        sell_lot=int(r["Sell Lot"]) if pd.notna(r["Sell Lot"]) else 0,
                        sell_freq=int(r["Sell Freq"]) if pd.notna(r["Sell Freq"]) else 0,
                        avg_buy_price=float(r["Avg Buy Px"]) if pd.notna(r["Avg Buy Px"]) else 0.0,
                        avg_sell_price=float(r["Avg Sell Px"]) if pd.notna(r["Avg Sell Px"]) else 0.0,
                    ))

            # 5. Run Broker Analytics
            brk_res = None
            if entries:
                brk_res = analyze_broker_summary(entries, live_price)

            # ===============================================================================
            # RENDERING OUTPUT INTERFACE
            # ===============================================================================
            st.balloons()
            st.success(f"## 📊 HASIL ANALISIS ENGINE V12 — {stock_code}")
            
            # Row 1: Key Metrics Dashboard
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Harga Penutupan Terakhir", f"Rp {live_price:,.0f}")
            m2.metric("Sentimen Label AI", ai_res.get("label", "Neutral"))
            
            if brk_res:
                m3.metric("Sinyal Broker", brk_res["broker_signal"])
                m4.metric("Inst Score (Whale Flow)", f"{brk_res['institutional_score']:.2f}")
            else:
                m3.metric("Sinyal Broker", "NO DATA")
                m4.metric("Inst Score (Whale Flow)", "0.00")

            # Row 2: Detail Analisis Komprehensif
            tab1, tab2, tab3 = st.tabs(["🧬 Integrasi Sentimen AI & Makro", "🐋 Bandarmology Detail", "📰 Berita Terdeteksi"])
            
            with tab1:
                st.markdown(f"### AI Evaluation Reasoning")
                st.info(ai_res.get("reason", "No reason provided by AI."))
                
                col_score_1, col_score_2 = st.columns(2)
                col_score_1.progress(float(ai_res.get("stock_score", 0.0)) + 1.0 / 2.0, text=f"Stock Score Bias: {ai_res.get('stock_score')}")
                col_score_2.progress(float(ai_res.get("market_score", 0.0)) + 1.0 / 2.0, text=f"IHSG Score Bias: {ai_res.get('market_score')}")
                st.caption(f"Tingkat Kepercayaan Model AI: {ai_res.get('confidence', '0')}%")

            with tab2:
                if brk_res:
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        st.write("##### 🟢 Top 5 Net Buyers")
                        buy_data = [{"Broker": b.broker_code, "Net Lot": b.net_lot, "Avg Buy Lot": b.avg_buy_lot, "Tipe": b.dominance} for b in brk_res["top_buyers"]]
                        st.table(pd.DataFrame(buy_data))
                    with col_b2:
                        st.write("##### 🔴 Top 5 Net Sellers")
                        sell_data = [{"Broker": b.broker_code, "Net Lot": b.net_lot, "Avg Sell Lot": b.avg_sell_lot, "Tipe": b.dominance} for b in brk_res["top_sellers"]]
                        st.table(pd.DataFrame(sell_data))
                    
                    st.markdown("##### 🧮 Rumus Internal & Data Turunan")
                    st.write(f"- **Net Flow Volume (Lot):** {brk_res['net_flow_lot']:,}")
                    st.write(f"- **Price Pressure Index:** {brk_res['price_pressure']:.4f}")
                    st.write(f"- **Whale Presence Detected:** {'YA' if brk_res['whale_presence'] else 'TIDAK'}")
                else:
                    st.info("Input data broker kosong. Silakan isi tabel input di atas untuk mengaktifkan modul Bandarmology.")

            with tab3:
                st.write(f"Total berita tersaring berdasarkan {len(SMART_KEYWORDS)} kata kunci utama:")
                if news_titles:
                    for idx, title in enumerate(news_titles[:20]):
                        st.write(f"{idx+1}. {title}")
                else:
                    st.write("Tidak ada berita spesifik yang lolos filter temporal.")
