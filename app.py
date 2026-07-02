import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, t as student_t
from scipy.optimize import minimize
import warnings
import urllib.parse
import re
import csv
import os
from datetime import datetime
import pytz
import google.generativeai as genai

# ====================== FALLBACK HANDLERS ======================
PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except ImportError:
    PLOTLY_AVAILABLE = False

SENTIMENT_AVAILABLE = True
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
except ImportError:
    SENTIMENT_AVAILABLE = False

RSS_AVAILABLE = True
try:
    import feedparser
except ImportError:
    RSS_AVAILABLE = False

TRANSLATOR_AVAILABLE = True
try:
    from deep_translator import GoogleTranslator
except ImportError:
    TRANSLATOR_AVAILABLE = False
# =================================================================

warnings.filterwarnings("ignore")

# ==========================================
# KONFIGURASI FILE RIWAYAT & SESSION STATE
# ==========================================
RIWAYAT_FILE = "riwayat_analisis.csv"

HEADER_RIWAYAT = [
    "Waktu", "Saham", "Harga", "Sinyal", "Estimasi", "Prob Naik",
    "RRR", "Sentimen", "Rezim", "TP%", "SL%", "AI_Insight", "Ringkasan_AI"
]

def simpan_riwayat(ringkasan):
    file_exists = os.path.isfile(RIWAYAT_FILE)
    with open(RIWAYAT_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER_RIWAYAT)
        if not file_exists:
            writer.writeheader()
        for key in HEADER_RIWAYAT:
            if key not in ringkasan:
                ringkasan[key] = ""
        writer.writerow(ringkasan)

def muat_riwayat_dari_csv():
    if not os.path.isfile(RIWAYAT_FILE):
        return []
    with open(RIWAYAT_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        riwayat = list(reader)
    riwayat.sort(key=lambda x: x.get('Waktu', ''), reverse=True)
    return riwayat

if "riwayat" not in st.session_state:
    st.session_state.riwayat = muat_riwayat_dari_csv()

# ==========================================
# FUNGSI AI GEMINI
# ==========================================
def dapatkan_model_gemini(api_key):
    if not api_key:
        return None, "API key belum diisi."
    try:
        genai.configure(api_key=api_key)
        available = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_id = m.name.split('/')[-1]
                available.append(model_id)
        if not available:
            return None, "Tidak ada model Gemini."
        for model_id in available:
            try:
                model = genai.GenerativeModel(model_id)
                model.generate_content("test", generation_config={"max_output_tokens": 1})
                return model, None
            except Exception:
                continue
        return None, "Model gagal digunakan."
    except Exception as e:
        return None, f"Error: {str(e)}"

def buat_ringkasan_ai(insight_panjang, api_key):
    model, error = dapatkan_model_gemini(api_key)
    if error:
        return ""
    prompt = f"Ringkas insight berikut menjadi 1-2 kalimat dalam Bahasa Indonesia yang padat dan informatif:\n\n{insight_panjang}"
    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except:
        return insight_panjang[:150] + "..."

def analisis_saham_dengan_ai(data_saham, riwayat, api_key, semua_ringkasan=""):
    model, error = dapatkan_model_gemini(api_key)
    if error:
        return None, error

    riwayat_text = ""
    if riwayat:
        riwayat_text = "Riwayat analisis terbaru:\n"
        for r in riwayat[:5]:
            base = f"- {r['Waktu']} | Sinyal: {r['Sinyal']} | RRR: {r['RRR']} | Rezim: {r['Rezim']}"
            riwayat_text += base + "\n"
    else:
        riwayat_text = "Belum ada riwayat sebelumnya."

    ringkasan_text = ""
    if semua_ringkasan:
        ringkasan_text = "Ringkasan analisis sebelumnya untuk saham ini:\n" + semua_ringkasan

    prompt = f"""
Anda adalah asisten analis saham profesional. Berikut data analisis teknikal dan fundamental saham {data_saham['Saham']}:

- Harga terakhir: Rp {data_saham['Harga']}
- Sinyal saat ini: {data_saham['Sinyal']}
- Rezim Pasar: {data_saham['Rezim']}
- Sentimen Berita: {data_saham['Sentimen']}
- Risk/Reward Ratio (RRR): {data_saham['RRR']}
- Probabilitas Naik Besok: {data_saham['Prob Naik']}
- Take Profit (R1): +{data_saham['TP%']}%
- Stop Loss (S2): -{data_saham['SL%']}%
- Estimasi Harga Besok: Rp {data_saham['Estimasi']}
- Beta terhadap IHSG: {data_saham.get('Beta', 'N/A')}
- Win Rate Backtest: {data_saham.get('WinRate', 'N/A')}
- Profit Factor Backtest: {data_saham.get('ProfitFactor', 'N/A')}
- Max Drawdown Backtest: {data_saham.get('MaxDD', 'N/A')}
- Alokasi Kelly Maks: {data_saham.get('Kelly', 'N/A')}%
- Fundamental: Market Cap: {data_saham.get('Fundamental_MC', 'N/A')}, PER: {data_saham.get('Fundamental_PER', 'N/A')}, PBV: {data_saham.get('Fundamental_PBV', 'N/A')}, ROE: {data_saham.get('Fundamental_ROE', 'N/A')}, D/E: {data_saham.get('Fundamental_DE', 'N/A')}

{ringkasan_text}

{riwayat_text}

Berdasarkan data di atas, berikan analisis ringkas (Bahasa Indonesia) yang mencakup:
- Makna sinyal dalam konteks saat ini
- Kekuatan dan kelemahan saham
- Risiko utama
- Rekomendasi langkah selanjutnya (buy/hold/sell) dengan alasan singkat
- Jika ada pola dari riwayat, sebutkan.
Gunakan bahasa mudah dipahami trader, maksimal 4 paragraf pendek.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), None
    except Exception as e:
        return None, f"Gagal menghasilkan insight AI: {str(e)}"

def analisis_riwayat_global(riwayat_data, api_key):
    model, error = dapatkan_model_gemini(api_key)
    if error:
        return None, error
    if not riwayat_data:
        return None, "Belum ada riwayat."

    prompt = "Berikut adalah riwayat analisis saham yang telah dilakukan:\n\n"
    for r in riwayat_data[:30]:
        prompt += (
            f"- {r['Waktu']} | {r['Saham']} | Sinyal: {r['Sinyal']} | "
            f"Harga: {r['Harga']} | RRR: {r['RRR']} | Sentimen: {r['Sentimen']} | "
            f"Rezim: {r['Rezim']} | TP%: {r['TP%']}% | SL%: {r['SL%']}%\n"
        )
    prompt += (
        "\nBerdasarkan data di atas, berikan analisis ringkas (Bahasa Indonesia) dalam bentuk NARASI yang mengalir, "
        "TANPA bullet poin, TANPA header, TANPA kata 'Format:', 'Columns:', 'Timestamp', 'Total entries', 'Timeframe'. "
        "JANGAN gunakan notasi $ atau LaTeX. "
        "Sampaikan analisis Anda dalam 4-5 paragraf pendek yang mencakup: "
        "pola sinyal yang sering muncul, saham dengan peluang terbaik, rekomendasi perbaikan strategi, dan insight tambahan."
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), None
    except Exception as e:
        return None, f"Gagal menghasilkan insight: {str(e)}"

def bersihkan_teks_ai(teks):
    if not teks:
        return teks
    # Hapus header markdown
    teks = re.sub(r'^#{1,3}\s*', '', teks, flags=re.MULTILINE)
    # Hapus bold/italic
    teks = re.sub(r'\*\*', '', teks)
    teks = re.sub(r'\*', '', teks)
    # Hapus tanda $ (LaTeX)
    teks = re.sub(r'\$', '', teks)
    # Hapus baris yang mengandung kata-kunci tidak diinginkan
    baris = teks.split('\n')
    baris_bersih = []
    for b in baris:
        b_lower = b.lower()
        if not any(kata in b_lower for kata in ['format:', 'columns:', 'timestamp', 'total entries:', 'timeframe:']):
            baris_bersih.append(b)
    teks = '\n'.join(baris_bersih)
    # Ganti newline dengan <br> untuk HTML
    teks = teks.replace('\n', '<br>')
    return teks

# ==========================================
# KONFIGURASI HALAMAN & STYLING
# ==========================================
st.set_page_config(page_title="Quant Risk Engine Pro v8", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0f1116; color: #ffffff; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; color: #00ffcc; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #8892b0; }
    .stButton>button { width: 100%; background-color: #1f2937; color: white; border: 1px solid #374151; }
    .stButton>button:hover { background-color: #374151; border-color: #00ffcc; }
    h1, h2, h3 { color: #f3f4f6; }
    .translated { color: #cbd5e1; font-size: 13px; }
    .source { color: #6b7280; font-size: 11px; }
    .summary-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px; padding: 20px; margin: 10px 0; border: 1px solid #334155;
    }
    .action-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px; padding: 20px; margin: 10px 0; border-left: 5px solid #00ffcc;
    }
    .section-title { color: #00ffcc; font-size: 18px; font-weight: bold; margin-bottom: 12px; }
    .summary-item { color: #cbd5e1; font-size: 15px; margin-bottom: 8px; }
    .fundamental-table { width: 100%; border-collapse: collapse; color: #cbd5e1; }
    .fundamental-table td { padding: 6px 12px; border-bottom: 1px solid #334155; }
    .fundamental-table td:first-child { color: #8892b0; width: 180px; }
    
    .ai-insight-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #8b5cf6;
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.6;
    }
    .ai-insight-card h3 {
        color: #a78bfa;
        margin-top: 0;
        font-size: 20px;
    }
    .ai-insight-card p {
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 📊 QuantRisk Pro")
    st.markdown("Masukkan kode saham IHSG untuk analisis lengkap.")
    ticker_raw = st.text_input("🔍 Kode Saham", value="BBRI", placeholder="Contoh: BBRI, TLKM, BMRI").upper().strip()
    if ticker_raw and not ticker_raw.endswith(".JK"):
        ticker_input = f"{ticker_raw}.JK"
    else:
        ticker_input = ticker_raw

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("🚀 ANALISIS", use_container_width=True)
    with col2:
        if st.button("🗑️ Reset Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache dibersihkan!")
    st.markdown("---")

    st.subheader("📜 Riwayat Analisis")
    if st.session_state.riwayat:
        for r in st.session_state.riwayat[:10]:
            with st.expander(f"{r['Saham']} - {r['Sinyal']} ({r['Waktu']})"):
                st.markdown(f"**Harga:** Rp {r['Harga']}")
                st.markdown(f"**Estimasi Besok:** Rp {r['Estimasi']}")
                st.markdown(f"**Prob Naik:** {r['Prob Naik']}")
                st.markdown(f"**RRR:** {r['RRR']}")
                st.markdown(f"**Sentimen:** {r['Sentimen']}")
                st.markdown(f"**Rezim:** {r['Rezim']}")
                st.markdown(f"**TP%:** {r['TP%']}% | **SL%:** {r['SL%']}%")
                ringkasan_ai = r.get("Ringkasan_AI", "").strip()
                if ringkasan_ai:
                    st.markdown("💬 **Ringkasan AI:**")
                    st.caption(ringkasan_ai)
        if len(st.session_state.riwayat) > 10:
            st.caption(f"Menampilkan 10 dari {len(st.session_state.riwayat)} riwayat.")
    else:
        st.caption("Belum ada riwayat.")

    st.markdown("---")
    st.subheader("🧠 AI (Gemini)")

    def get_api_key():
        try:
            return st.secrets["GEMINI_API_KEY"]
        except KeyError:
            pass
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            return env_key
        return st.session_state.get("gemini_api_key", "")

    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = get_api_key()

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key,
        placeholder="AIza...",
        help="Kunci API Gemini. Disimpan di secrets atau env."
    )
    if api_key:
        st.session_state.gemini_api_key = api_key

    ai_riwayat_btn = st.button("📊 Analisis Riwayat dgn AI", use_container_width=True)

    if st.button("🗑️ Hapus Semua Riwayat"):
        if os.path.isfile(RIWAYAT_FILE):
            os.remove(RIWAYAT_FILE)
        st.session_state.riwayat = []
        st.success("Riwayat dihapus!")

    st.markdown("---")
    st.caption("Data dari Yahoo Finance. Bukan rekomendasi investasi.")

# ==================== FUNGSI DATA & INDIKATOR ====================
@st.cache_data(ttl=3600)
def load_stock_data(ticker):
    df = yf.download(ticker, period="2y")
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=3600)
def load_ihsg_data():
    df = yf.download("^JKSE", period="2y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def compute_adx_series(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()

def get_google_news_rss(query_str, num=5):
    if not RSS_AVAILABLE: return [], "RSS tidak tersedia"
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query_str)}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)
        news = []
        for e in feed.entries[:num]:
            title = e.get('title', '').strip()
            summary = re.sub('<[^<]+?>', '', e.get('summary', ''))
            news.append({'title': title, 'summary': summary, 'source': 'Google News'})
        return news, None
    except Exception as e:
        return [], str(e)

def get_yahoo_search_news(query_str, num=5):
    try:
        items = yf.Search(query_str).news or []
        news = []
        for item in items[:num]:
            inner = item.get('content') or item
            title = (inner.get('title') or inner.get('shortTitle') or inner.get('headline') or '')
            summary = (inner.get('summary') or inner.get('longSummary') or inner.get('description') or '')
            if title: news.append({'title': title, 'summary': summary, 'source': 'Yahoo Search'})
        return news, None
    except: return [], "Yahoo Search gagal"

def filter_relevant(news_list, ticker):
    keywords = [ticker.lower(), 'saham', 'ihsg', 'bei', 'idx']
    filtered = [n for n in news_list if any(k in (n['title']+n['summary']).lower() for k in keywords)]
    return filtered if filtered else news_list

def analyze_sentiment_weighted(news_items, translator):
    if not SENTIMENT_AVAILABLE or not news_items: return 0.0
    analyzer = SentimentIntensityAnalyzer()
    total_w, w_sum = 0, 0
    for i, item in enumerate(news_items):
        text = f"{item['title']}. {item['summary']}" if item['summary'] else item['title']
        if any(ord(c) > 127 for c in text) and translator:
            try: text = translator.translate(text)
            except: pass
        score = analyzer.polarity_scores(text)['compound']
        weight = 1 / (i + 1)
        w_sum += score * weight
        total_w += weight
    return w_sum / total_w if total_w > 0 else 0.0

def estimate_theta_ou(close_series):
    log_price = np.log(close_series.dropna())
    log_lag = log_price.shift(1).dropna()
    diff = log_price.diff().dropna()
    common_idx = diff.index.intersection(log_lag.index)
    if len(common_idx) < 20: return 0.05
    y = diff.loc[common_idx].values
    X = np.vstack([np.ones(len(common_idx)), log_lag.loc[common_idx].values]).T
    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    theta = -coeff[1] if coeff[1] < 0 else 0.05
    return theta

REGIME_INFO = {
    "Strong Bullish 🚀": "Tren naik kuat dengan momentum tinggi.",
    "Bullish 📈": "Tren naik stabil. Kondisi sehat untuk akumulasi.",
    "Panic Sell 🚨": "Penurunan tajam, sering oversold.",
    "Bearish 🔻": "Tren turun terkendali.",
    "Early Recovery 🔄": "Harga di atas EMA20 tapi EMA20 < EMA50.",
    "Distribution 📉": "Harga di bawah EMA20, EMA20 > EMA50.",
    "Konsolidasi Tren ↔️": "Trending namun harga bolak-balik di EMA.",
    "Bullish Accumulation 🏗️": "Sideways dengan harga > EMA.",
    "Bearish Accumulation 🧊": "Sideways di bawah EMA.",
    "Sideways Bias Naik ↗️": "Sideways cenderung naik.",
    "Sideways Bias Turun ↘️": "Sideways cenderung turun.",
    "Sideways Normal ↔️": "Sideways moderat, tunggu katalis."
}

# ==================== PROSES ANALISIS ====================
if run_btn:
    # ... (seluruh perhitungan kuantitatif tetap sama seperti kode sebelumnya)
    # ... (bagian ini tidak diubah)

    # ==================== AI INSIGHT OTOMATIS ====================
    st.markdown("---")
    if st.session_state.get("gemini_api_key"):
        with st.spinner("🧠 AI sedang menganalisis hasil dan riwayat..."):
            data_ai = {
                "Saham": ticker_input,
                "Harga": f"{harga_terakhir:,.0f}",
                "Sinyal": signal,
                "Rezim": regime,
                "Sentimen": f"{avg_sentiment:.2f} ({sentimen_status})",
                "RRR": f"{rrr:.2f}",
                "Prob Naik": f"{prob_bull:.1f}%",
                "TP%": f"{tp_pct:.1f}",
                "SL%": f"{sl_pct:.1f}",
                "Estimasi": f"{est_besok:,.0f}",
                "Beta": f"{beta_ihsg:.2f}x",
                "WinRate": f"{win_bt:.1%}" if trades_bt else "N/A",
                "ProfitFactor": f"{pf_bt:.2f}" if trades_bt else "N/A",
                "MaxDD": f"{max_dd_bt:.2f}%" if trades_bt else "N/A",
                "Kelly": f"{kelly_adj*100:.1f}",
                "Fundamental_MC": f"{mc:,.0f}" if mc else "N/A",
                "Fundamental_PER": f"{per:.2f}" if per else "N/A",
                "Fundamental_PBV": f"{pbv:.2f}" if pbv else "N/A",
                "Fundamental_ROE": f"{roe*100:.1f}" if roe else "N/A",
                "Fundamental_DE": f"{de:.2f}" if de else "N/A"
            }

            riwayat_saham = [r for r in st.session_state.riwayat if r['Saham'] == ticker_input]
            
            semua_ringkasan = "\n".join([
                f"- {r['Waktu']}: {r.get('Ringkasan_AI', '').strip()}"
                for r in riwayat_saham if r.get('Ringkasan_AI', '').strip()
            ])
            
            hasil_ai, error_ai = analisis_saham_dengan_ai(
                data_ai,
                riwayat_saham[:5],
                st.session_state.gemini_api_key,
                semua_ringkasan
            )
            
            if not error_ai and hasil_ai:
                ringkasan_baru = buat_ringkasan_ai(hasil_ai, st.session_state.gemini_api_key)
                ringkasan["AI_Insight"] = hasil_ai
                ringkasan["Ringkasan_AI"] = ringkasan_baru if ringkasan_baru else hasil_ai[:150]
                
                hasil_ai_bersih = bersihkan_teks_ai(hasil_ai)
                html_ai = f"""
                <div class="ai-insight-card">
                    <h3>🤖 Insight AI</h3>
                    <p>{hasil_ai_bersih}</p>
                </div>
                """
                st.markdown(html_ai, unsafe_allow_html=True)
            elif error_ai:
                st.warning(f"AI tidak dapat memberikan insight: {error_ai}")
    else:
        st.info("💡 Isi API Key Gemini di sidebar untuk mendapatkan insight AI otomatis.")

    simpan_riwayat(ringkasan)
    st.session_state.riwayat.insert(0, ringkasan)
    if len(st.session_state.riwayat) > 50:
        st.session_state.riwayat.pop()

# --- ANALISIS RIWAYAT DENGAN AI (TOMBOL SIDEBAR) ---
if ai_riwayat_btn:
    if not st.session_state.gemini_api_key:
        st.error("Masukkan API Key terlebih dahulu!")
    elif not st.session_state.riwayat:
        st.warning("Belum ada riwayat.")
    else:
        with st.spinner("🧠 AI menganalisis riwayat..."):
            hasil, error = analisis_riwayat_global(st.session_state.riwayat, st.session_state.gemini_api_key)
            if error:
                st.error(error)
            elif hasil:
                hasil_bersih = bersihkan_teks_ai(hasil)
                html_riwayat_ai = f"""
                <div class="ai-insight-card" style="border-left-color: #06b6d4;">
                    <h3 style="color: #67e8f9;">📊 Insight AI dari Riwayat</h3>
                    <p>{hasil_bersih}</p>
                </div>
                """
                st.markdown(html_riwayat_ai, unsafe_allow_html=True)
