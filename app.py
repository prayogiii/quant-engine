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
# FUNGSI AI GEMINI (PROMPT & PEMBERSIHAN AGGRESIF)
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

Tulis analisis ringkas dalam BAHASA INDONESIA, 4 paragraf pendek, langsung ke inti. Jangan gunakan bullet poin atau header.
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

    # Prompt ultra‑ketat: hanya data, larangan keras bahasa Inggris & proses berpikir
    prompt = "Data riwayat analisis saham:\n"
    for r in riwayat_data[:30]:
        prompt += (
            f"{r['Waktu']} | {r['Saham']} | Sinyal: {r['Sinyal']} | "
            f"RRR: {r['RRR']} | Sentimen: {r['Sentimen']} | Rezim: {r['Rezim']}\n"
        )
    prompt += (
        "\nTULIS HANYA ANALISIS AKHIR DALAM BAHASA INDONESIA, 3-4 PARAGRAF PENDEK. "
        "JANGAN SERTAKAN PROSES BERPIKIR, INSTRUKSI, DRAFT, ATAU BAHASA INGGRIS. "
        "JANGAN MENGGUNAKAN BULLET POIN ATAU HEADER. "
        "FOKUS PADA POLA SINYAL, SAHAM TERBAIK, PERBAIKAN STRATEGI, DAN INSIGHT TAMBAHAN."
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), None
    except Exception as e:
        return None, f"Gagal menghasilkan insight: {str(e)}"

def bersihkan_teks_ai(teks):
    if not teks:
        return teks
    # Hapus markdown dan karakter aneh
    teks = re.sub(r'^#{1,3}\s*', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'\*\*', '', teks)
    teks = re.sub(r'\*', '', teks)
    teks = re.sub(r'\$', '', teks)
    # Proses per baris
    baris = teks.split('\n')
    baris_bersih = []
    for b in baris:
        b_stripped = b.strip()
        if not b_stripped:
            continue
        # Buang baris yang hanya berisi bullet
        if b_stripped in ['-', '*', '•']:
            continue
        # Buang baris yang merupakan pertanyaan retoris / pendek
        if '?' in b_stripped and len(b_stripped) < 100:
            continue
        b_lower = b_stripped.lower()
        # Daftar kata kunci instruksional / meta / bahasa Inggris yang tidak diinginkan
        if any(kata in b_lower for kata in [
            'no bullets?', 'no headers?', 'no ?', 'indonesian language?',
            'check constraints', 'constraint', 'draft', 'final polish',
            'self-correction', 'paragraph', 'input:', 'goal:', 'dates:',
            'signals:', 'patterns:', 'formatting:', 'tone:', 'directly cover:',
            'language:', 'date:', 'strong buy:', 'buy (tactical):', 'hold/wait:',
            'avoid:', 'stocks analysis:', 'rrr (risk-reward ratio):',
            'sentiments/regimes:', 'note the shift.', 'look at the', 'focus on',
            'no instructions', 'no thinking process', 'just the final analysis',
            'berdasarkan data di atas', 'tulis analisis ringkas', 'bahasa indonesia',
            'langsung ke inti', 'jangan gunakan bullet poin', 'jangan ulangi instruksi',
            'fokus pada makna sinyal', 'kekuatan/kelemahan', 'risiko utama',
            'rekomendasi langkah selanjutnya', 'jika ada pola dari riwayat',
            'format:', 'columns:', 'timestamp', 'total entries:', 'timeframe:',
        ]):
            continue
        # Buang baris yang hanya berisi angka/tanggal
        if re.match(r'^[\d\s\-:,.]+$', b_lower):
            continue
        # Buang baris yang mayoritas karakter huruf kecil (indikasi bahasa Inggris)
        alpha_count = sum(c.isalpha() for c in b_stripped)
        if alpha_count > 0:
            lower_alpha = sum(c.islower() for c in b_stripped if c.isalpha())
            if lower_alpha / alpha_count > 0.7 and alpha_count > 10:
                continue
        # Buang baris yang terlalu pendek dan mengandung alfabet (mungkin sisa bullet)
        if len(b_stripped) < 10 and alpha_count > 0:
            continue
        # Hapus bullet di awal baris
        b_stripped = re.sub(r'^[\s]*[-*•]\s*', '', b_stripped)
        baris_bersih.append(b_stripped)

    teks = '\n'.join(baris_bersih)
    # Hapus paragraf duplikat
    paragraf = teks.split('\n')
    paragraf_unik = []
    for p in paragraf:
        if p not in paragraf_unik:
            paragraf_unik.append(p)
    teks = '\n'.join(paragraf_unik)
    # Batasi panjang maksimal (opsional)
    if len(teks) > 1500:
        teks = teks[:1500] + "..."
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
    if not ticker_input:
        st.warning("⚠️ Kode saham tidak boleh kosong!")
        st.stop()

    with st.spinner("🤖 Mengunduh data dan memproses analitika kuantitatif..."):
        df = load_stock_data(ticker_input)
        if df.empty:
            st.error("❌ Data tidak ditemukan untuk ticker tersebut.")
            st.stop()

        harga_terakhir = float(df['Close'].iloc[-1])
        returns = df['Close'].pct_change().dropna()
        if len(returns) < 50:
            st.error("❌ Data historis kurang untuk analisa kuantitatif.")
            st.stop()

        # ============ INDIKATOR ============
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['ADX'] = compute_adx_series(df)
        df['Mom5D'] = df['Close'].pct_change(5) * 100
        df['ZScore'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        df['Vol_MA20'] = df['Volume'].rolling(20).mean() if 'Volume' in df.columns else 0

        # ============ FUNDAMENTAL ============
        try:
            ticker_info = yf.Ticker(ticker_input).info
        except:
            ticker_info = {}
        mc = ticker_info.get('marketCap')
        per = ticker_info.get('trailingPE') or ticker_info.get('forwardPE')
        pbv = ticker_info.get('priceToBook')
        roe = ticker_info.get('returnOnEquity')
        de = ticker_info.get('debtToEquity')

        # ============ BERITA & SENTIMEN ============
        news_pool = []
        translator_en = GoogleTranslator(source='auto', target='en') if TRANSLATOR_AVAILABLE else None
        translator_id = GoogleTranslator(source='auto', target='id') if TRANSLATOR_AVAILABLE else None
        rss, _ = get_google_news_rss(f"{ticker_raw} saham")
        if rss: news_pool.extend(rss)
        ysearch, _ = get_yahoo_search_news(f"{ticker_raw} saham")
        if ysearch: news_pool.extend(ysearch)
        news_pool = filter_relevant(news_pool, ticker_raw)
        seen = set()
        unique_news = []
        for n in news_pool:
            if n['title'] not in seen:
                seen.add(n['title'])
                unique_news.append(n)
            if len(unique_news) >= 5:
                break
        avg_sentiment = analyze_sentiment_weighted(unique_news, translator_en)
        headlines = [n['title'] for n in unique_news]
        sources = [n['source'] for n in unique_news]
        translated = []
        for n in unique_news:
            if TRANSLATOR_AVAILABLE and translator_id:
                try:
                    translated.append(translator_id.translate(n['title']))
                except:
                    translated.append("")
            else:
                translated.append("")
        sentimen_status = "Positif 🟢" if avg_sentiment >= 0.05 else ("Negatif 🔴" if avg_sentiment <= -0.05 else "Netral ⚪")

        # ============ THRESHOLD HISTORIS ============
        split_idx = max(126, len(df) - 126)
        df_thresh = df.iloc[:split_idx]
        returns_thresh = df_thresh['Close'].pct_change().dropna()
        adx_threshold = np.percentile(df_thresh['ADX'].dropna(), 75) if not df_thresh['ADX'].dropna().empty else 20
        z_oversold_th = -1.5
        mom_median_th = np.percentile(df_thresh['Mom5D'].dropna(), 50) if not df_thresh['Mom5D'].dropna().empty else 0.0

        def t_loglike(p, d):
            if p[0] <= 2 or p[2] <= 0:
                return np.inf
            return -np.sum(student_t.logpdf(d, p[0], p[1], p[2]))

        res = minimize(t_loglike, [5, returns_thresh.mean(), returns_thresh.std()],
                       bounds=[(2.1, 100), (-0.1, 0.1), (1e-6, None)], args=(returns_thresh,), method='L-BFGS-B')
        df_est, t_loc, t_scale = res.x if res.success else (5, returns_thresh.mean(), returns_thresh.std())

        # ============ REGIME ============
        def get_regime_row(row):
            h, e20, e50, a, z, m = row['Close'], row['EMA20'], row['EMA50'], row['ADX'], row['ZScore'], row['Mom5D']
            if a > adx_threshold:
                if h > e20 and e20 > e50:
                    return ("Strong Bullish 🚀", "RISK-ON 🔥") if (m > mom_median_th or z > z_oversold_th) else ("Bullish 📈", "RISK-ON 🔥")
                elif h < e20 and e20 < e50:
                    return ("Panic Sell 🚨", "RISK-OFF 🛑") if (m < mom_median_th or z < z_oversold_th) else ("Bearish 🔻", "RISK-OFF 🛑")
                elif h > e20 and e20 < e50:
                    return ("Early Recovery 🔄", "TRANSISI ⚠️")
                elif h < e20 and e20 > e50:
                    return ("Distribution 📉", "TRANSISI ⚠️")
                else:
                    return ("Konsolidasi Tren ↔️", "NEUTRAL ⚖️")
            else:
                if h > e20 and e20 > e50:
                    return ("Bullish Accumulation 🏗️", "NEUTRAL ⚖️")
                elif h < e20 and e20 < e50:
                    return ("Bearish Accumulation 🧊", "NEUTRAL ⚖️")
                elif h > e20 and e20 < e50:
                    return ("Sideways Bias Naik ↗️", "NEUTRAL ⚖️")
                elif h < e20 and e20 > e50:
                    return ("Sideways Bias Turun ↘️", "NEUTRAL ⚖️")
                else:
                    return ("Sideways Normal ↔️", "NEUTRAL ⚖️")

        regime, ihsg_cond = get_regime_row(df.iloc[-1])
        adx = df['ADX'].iloc[-1]

        # ============ BETA ============
        try:
            ihsg = load_ihsg_data()
            ihsg_ret = ihsg['Close'].pct_change().dropna()
            common = returns.index.intersection(ihsg_ret.index)
            if len(common) > 20:
                cov = np.cov(returns.loc[common], ihsg_ret.loc[common])
                beta_ihsg = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
            else:
                beta_ihsg = 1.0
        except:
            beta_ihsg = 1.0

        # ============ PIVOT ============
        hi, lo = float(df['High'].iloc[-1]), float(df['Low'].iloc[-1])
        pp = (hi + lo + harga_terakhir) / 3
        r1, s1 = 2 * pp - lo, 2 * pp - hi
        r2, s2 = pp + (hi - lo), pp - (hi - lo)
        res20 = float(df['High'].iloc[-21:-1].max())
        breakout = "YES (🔥)" if harga_terakhir > res20 else "NO"

        # ============ SINYAL ============
        def generate_signals_vectorized(dataframe, mom_th):
            score = pd.Series(0, index=dataframe.index)
            is_uptrend = (dataframe['Close'] > dataframe['EMA20']) & (dataframe['EMA20'] > dataframe['EMA50'])
            score += is_uptrend.astype(int) * 2
            score += (dataframe['Mom5D'] > mom_th).astype(int)
            if 'Volume' in dataframe.columns:
                score += (dataframe['Volume'] > dataframe['Vol_MA20']).astype(int)
            sig = pd.Series("🚨 AVOID", index=dataframe.index)
            sig[score == 1] = "⏸️ HOLD / WAIT"
            sig[score >= 2] = "⚡ BUY (TACTICAL)"
            sig[score >= 3] = "🔥 STRONG BUY"
            is_choppy = (dataframe['ADX'] < 20)
            sig[is_choppy & sig.str.contains("BUY")] = "⏸️ HOLD / WAIT"
            is_oversold = (dataframe['ZScore'] < -1.5) & (dataframe['Close'] < dataframe['EMA20'])
            sig[is_oversold] = "⚡ BUY (TACTICAL)"
            return sig

        df['Signal'] = generate_signals_vectorized(df, mom_median_th)
        signal = df['Signal'].iloc[-1]

        # ============ BACKTEST ============
        df_back = df.iloc[-126:].copy()
        trades, daily_returns = [], []
        in_position, entry_price = False, 0.0
        for i in range(len(df_back)):
            curr_sig = df_back['Signal'].iloc[i]
            curr_close = float(df_back['Close'].iloc[i])
            prev_close = float(df_back['Close'].iloc[i - 1]) if i > 0 else curr_close
            if in_position:
                daily_returns.append((curr_close - prev_close) / prev_close if prev_close else 0)
                if "AVOID" in curr_sig or i == len(df_back) - 1:
                    trades.append((curr_close - entry_price) / entry_price)
                    in_position = False
            else:
                daily_returns.append(0.0)
                if "BUY" in curr_sig:
                    in_position, entry_price = True, curr_close

        if trades:
            win_bt = sum(1 for r in trades if r > 0) / len(trades)
            loss_trades = [r for r in trades if r < 0]
            profit_trades = [r for r in trades if r > 0]
            pf_bt = abs(sum(profit_trades) / sum(loss_trades)) if loss_trades else np.inf
            avg_bt = np.mean(trades)
            equity = np.cumprod([1 + r for r in trades])
            max_dd_bt = float(np.min(equity / np.maximum.accumulate(equity) - 1) * 100) if len(equity) else 0
            daily_ret = np.array(daily_returns)
            sharpe_bt = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() else 0
            trades_bt = len(trades)
        else:
            win_bt = pf_bt = avg_bt = max_dd_bt = sharpe_bt = trades_bt = 0

        # ============ KELLY ============
        roll_max_th = df_thresh['Close'].cummax()
        drawdown_th = (df_thresh['Close'] - roll_max_th) / roll_max_th
        max_dd = float(drawdown_th.min() * 100)
        max_dd_30 = float(drawdown_th.tail(30).min() * 100) if len(drawdown_th) >= 30 else max_dd
        if trades_bt >= 2:
            win_r, avg_g, avg_l = win_bt, np.mean(profit_trades) if profit_trades else 0.01, abs(np.mean(loss_trades)) if loss_trades else 0.01
        else:
            win_r = len(returns_thresh[returns_thresh > 0]) / len(returns_thresh)
            avg_g = returns_thresh[returns_thresh > 0].mean() if win_r > 0 else 0.01
            avg_l = abs(returns_thresh[returns_thresh < 0].mean()) if len(returns_thresh[returns_thresh < 0]) else 0.01
        wl = avg_g / avg_l if avg_l else 1
        kelly_raw = win_r - (1 - win_r) / wl
        ret_skew = float(skew(returns_thresh))
        ret_kurt = float(kurtosis(returns_thresh, fisher=True))
        kurt_penalty = 0.5 if ret_kurt > 3 else 1.0
        kelly_adj = min(0.25, max(0.0, kelly_raw * 0.3 * (0.5 if ret_skew < -0.5 else 1) * kurt_penalty))

        # ============ MONTE CARLO ============
        n_sim, n_days = 2000, 30
        latest_vol = np.sqrt(df['Close'].pct_change().ewm(alpha=0.06).var().iloc[-1])
        scale_corrected = latest_vol / np.sqrt(df_est / (df_est - 2)) if df_est > 2 else latest_vol
        theta_ou = estimate_theta_ou(df['Close'])
        locked_log_mean20 = np.log(df['Close']).tail(20).mean()
        paths = np.zeros((n_days, n_sim))
        current_log = np.expand_dims(np.log(harga_terakhir) * np.ones(n_sim), axis=0)
        for day in range(n_days):
            inov = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=n_sim)
            next_log = current_log[-1] + theta_ou * (locked_log_mean20 - current_log[-1]) + inov
            current_log = np.vstack([current_log, next_log])
            paths[day] = np.exp(next_log)
        mu_ou = theta_ou * (locked_log_mean20 - np.log(harga_terakhir))
        est_besok = float(np.exp(np.log(harga_terakhir) + mu_ou))
        sim_h1 = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=2000)
        prices_besok = harga_terakhir * np.exp(mu_ou + sim_h1)
        low_est, up_est = float(np.percentile(prices_besok, 25)), float(np.percentile(prices_besok, 75))
        hit_tp = (np.any(paths >= r1, axis=0).sum() / n_sim) * 100
        hit_sl = (np.any(paths <= s2, axis=0).sum() / n_sim) * 100
        prob_bull = ((mu_ou + sim_h1 > 0).sum() / 2000) * 100

        tp_pct = ((r1 - harga_terakhir) / harga_terakhir) * 100 if harga_terakhir else 0
        sl_pct = ((harga_terakhir - s2) / harga_terakhir) * 100 if harga_terakhir else 0
        rrr = tp_pct / sl_pct if sl_pct else 0
        rrr_status = "Ideal (≥ 1.5) 🟢" if rrr >= 1.5 else ("Cukup (1.0 - 1.5) 🟡" if rrr >= 1 else "Buruk (< 1.0) 🔴")

        # ============ RINGKASAN AWAL ============
        ringkasan = {
            "Waktu": datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M"),
            "Saham": ticker_input,
            "Harga": f"{harga_terakhir:,.0f}",
            "Sinyal": signal,
            "Estimasi": f"{est_besok:,.0f}",
            "Prob Naik": f"{prob_bull:.1f}%",
            "RRR": f"{rrr:.2f}",
            "Sentimen": f"{avg_sentiment:.2f} ({sentimen_status})",
            "Rezim": regime,
            "TP%": f"{tp_pct:.1f}",
            "SL%": f"{sl_pct:.1f}",
            "AI_Insight": "",
            "Ringkasan_AI": ""
        }

    # ==================== TAMPILAN UTAMA ====================
    st.title("📊 Quant & Risk Engine Pro")
    st.write("Algoritma kuantitatif + Berita + Backtest + AI + Grafik Interaktif + Fundamental")
    st.success(f"✅ Analisis Berhasil: {ticker_input} | Closing Price: Rp {harga_terakhir:,.0f}".replace(",", "."))

    col1, col2, col3 = st.columns(3)
    col1.metric("Sinyal Eksekusi", signal)
    col2.metric("Estimasi Besok", f"Rp {est_besok:,.0f}".replace(",", "."),
                f"50% range: Rp {low_est:,.0f} - {up_est:,.0f}".replace(",", "."))
    col3.metric("Prob. Naik Besok", f"{prob_bull:.1f}%")

    if PLOTLY_AVAILABLE:
        st.header("📈 Chart Harga & Sinyal")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#00ffcc')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', line=dict(color='#f59e0b', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50', line=dict(color='#ef4444', dash='dot')))
        buy_signals = df_back[df_back['Signal'].str.contains("BUY")]
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['Close'],
            mode='markers', name='Buy Signal',
            marker=dict(symbol='triangle-up', size=10, color='#10b981')
        ))
        for level, label, col in [(r1, 'R1', 'orange'), (s1, 'S1', 'red'), (pp, 'PP', 'gray')]:
            fig.add_hline(y=level, line_dash="dash", line_color=col, annotation_text=label)
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # --- RINGKASAN EKSEKUTIF ---
    st.markdown("---")
    st.header("📋 Ringkasan Eksekutif & Rekomendasi")
    if rrr < 1.0 and ("BUY" in signal):
        action_color, action_icon = "#ef4444", "⚠️"
        action_text = f"• <b>KONDISI:</b> Sinyal {signal} tapi <b>RRR BURUK ({rrr:.2f})</b><br>• <b>REKOMENDASI:</b> WAIT & SEE<br>• <b>LANGKAH:</b> Tunda entry, tunggu koreksi ke Support 1."
    elif "STRONG BUY" in signal:
        action_color, action_icon = "#10b981", "🟢"
        action_text = f"• <b>KONDISI:</b> Tren Kuat & Akumulasi Volume<br>• <b>REKOMENDASI:</b> AGGRESSIVE BUY<br>• <b>LANGKAH:</b> Beli bertahap, SL di S2 ({s2:,.0f})."
    elif "BUY" in signal:
        action_color, action_icon = "#f59e0b", "🟡"
        action_text = f"• <b>KONDISI:</b> Tren Valid, RRR Memadai ({rrr:.2f})<br>• <b>REKOMENDASI:</b> BUY ON WEAKNESS<br>• <b>LANGKAH:</b> Entry ideal {s1:,.0f}-{pp:,.0f}, SL {s2:,.0f}."
    elif "HOLD" in signal:
        action_color, action_icon = "#3b82f6", "🔵"
        action_text = f"• <b>KONDISI:</b> Konsolidasi / Transisi<br>• <b>REKOMENDASI:</b> HOLD<br>• <b>LANGKAH:</b> Jangan tambah posisi, pantau SL."
    else:
        action_color, action_icon = "#ef4444", "🔴"
        action_text = f"• <b>KONDISI:</b> Risiko Penurunan / Distribusi<br>• <b>REKOMENDASI:</b> AVOID / LIQUIDATE<br>• <b>LANGKAH:</b> Amankan modal."

    col1, col2 = st.columns([1, 1])
    with col1:
        html_summary = f'<div class="summary-card"><div class="section-title">📌 Profil Risiko Saham Terkalibrasi</div><div class="summary-item">🛡️ <b>Jarak Stop Loss:</b> -{sl_pct:.1f}% (Rp {s2:,.0f})</div><div class="summary-item">🎯 <b>Potensi Keuntungan:</b> +{tp_pct:.1f}% (Rp {r1:,.0f})</div><div class="summary-item">⚖️ <b>Risk:Reward Ratio:</b> 1 : {rrr:.2f} ({rrr_status})</div><div class="summary-item">🏷️ <b>Kategori Rezim & Makro:</b> {regime} | {ihsg_cond}</div><div class="summary-item">🛡️ <b>Alokasi Maks (Kelly):</b> {kelly_adj*100:.1f}% dari Total Ekuitas</div></div>'
        st.markdown(html_summary, unsafe_allow_html=True)
    with col2:
        html_action = f'<div class="action-card" style="border-left-color: {action_color};"><div class="section-title">{action_icon} Panduan Eksekusi Trader</div><div class="summary-item" style="font-size: 15px; margin-top: 8px; line-height: 1.6;">{action_text}</div><hr style="border-color: #334155; margin: 15px 0;"><div style="color: #94a3b8; font-size: 13px;">⚠️ <i>Disclaimer: Hasil pengujian berbasis permodelan matematika probabilitas kuantitatif historis. Keputusan akhir eksekusi modal tetap merupakan tanggung jawab penuh masing-masing investor.</i></div></div>'
        st.markdown(html_action, unsafe_allow_html=True)

    # --- DETAIL EXPANDER ---
    with st.expander("🔍 Lihat Detail Analisis (Berita, Fundamental, Backtest, dll)"):
        st.subheader("📰 Sentimen Berita Terbobot")
        c1, c2 = st.columns([1, 2])
        c1.metric("Sentimen Skor", f"{avg_sentiment:.2f}", sentimen_status)
        with c2:
            st.markdown("**5 Berita Utama Pasar:**")
            for i, h in enumerate(headlines):
                src = sources[i] if i < len(sources) else ""
                t = translated[i] if i < len(translated) else ""
                st.markdown(f"{i+1}. **{h}** <span class='source'>({src})</span>", unsafe_allow_html=True)
                if t and t != h:
                    st.markdown(f"<span class='translated'>🇮🇩 {t}</span>", unsafe_allow_html=True)
        st.divider()

        st.subheader("🧬 Regime Pasar & Volatilitas")
        m1, m2, m3 = st.columns(3)
        m1.metric("Market Regime", regime)
        m2.metric("Kondisi Makro IHSG", ihsg_cond)
        m3.metric("ADX Adaptif", f"{adx:.1f} (Thresh: {adx_threshold:.1f})")
        st.markdown(f"**Insight Regime:** {REGIME_INFO.get(regime, 'Regime tidak terdefinisi.')}")
        st.divider()

        st.subheader("📊 Metrik Fundamental Saham (IDX)")
        if ticker_info:
            def clean_val(val, fmt="{:.2f}"):
                return "N/A" if val is None else fmt.format(val)
            table_html = f"""
            <table class='fundamental-table'>
                <tr><td>Market Cap</td><td>{clean_val(mc, '{:,.0f} IDR')}</td></tr>
                <tr><td>PER</td><td>{clean_val(per, '{:.2f}x')}</td></tr>
                <tr><td>PBV</td><td>{clean_val(pbv, '{:.2f}x')}</td></tr>
                <tr><td>ROE</td><td>{clean_val(roe * 100 if roe else None, '{:.1f}%')}</td></tr>
                <tr><td>D/E</td><td>{clean_val(de, '{:.2f}%')}</td></tr>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)

            interpretation_items = []
            if mc is not None:
                if mc >= 1e13: mc_text = f"Market Cap Rp {mc:,.0f} tergolong sangat besar (Mega Cap)."
                elif mc >= 1e12: mc_text = f"Market Cap Rp {mc:,.0f} tergolong besar (Blue Chip)."
                elif mc >= 1e10: mc_text = f"Market Cap Rp {mc:,.0f} tergolong menengah (Mid Cap)."
                else: mc_text = f"Market Cap Rp {mc:,.0f} tergolong kecil (Small Cap)."
            else: mc_text = "Market Cap tidak tersedia."
            interpretation_items.append(f"<li><b>Market Cap:</b> {mc_text}</li>")

            if per is not None:
                if per < 10: per_text = f"PER {per:.2f}x tergolong rendah (potensi undervalue)."
                elif per < 20: per_text = f"PER {per:.2f}x moderat."
                else: per_text = f"PER {per:.2f}x tergolong tinggi (premium)."
            else: per_text = "PER tidak tersedia."
            interpretation_items.append(f"<li><b>PER:</b> {per_text}</li>")

            if pbv is not None:
                if pbv < 1: pbv_text = f"PBV {pbv:.2f}x di bawah 1 (di bawah nilai buku, bisa undervalue)."
                elif pbv < 3: pbv_text = f"PBV {pbv:.2f}x moderat."
                else: pbv_text = f"PBV {pbv:.2f}x tinggi (premium)."
            else: pbv_text = "PBV tidak tersedia."
            interpretation_items.append(f"<li><b>PBV:</b> {pbv_text}</li>")

            if roe is not None:
                roe_pct = roe * 100
                if roe_pct > 20: roe_text = f"ROE {roe_pct:.1f}% sangat baik (profitabilitas tinggi)."
                elif roe_pct > 10: roe_text = f"ROE {roe_pct:.1f}% cukup baik."
                else: roe_text = f"ROE {roe_pct:.1f}% rendah."
            else: roe_text = "ROE tidak tersedia."
            interpretation_items.append(f"<li><b>ROE:</b> {roe_text}</li>")

            if de is not None:
                if de > 1: de_text = f"D/E {de:.2f} tinggi (leverage tinggi, risiko lebih besar)."
                elif de > 0.5: de_text = f"D/E {de:.2f} moderat."
                else: de_text = f"D/E {de:.2f} rendah (konservatif)."
            else: de_text = "D/E tidak tersedia."
            interpretation_items.append(f"<li><b>D/E:</b> {de_text}</li>")

            interpretation_html = f"""
            <div style="background-color: #1e293b; border-radius: 12px; padding: 15px; margin-top: 15px; color: #cbd5e1; font-size: 14px;">
                <b style="color: #00ffcc;">📝 Interpretasi Metrik:</b>
                <ul style="margin-top: 8px; padding-left: 20px;">
                    {''.join(interpretation_items)}
                </ul>
            </div>
            """
            st.markdown(interpretation_html, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data fundamental finansial tidak tersedia.")
        st.divider()

        st.subheader("🎯 Target Pivot & Support/Resistance")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("R2", f"Rp {r2:,.0f}".replace(",", "."))
        p2.metric("R1", f"Rp {r1:,.0f}".replace(",", "."))
        p3.metric("Pivot", f"Rp {pp:,.0f}".replace(",", "."))
        p4.metric("S1", f"Rp {s1:,.0f}".replace(",", "."))
        p5.metric("S2", f"Rp {s2:,.0f}".replace(",", "."))
        st.write(f"Kondisi Breakout 20 Hari: **{breakout}**")
        st.divider()

        st.subheader("🔮 Sinyal Kuantitatif & Hasil Backtest (6 Bulan)")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Sinyal", signal)
        t2.metric("Estimasi Besok", f"Rp {est_besok:,.0f}".replace(",", "."))
        t3.metric("Entry Ideal", f"Rp {s1:,.0f} - {pp:,.0f}".replace(",", "."))
        t4.metric("Take Profit", f"Rp {r1:,.0f} (+{tp_pct:.1f}%)".replace(",", "."))
        t5.metric("Stop Loss", f"Rp {s2:,.0f} (-{sl_pct:.1f}%)".replace(",", "."))
        st.markdown("**Hasil Backtest Stateful Tracking 126 Hari:**")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Win Rate", f"{win_bt:.1%}" if trades_bt else "N/A")
        b2.metric("Profit Factor", f"{pf_bt:.2f}" if trades_bt else "N/A")
        b3.metric("Avg Return/Trade", f"{avg_bt:.2%}" if trades_bt else "N/A")
        b4.metric("Max DD Strat", f"{max_dd_bt:.2f}%" if trades_bt else "N/A")
        b5.metric("Sharpe", f"{sharpe_bt:.2f}" if trades_bt else "N/A")
        b6.metric("Total Trades", trades_bt)
        st.divider()

        st.subheader("🛡️ Manajemen Risiko Portofolio (Kelly)")
        r_c1, r_c2 = st.columns(2)
        r_c1.metric("Alokasi Maks (Kelly)", f"{kelly_adj*100:.1f}%")
        r_c2.metric("Beta IHSG", f"{beta_ihsg:.2f}x")
        st.markdown(f"**Interpretasi:** Berdasarkan Win Rate **{win_bt:.1%}**, maksimal alokasi **{kelly_adj*100:.1f}%** dari total ekuitas.")
        st.markdown(f"Max DD Historis: `{max_dd:.2f}%` | DD 30 Hari: `{max_dd_30:.2f}%`")
        st.divider()

        st.subheader("🎲 Simulasi Monte Carlo Ornstein-Uhlenbeck")
        pr1, pr2, pr3 = st.columns(3)
        pr1.metric("Prob. Naik Besok", f"{prob_bull:.1f}%")
        pr2.metric("Prob. Sentuh R1 (30H)", f"{hit_tp:.1f}%")
        pr3.metric("Prob. Sentuh S2 (30H)", f"{hit_sl:.1f}%")

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
