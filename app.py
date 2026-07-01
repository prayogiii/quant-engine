import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, t as student_t
from scipy.optimize import minimize
import warnings
import urllib.parse
import re

# ====================== FALLBACK PLOTLY ======================
PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except ImportError:
    PLOTLY_AVAILABLE = False
# ============================================================

# ====================== FALLBACK SENTIMEN ======================
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

# ====================== FALLBACK RSS ======================
RSS_AVAILABLE = True
try:
    import feedparser
except ImportError:
    RSS_AVAILABLE = False

# ====================== FALLBACK TRANSLATOR ======================
TRANSLATOR_AVAILABLE = True
try:
    from deep_translator import GoogleTranslator
except ImportError:
    TRANSLATOR_AVAILABLE = False
# =================================================================

warnings.filterwarnings("ignore")

# ==========================================
# 1. KONFIGURASI HALAMAN & ENGINE CACHING
# ==========================================
st.set_page_config(page_title="Quant Risk Engine Pro v2", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { font-size: 20px; font-weight: bold; }
    .translated { color: #8892b0; font-size: 13px; font-style: italic; }
    .source { color: #ff4b4b; font-size: 11px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

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

# ==================== UTILITIES & INDICATORS ====================
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
    "Strong Bullish 🚀": "Tren naik kuat dengan momentum tinggi. Ideal untuk swing buy agresif, waspadai overbought.",
    "Bullish 📈": "Tren naik stabil. Kondisi sehat untuk akumulasi.",
    "Panic Sell 🚨": "Penurunan tajam, sering oversold. Peluang buy-back jika reversal terkonfirmasi.",
    "Bearish 🔻": "Tren turun terkendali. Hindari buy, pertimbangkan short.",
    "Early Recovery 🔄": "Harga di atas EMA20 tapi EMA20 < EMA50. Potensi reversal bullish, perlu konfirmasi.",
    "Distribution 📉": "Harga di bawah EMA20, EMA20 > EMA50. Distribusi setelah uptrend panjang.",
    "Konsolidasi Tren ↔️": "Trending namun harga bolak-balik di EMA. Tunggu penembusan.",
    "Bullish Accumulation 🏗️": "Sideways dengan harga > EMA. Akumulasi, potensi breakout.",
    "Bearish Accumulation 🧊": "Sideways di bawah EMA. Distribusi pelan, waspadai breakdown.",
    "Sideways Bias Naik ↗️": "Sideways cenderung naik, potensi bullish belum kuat.",
    "Sideways Bias Turun ↘️": "Sideways cenderung turun, potensi bearish.",
    "Sideways Choppy 🌊": "Sideways volatilitas tinggi. Hindari entry.",
    "Sideways Calm 😴": "Sideways sepi, menjelang breakout.",
    "Sideways Normal ↔️": "Sideways moderat, tunggu katalis."
}

# ==========================================
# 2. USER INTERFACE INPUT (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("⚙️ Kontrol Engine")
    ticker_raw = st.text_input("Kode Saham IHSG:", value="BBRI").upper().strip()
    if ticker_raw and not ticker_raw.endswith(".JK"):
        ticker_input = f"{ticker_raw}.JK"
    else:
        ticker_input = ticker_raw
        
    jalankan = st.button("JALANKAN QUANT ENGINE", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("Engine analisis volatilitas kuantitatif makro dan mikro.")

st.title("📊 Quant & Risk Engine Pro")

if jalankan:
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

        # PRE-COMPUTE INDICATORS
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['ADX'] = compute_adx_series(df)
        df['Mom3D'] = df['Close'].pct_change(3) * 100
        df['Mom5D'] = df['Close'].pct_change(5) * 100
        df['Mom10D'] = df['Close'].pct_change(10) * 100
        df['ZScore'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        df['Vol_MA20'] = df['Volume'].rolling(20).mean() if 'Volume' in df.columns else 0

        try: ticker_info = yf.Ticker(ticker_input).info
        except: ticker_info = {}

        # BERITA & SENTIMEN
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
            if len(unique_news) >= 5: break
            
        avg_sentiment = analyze_sentiment_weighted(unique_news, translator_en)
        headlines = [n['title'] for n in unique_news]
        sources = [n['source'] for n in unique_news]
        translated = []
        for n in unique_news:
            if TRANSLATOR_AVAILABLE and translator_id:
                try: translated.append(translator_id.translate(n['title']))
                except: translated.append("")
            else: translated.append("")
            
        sentimen_status = "Positif 🟢" if avg_sentiment >= 0.05 else ("Negatif 🔴" if avg_sentiment <= -0.05 else "Netral ⚪")

        # ESTIMASI AMBANG BATAS HISTORIS
        split_idx = max(126, len(df) - 126)
        df_thresh = df.iloc[:split_idx]
        returns_thresh = df_thresh['Close'].pct_change().dropna()
        
        adx_threshold = np.percentile(df_thresh['ADX'].dropna(), 75) if not df_thresh['ADX'].dropna().empty else 20
        z_oversold_th = -1.5  
        mom_median_th = np.percentile(df_thresh['Mom5D'].dropna(), 50) if not df_thresh['Mom5D'].dropna().empty else 0.0
        
        # ESTIMASI DISTRIBUSI STUDENT-T
        def t_loglike(p, d):
            if p[0] <= 2 or p[2] <= 0: return np.inf
            return -np.sum(student_t.logpdf(d, p[0], p[1], p[2]))
            
        res_thresh = minimize(t_loglike, [5, returns_thresh.mean(), returns_thresh.std()],
                             bounds=[(2.1, 100), (-0.1, 0.1), (1e-6, None)], args=(returns_thresh,), method='L-BFGS-B')
        df_est, t_loc, t_scale = res_thresh.x if res_thresh.success else (5, returns_thresh.mean(), returns_thresh.std())

        # DETEKSI REGIME MARKET LENGKAP
        def get_regime_row(row):
            h = row['Close']
            ema20s = row['EMA20']
            ema50s = row['EMA50']
            adxs = row['ADX']
            zs = row['ZScore']
            m5 = row['Mom5D']
            
            if adxs > adx_threshold:
                if h > ema20s and ema20s > ema50s:
                    return ("Strong Bullish 🚀", "RISK-ON 🔥") if (m5 > mom_median_th or zs > z_oversold_th) else ("Bullish 📈", "RISK-ON 🔥")
                elif h < ema20s and ema20s < ema50s:
                    return ("Panic Sell 🚨", "RISK-OFF 🛑") if (m5 < mom_median_th or zs < z_oversold_th) else ("Bearish 🔻", "RISK-OFF 🛑")
                elif h > ema20s and ema20s < ema50s: return ("Early Recovery 🔄", "TRANSISI ⚠️")
                elif h < ema20s and ema20s > ema50s: return ("Distribution 📉", "TRANSISI ⚠️")
                else: return ("Konsolidasi Tren ↔️", "NEUTRAL ⚖️")
            else:
                if h > ema20s and ema20s > ema50s: return ("Bullish Accumulation 🏗️", "NEUTRAL ⚖️")
                elif h < ema20s and ema20s < ema50s: return ("Bearish Accumulation 🧊", "NEUTRAL ⚖️")
                elif h > ema20s and ema20s < ema50s: return ("Sideways Bias Naik ↗️", "NEUTRAL ⚖️")
                elif h < ema20s and ema20s > ema50s: return ("Sideways Bias Turun ↘️", "NEUTRAL ⚖️")
                else: return ("Sideways Normal ↔️", "NEUTRAL ⚖️")

        regime, ihsg_cond = get_regime_row(df.iloc[-1])
        adx = df['ADX'].iloc[-1]

        # PIVOT CALCULATION
        hi, lo = float(df['High'].iloc[-1]), float(df['Low'].iloc[-1])
        pp = (hi + lo + harga_terakhir) / 3
        r1, s1 = 2*pp - lo, 2*pp - hi
        r2, s2 = pp + (hi - lo), pp - (hi - lo)
        res20 = float(df['High'].iloc[-21:-1].max())
        breakout = "YES (🔥)" if harga_terakhir > res20 else "NO"

        # GENERATE SIGNALS
        def generate_signals_vectorized(dataframe, mom_th):
            score = pd.Series(0, index=dataframe.index)
            is_uptrend = (dataframe['Close'] > dataframe['EMA20']) & (dataframe['EMA20'] > dataframe['EMA50'])
            score += is_uptrend.astype(int) * 2
            score += (dataframe['Mom5D'] > mom_th).astype(int)
            if 'Volume' in dataframe.columns:
                score += (dataframe['Volume'] > dataframe['Vol_MA20']).astype(int)
                
            sig_series = pd.Series("🚨 AVOID", index=dataframe.index)
            sig_series[score == 1] = "⏸️ HOLD / WAIT"
            sig_series[score >= 2] = "⚡ BUY (TACTICAL)"
            sig_series[score >= 3] = "🔥 STRONG BUY"
            
            is_choppy = (dataframe['ADX'] < 20)
            sig_series[is_choppy & (sig_series.str.contains("BUY"))] = "⏸️ HOLD / WAIT"
            
            is_oversold = (dataframe['ZScore'] < -1.5) & (dataframe['Close'] < dataframe['EMA20'])
            sig_series[is_oversold] = "⚡ BUY (TACTICAL)"
            return sig_series

        df['Signal'] = generate_signals_vectorized(df, mom_median_th)
        signal = df['Signal'].iloc[-1]

        # BACKTEST STATEFUL TRACKING
        backtest_periods = 126
        df_back = df.iloc[-backtest_periods:].copy()
        trades = []
        daily_returns = []
        in_position = False
        entry_price = 0.0
        
        for i in range(len(df_back)):
            current_sig = df_back['Signal'].iloc[i]
            current_close = float(df_back['Close'].iloc[i])
            prev_close = float(df_back['Close'].iloc[i-1]) if i > 0 else current_close
            if in_position:
                asset_return = (current_close - prev_close) / prev_close if prev_close > 0 else 0.0
                daily_returns.append(asset_return)
                if "AVOID" in current_sig or i == len(df_back) - 1:
                    exit_price = current_close
                    trade_return = (exit_price - entry_price) / entry_price
                    trades.append(trade_return)
                    in_position = False
            else:
                daily_returns.append(0.0)
                if "BUY" in current_sig:
                    in_position = True
                    entry_price = current_close

        if trades:
            win_bt = sum(1 for r in trades if r > 0) / len(trades)
            loss_trades = [r for r in trades if r < 0]
            profit_trades = [r for r in trades if r > 0]
            pf_bt = abs(sum(profit_trades) / sum(loss_trades)) if loss_trades else np.inf
            avg_bt = np.mean(trades)
            equity = np.cumprod([1 + r for r in trades])
            max_dd_bt = float(np.min(equity / np.maximum.accumulate(equity) - 1) * 100) if len(equity) > 0 else 0
            daily_returns_arr = np.array(daily_returns)
            sharpe_bt = np.mean(daily_returns_arr) / np.std(daily_returns_arr) * np.sqrt(252) if np.std(daily_returns_arr) > 0 else 0
            trades_bt = len(trades)
        else:
            win_bt, pf_bt, avg_bt, max_dd_bt, sharpe_bt, trades_bt = 0, 0, 0, 0, 0, 0

        roll_max_th = df_thresh['Close'].cummax()
        drawdown_th = (df_thresh['Close'] - roll_max_th) / roll_max_th
        max_dd = float(drawdown_th.min() * 100)
        
        ret_skew = float(skew(returns_thresh))
        ret_kurt = float(kurtosis(returns_thresh, fisher=True))
        kurt_penalty = 0.5 if ret_kurt > 3 else 1.0
        
        if trades_bt >= 2:
            win_r = win_bt
            avg_g = np.mean(profit_trades) if len(profit_trades) > 0 else 0.01
            avg_l = abs(np.mean(loss_trades)) if len(loss_trades) > 0 else 0.01
            wl = avg_g / avg_l if avg_l > 0 else 1
            kelly_raw = win_r - (1 - win_r) / wl
        else:
            win_r = len(returns_thresh[returns_thresh > 0]) / len(returns_thresh)
            avg_g = returns_thresh[returns_thresh > 0].mean() if win_r > 0 else 0.01
            avg_l = abs(returns_thresh[returns_thresh < 0].mean()) if len(returns_thresh[returns_thresh < 0]) > 0 else 0.01
            wl = avg_g / avg_l if avg_l > 0 else 1
            kelly_raw = win_r - (1 - win_r) / wl
        kelly_adj = min(0.25, max(0.0, kelly_raw * 0.3 * (0.5 if ret_skew < -0.5 else 1.0) * kurt_penalty))

        # MONTE CARLO PROYEKSI MAJU (OU PROCESS)
        n_sim, n_days = 2000, 30
        latest_vol_daily = np.sqrt(df['Close'].pct_change().ewm(alpha=0.06).var().iloc[-1])
        scale_corrected = latest_vol_daily / np.sqrt(df_est / (df_est - 2)) if df_est > 2 else latest_vol_daily
        theta_ou = estimate_theta_ou(df['Close'])
        locked_log_mean20 = np.log(df['Close']).tail(20).mean()
        paths = np.zeros((n_days, n_sim))
        current_log = np.expand_dims(np.log(harga_terakhir) * np.ones(n_sim), axis=0)
        
        for day in range(n_days):
            innovations = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=n_sim)
            next_log = current_log[-1, :] + theta_ou * (locked_log_mean20 - current_log[-1, :]) + innovations
            current_log = np.vstack([current_log, next_log])
            paths[day, :] = np.exp(next_log)

        mu_ou = theta_ou * (locked_log_mean20 - np.log(harga_terakhir))
        est_besok = float(np.exp(np.log(harga_terakhir) + mu_ou))
        sim_h1 = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=2000)
        prices_besok = harga_terakhir * np.exp(mu_ou + sim_h1)
        low_est, up_est = float(np.percentile(prices_besok, 25)), float(np.percentile(prices_besok, 75))
        
        hit_tp = (np.any(paths >= r1, axis=0).sum() / n_sim) * 100
        hit_sl = (np.any(paths <= s2, axis=0).sum() / n_sim) * 100
        prob_bull = ((mu_ou + sim_h1 > 0).sum() / 2000) * 100

        tp_pct = ((r1 - harga_terakhir) / harga_terakhir) * 100 if harga_terakhir > 0 else 0
        sl_pct = ((harga_terakhir - s2) / harga_terakhir) * 100 if harga_terakhir > 0 else 0
        rrr = tp_pct / sl_pct if sl_pct > 0 else 0

        # Dynamic Entry Price Area Calculation
        if "STRONG BUY" in signal:
            harga_entry_ideal = f"Rp {harga_terakhir:,.0f} (Market Order)".replace(",", ".")
        elif "BUY" in signal:
            harga_entry_ideal = f"Rp {s1:,.0f} - {pp:,.0f}".replace(",", ".")
        else:
            harga_entry_ideal = "N/A (Tunggu Momentum)"

        # ==========================================
        # 3. PENYUSUNAN UI EKSEKUTIF UTAMA (TOP AREA)
        # ==========================================
        st.subheader(f"📊 Live Report: {ticker_input}")
        
        # Grid Atas yang Diperbarui: Sinyal Entry & Harga Entry Eksplisit
        top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns(5)
        with top_c1:
            st.metric("Harga Terakhir", f"Rp {harga_terakhir:,.0f}".replace(",", "."))
        with top_c2:
            st.metric("Sinyal Entry", signal)  # <-- SEKARANG MENGGUNAKAN NAMA "SINYAL ENTRY"
        with top_c3:
            st.metric("Area Entry Ideal", harga_entry_ideal)  # <-- METRIK BARU HARGA ENTRY EKSPLISIT
        with top_c4:
            st.metric("Porsi Portofolio (Kelly)", f"{kelly_adj*100:.1f}%")
        with top_c5:
            st.metric("Risk-Reward Ratio", f"{rrr:.2f}")

        # LOGIC WARNA CARD REKOMENDASI EKSEKUTIF
        if rrr < 1.0 and ("BUY" in signal):
            action_icon = "⚠️"
            action_title = "WAIT & SEE (Tunda Entry)"
            action_text = f"Sinyal Kuantitatif merekomendasikan {signal}, namun nilai **Risk-to-Reward Ratio BURUK ({rrr:.2f})**. Jarak batas Stop Loss (-{sl_pct:.1f}%) saat ini lebih lebar dibandingkan target Take Profit (+{tp_pct:.1f}%). Jangan mengejar harga atas; tunggu koreksi sehat mendekati area Support 1 (Rp {s1:,.0f}) untuk memperkecil risiko modal."
        elif "STRONG BUY" in signal:
            action_icon = "🟢"
            action_title = "AGGRESSIVE BUY / ACCUMULATE"
            action_text = f"Terdeteksi konfirmasi tren kuat berlandaskan akumulasi volume transaksi yang signifikan. Disarankan melakukan akumulasi pembelian secara langsung pada level harga pasar saat ini dengan kapasitas maksimal **{kelly_adj*100:.1f}%** dari modal portofolio Anda. Amankan pembatas risiko ketat pada batas Stop Loss S2 di level Rp {s2:,.0f} (-{sl_pct:.1f}%)."
        elif "BUY" in signal:
            action_icon = "🟡"
            action_title = "BUY ON WEAKNESS (BoW)"
            action_text = f"Struktur tren valid dengan dukungan nilai Risk-to-Reward Ratio yang memadai. Anda dapat menempatkan antrean order beli di wilayah ideal antara **{harga_entry_ideal}**. Disiplin untuk melakukan proteksi modal atau cut-loss tanpa kompromi apabila harga merosot di bawah jangkar pertahanan Rp {s2:,.0f} (-{sl_pct:.1f}%)."
        elif "HOLD" in signal:
            action_icon = "🔵"
            action_title = "HOLD POSITION"
            action_text = "Instrumen berada dalam fase konsolidasi transisi atau pergerakan sideways yang stabil. Diimbau untuk mempertahankan sisa kepemilikan saham yang ada tanpa melakukan penambahan posisi modal baru (*no average up/down*). Pantau kekuatan area support terbawah di level Rp {s2:,.0f}."
        else:
            action_icon = "🔴"
            action_title = "AVOID / LIQUIDATE"
            action_text = f"Engine mendeteksi probabilitas penurunan yang tinggi akibat pengaruh fase distribusi pasar (**{regime}**). Langkah taktis terbaik saat ini adalah mengamankan modal Anda ke dalam instrumen kas, menjauhi emiten ini untuk sementara waktu, atau merealisasikan aksi *Take Profit/Cut Loss* dari kepemilikan Anda saat ini."

        with st.container(border=True):
            st.markdown(f"### {action_icon} Panduan Strategi Eksekutif: **{action_title}**")
            st.write(action_text)

        st.markdown("### 🔍 Detail Analitika Komprehensif")
        
        # TAB LAYOUT
        tab_chart, tab_tech, tab_funda, tab_quant = st.tabs([
            "📈 Grafik & Sinyal", 
            "🎯 Pivot & Struktur Tren", 
            "🏛️ Data Fundamental", 
            "🎲 Sentimen & Proyeksi Kuantitatif"
        ])

        # --- TAB 1: GRAFIK & SINYAL HISTORIS ---
        with tab_chart:
            if PLOTLY_AVAILABLE:
                st.markdown("#### Histori Pergerakan Harga & Titik Eksekusi")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Harga Close', line=dict(color='#00ffcc', width=2)))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20 (Tren Pendek)', line=dict(color='#f59e0b', dash='dot')))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50 (Tren Menengah)', line=dict(color='#ef4444', dash='dot')))
                
                buy_signals = df_back[df_back['Signal'].str.contains("BUY")]
                fig.add_trace(go.Scatter(
                    x=buy_signals.index, y=buy_signals['Close'],
                    mode='markers', name='Sinyal Transaksi Buy',
                    marker=dict(symbol='triangle-up', size=11, color='#10b981', line=dict(width=1, color='white'))
                ))
                
                for level, label, col in [(r1, 'R1', 'orange'), (s1, 'S1', 'red'), (pp, 'PP', 'gray')]:
                    fig.add_hline(y=level, line_dash="dash", line_color=col, annotation_text=label, annotation_position="right")
                    
                fig.update_layout(template="plotly_dark", height=480, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Visualisasi grafik interaktif tidak tersedia.")

        # --- TAB 2: PIVOT & TEKNIKAL TREN ---
        with tab_tech:
            st.markdown("#### Pemetaan Struktur Level Klasik & Kondisi Tren")
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                with st.container(border=True):
                    st.markdown("**Matriks Batas Pivot Saham (IDR):**")
                    st.write(f"🔹 **Resistance 2 (R2):** Rp {r2:,.0f}".replace(",", "."))
                    st.write(f"🔹 **Resistance 1 (R1):** Rp {r1:,.0f}".replace(",", "."))
                    st.write(f"📍 **Pivot Point (PP):** Rp {pp:,.0f}".replace(",", "."))
                    st.write(f"🔸 **Support 1 (S1):** Rp {s1:,.0f}".replace(",", "."))
                    st.write(f"🔸 **Support 2 (S2):** Rp {s2:,.0f}".replace(",", "."))
                    st.write(f"🔥 **Kondisi Breakout 20 Hari:** {breakout}")
            with c_p2:
                with st.container(border=True):
                    st.markdown("**Kondisi Karakteristik Regime:**")
                    st.write(f"🧬 **Regime Saham:** {regime}")
                    st.write(f"🌏 **Kondisi Makro IHSG:** {ihsg_cond}")
                    st.write(f"📊 **ADX Adaptif:** {adx:.1f} *(Ambang Batas: {adx_threshold:.1f})*")
                    st.write(f"ℹ️ **Insight:** {REGIME_INFO.get(regime, 'Kondisi tidak terdefinisi.')}")

        # --- TAB 3: DATA FUNDAMENTAL ---
        with tab_funda:
            st.markdown("#### Laporan Rasio Fundamental Keuangan (Yahoo Finance)")
            if ticker_info:
                def clean_val(val, fmt="{:.2f}"): return "N/A" if val is None else fmt.format(val)
                mc = ticker_info.get('marketCap')
                per = ticker_info.get('trailingPE') or ticker_info.get('forwardPE')
                pbv = ticker_info.get('priceToBook')
                roe = ticker_info.get('returnOnEquity')
                de = ticker_info.get('debtToEquity')
                
                f_c1, f_c2 = st.columns(2)
                with f_c1:
                    st.metric("Total Market Cap", clean_val(mc, "Rp {:,.0f} IDR").replace(",", "."))
                    st.metric("Price to Earnings Ratio (PER)", clean_val(per, "{:.2f}x"))
                    st.metric("Price to Book Value (PBV)", clean_val(pbv, "{:.2f}x"))
                with f_c2:
                    st.metric("Return On Equity (ROE)", clean_val(roe * 100 if roe else None, "{:.1f}%"))
                    st.metric("Debt to Equity Ratio (D/E)", clean_val(de, "{:.2f}%"))
            else:
                st.warning("⚠️ Data fundamental finansial emiten tidak berhasil ditarik.")

        # --- TAB 4: SENTIMEN & PROYEKSI KUANTITATIF ---
        with tab_quant:
            st.markdown("#### Pengujian Algoritma & Model Prediksi Ke Depan")
            st.markdown("**🎯 Kinerja Histori Model Kuantitatif (Backtest 126 Hari Terakhir):**")
            b_c1, b_c2, b_c3 = st.columns(3)
            with b_c1:
                st.metric("Win Rate Strategi", f"{win_bt:.1%}" if trades_bt else "N/A")
                st.metric("Profit Factor", f"{pf_bt:.2f}" if trades_bt and pf_bt != np.inf else "N/A")
            with b_c2:
                st.metric("Rata-rata Return/Trade", f"{avg_bt:.2%}" if trades_bt else "N/A")
                st.metric("Max Drawdown Strategi", f"{max_dd_bt:.2f}%" if trades_bt else "N/A")
            with b_c3:
                st.metric("Sharpe Ratio Tahunan", f"{sharpe_bt:.2f}" if trades_bt else "N/A")
                st.metric("Total Eksekusi Trades", f"{trades_bt}")
            
            st.markdown("---")
            st.markdown("**🎲 Hasil Proyeksi Simulasi Monte Carlo (Ornstein-Uhlenbeck 30 Hari Ke Depan):**")
            p_c1, p_c2 = st.columns(2)
            with p_c1:
                with st.container(border=True):
                    st.write(f"📈 **Estimasi Harga Esok Hari:** Rp {est_besok:,.0f}".replace(",", "."))
                    st.write(f"🔮 **Rentang Probabilitas (50%):** Rp {low_est:,.0f} - {up_est:,.0f}".replace(",", "."))
                    st.write(f"🟢 **Probabilitas Hari Esok Ditutup Naik:** {prob_bull:.1f}%")
            with p_c2:
                with st.container(border=True):
                    st.write(f"🎯 **Probabilitas Menyentuh Target R1:** {hit_tp:.1f}%")
                    st.write(f"🛑 **Probabilitas Jebol Batas Risiko S2:** {hit_sl:.1f}%")
                    st.write(f"📉 **Statistik Max Drawdown Historis:** {max_dd:.2f}%")
            
            st.markdown("---")
            st.markdown("#### 📰 Analisis Sentimen Berita Terbobot")
            s_col1, s_col2 = st.columns([1, 2])
            with s_col1:
                st.metric("Skor Sentimen Berita", f"{avg_sentiment:.2f}", sentimen_status)
            with s_col2:
                st.markdown("**Arus Berita Terkini:**")
                for idx, h in enumerate(headlines):
                    src = sources[idx] if idx < len(sources) else ""
                    t = translated[idx] if idx < len(translated) else ""
                    st.markdown(f"{idx+1}. **{h}** <span class='source'>({src})</span>", unsafe_allow_html=True)
                    if t and t != h: st.markdown(f"<span class='translated'>🇮🇩 {t}</span>", unsafe_allow_html=True)
