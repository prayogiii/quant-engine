import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, t as student_t
from scipy.optimize import minimize
import warnings
import urllib.parse
import re
from datetime import datetime, timedelta

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
# 1. KONFIGURASI HALAMAN & THEMA GLOBAL
# ==========================================
st.set_page_config(
    page_title="QuantRisk Pro v3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS yang lebih modern
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0a0e1a 0%, #121a2f 100%);
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }

    /* Card Modern */
    .neo-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .neo-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px -10px rgba(0,255,204,0.15);
    }

    /* Metric styling */
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #00ffcc, #00b8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 5px;
    }

    /* Action card */
    .action-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 24px;
        margin: 10px 0;
        border-left: 5px solid;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.4);
    }

    /* Section title */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
        color: #f1f5f9;
    }

    /* News item */
    .news-item {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid #334155;
    }

    .stButton>button {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #00ffcc;
        border: 1px solid #334155;
        border-radius: 12px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        border-color: #00ffcc;
        box-shadow: 0 0 15px rgba(0,255,204,0.3);
        transform: scale(1.02);
    }

    .stTextInput>div>div>input {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #334155;
        border-radius: 12px;
        color: white;
        padding: 12px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 16px;
        padding: 5px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 600;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: #00ffcc !important;
        box-shadow: 0 0 15px rgba(0,255,204,0.2);
    }

    hr {
        border-color: #334155;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=80)
    st.markdown("## 📊 QuantRisk Pro")
    st.markdown("Engine analisis kuantitatif saham Indonesia terintegrasi.")
    
    st.markdown("---")
    
    # Ticker input
    ticker_raw = st.text_input(
        "🔍 Kode Saham",
        value="BBRI",
        placeholder="Contoh: BBRI, TLKM, BMRI",
        help="Masukkan kode saham IHSG (tanpa .JK)"
    ).upper().strip()
    
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
    st.markdown("""
    <div style="font-size:11px; color:#64748b;">
    ⚠️ Data bersumber dari Yahoo Finance. Hasil analisis bersifat informatif dan bukan rekomendasi investasi.
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================
st.title("🔬 Quant & Risk Intelligence Engine")
st.markdown("Analisis menyeluruh dari harga historis, sentimen berita, fundamental, hingga simulasi probabilistik.")

if not run_btn:
    st.info("👈 Masukkan kode saham di sidebar lalu klik **ANALISIS** untuk memulai.")
    st.stop()

if not ticker_input:
    st.warning("⚠️ Kode saham tidak boleh kosong!")
    st.stop()

with st.spinner("🔄 Mengunduh data dan memproses analitika kuantitatif..."):
    # ==================== DATA LOADING ====================
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

    df = load_stock_data(ticker_input)
    if df.empty:
        st.error("❌ Data tidak ditemukan untuk ticker tersebut. Periksa kode atau coba lagi nanti.")
        st.stop()
    
    harga_terakhir = float(df['Close'].iloc[-1])
    returns = df['Close'].pct_change().dropna()
    if len(returns) < 50:
        st.error("❌ Data historis kurang untuk analisa kuantitatif.")
        st.stop()

    # ==================== INDIKATOR ====================
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

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ADX'] = compute_adx_series(df)
    df['Mom3D'] = df['Close'].pct_change(3) * 100
    df['Mom5D'] = df['Close'].pct_change(5) * 100
    df['ZScore'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean() if 'Volume' in df.columns else 0

    # ==================== FUNDAMENTAL ====================
    try:
        ticker_info = yf.Ticker(ticker_input).info
    except:
        ticker_info = {}

    # ==================== BERITA & SENTIMEN ====================
    # (fungsi-fungsi utilitas sama persis, tidak diubah)
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

    # ==================== THRESHOLD & DISTRIBUSI ====================
    split_idx = max(126, len(df) - 126)
    df_thresh = df.iloc[:split_idx]
    returns_thresh = df_thresh['Close'].pct_change().dropna()
    
    adx_threshold = np.percentile(df_thresh['ADX'].dropna(), 75) if not df_thresh['ADX'].dropna().empty else 20
    z_oversold_th = -1.5
    mom_median_th = np.percentile(df_thresh['Mom5D'].dropna(), 50) if not df_thresh['Mom5D'].dropna().empty else 0.0
    
    def t_loglike(p, d):
        if p[0] <= 2 or p[2] <= 0: return np.inf
        return -np.sum(student_t.logpdf(d, p[0], p[1], p[2]))
        
    res_thresh = minimize(t_loglike, [5, returns_thresh.mean(), returns_thresh.std()],
                          bounds=[(2.1, 100), (-0.1, 0.1), (1e-6, None)], args=(returns_thresh,), method='L-BFGS-B')
    df_est, t_loc, t_scale = res_thresh.x if res_thresh.success else (5, returns_thresh.mean(), returns_thresh.std())

    # Regime detection
    def get_regime_row(row):
        h, ema20s, ema50s, adxs, zs, m5 = row['Close'], row['EMA20'], row['EMA50'], row['ADX'], row['ZScore'], row['Mom5D']
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

    # Beta
    try:
        ihsg = load_ihsg_data()
        ihsg_ret = ihsg['Close'].pct_change().dropna()
        common = returns.index.intersection(ihsg_ret.index)
        if len(common) > 20:
            cov = np.cov(returns.loc[common], ihsg_ret.loc[common])
            beta_ihsg = cov[0,1]/cov[1,1] if cov[1,1] > 0 else 1.0
        else: beta_ihsg = 1.0
    except: beta_ihsg = 1.0

    # Pivot
    hi, lo = float(df['High'].iloc[-1]), float(df['Low'].iloc[-1])
    pp = (hi + lo + harga_terakhir) / 3
    r1, s1 = 2*pp - lo, 2*pp - hi
    r2, s2 = pp + (hi - lo), pp - (hi - lo)
    res20 = float(df['High'].iloc[-21:-1].max())
    breakout = "YES 🔥" if harga_terakhir > res20 else "NO"

    # Sinyal
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

    # Backtest
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
        win_bt = pf_bt = avg_bt = max_dd_bt = sharpe_bt = trades_bt = 0

    # Kelly
    roll_max_th = df_thresh['Close'].cummax()
    drawdown_th = (df_thresh['Close'] - roll_max_th) / roll_max_th
    max_dd = float(drawdown_th.min() * 100)
    max_dd_30 = float(drawdown_th.tail(30).min() * 100) if len(drawdown_th) >= 30 else max_dd
    
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
    
    ret_skew = float(skew(returns_thresh))
    ret_kurt = float(kurtosis(returns_thresh, fisher=True))
    kurt_penalty = 0.5 if ret_kurt > 3 else 1.0
    kelly_adj = min(0.25, max(0.0, kelly_raw * 0.3 * (0.5 if ret_skew < -0.5 else 1.0) * kurt_penalty))

    # Monte Carlo OU
    n_sim, n_days = 2000, 30
    latest_vol_daily = np.sqrt(df['Close'].pct_change().ewm(alpha=0.06).var().iloc[-1])
    scale_corrected = latest_vol_daily / np.sqrt(df_est / (df_est - 2)) if df_est > 2 else latest_vol_daily
    
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

    # Persentase TP/SL
    tp_pct = ((r1 - harga_terakhir) / harga_terakhir) * 100 if harga_terakhir > 0 else 0
    sl_pct = ((harga_terakhir - s2) / harga_terakhir) * 100 if harga_terakhir > 0 else 0
    rrr = tp_pct / sl_pct if sl_pct > 0 else 0
    rrr_status = "Ideal (≥ 1.5) 🟢" if rrr >= 1.5 else ("Cukup (1.0 - 1.5) 🟡" if rrr >= 1.0 else "Buruk (< 1.0) 🔴")

    # ==================== UI DENGAN TABS ====================
    st.success(f"✅ Analisis selesai untuk **{ticker_input}** | Harga Terakhir: **Rp {harga_terakhir:,.0f}**".replace(",", "."))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Ringkasan", "📰 Berita & Sentimen", "📈 Teknikal & Regime",
        "🏢 Fundamental", "⚖️ Backtest & Risiko", "🎲 Proyeksi Monte Carlo"
    ])

    # ---------- TAB 1: RINGKASAN EKSEKUTIF ----------
    with tab1:
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown("<div class='section-title'>🚦 Sinyal & Rekomendasi</div>", unsafe_allow_html=True)
            signal_color = {
                "🔥 STRONG BUY": "#10b981",
                "⚡ BUY (TACTICAL)": "#f59e0b",
                "⏸️ HOLD / WAIT": "#3b82f6",
                "🚨 AVOID": "#ef4444"
            }.get(signal, "#ef4444")
            st.markdown(f"""
            <div class="neo-card" style="border-left: 5px solid {signal_color};">
                <div style="font-size: 36px; font-weight: 700; color: {signal_color};">{signal}</div>
                <div style="margin-top: 10px; color: #cbd5e1;">
                    📍 Rezim: <b>{regime}</b> | Makro: <b>{ihsg_cond}</b><br>
                    🎯 TP: <b>Rp {r1:,.0f} (+{tp_pct:.1f}%)</b> | 🛑 SL: <b>Rp {s2:,.0f} (-{sl_pct:.1f}%)</b><br>
                    ⚖️ RRR: <b>1:{rrr:.2f} ({rrr_status})</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Action card sesuai sinyal
            if rrr < 1.0 and "BUY" in signal:
                action_color, action_icon = "#ef4444", "⚠️"
                action_text = f"Tunda entry. RRR buruk ({rrr:.2f}), tunggu koreksi ke Support 1."
            elif "STRONG BUY" in signal:
                action_color, action_icon = "#10b981", "🟢"
                action_text = f"Aggressive buy, alokasi maks {kelly_adj*100:.1f}% portofolio, SL di Rp {s2:,.0f}."
            elif "BUY" in signal:
                action_color, action_icon = "#f59e0b", "🟡"
                action_text = f"Buy on Weakness, entry ideal Rp {s1:,.0f}-{pp:,.0f}, SL Rp {s2:,.0f}."
            elif "HOLD" in signal:
                action_color, action_icon = "#3b82f6", "🔵"
                action_text = "Tahan posisi, jangan tambah modal, pantau batas SL."
            else:
                action_color, action_icon = "#ef4444", "🔴"
                action_text = "Hindari/ Likuidasi, amankan modal."

            st.markdown(f"""
            <div class="action-card" style="border-left-color: {action_color};">
                <div style="font-size: 18px; font-weight: 600; color: white;">{action_icon} Panduan Eksekusi</div>
                <p style="color: #cbd5e1; margin-top: 10px;">{action_text}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='section-title'>📊 Profil Risiko</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="neo-card">
                <div class="metric-label">Alokasi Maks (Kelly)</div>
                <div class="metric-value">{kelly_adj*100:.1f}%</div>
                <div class="metric-label" style="margin-top: 15px;">Beta IHSG</div>
                <div class="metric-value">{beta_ihsg:.2f}x</div>
                <div class="metric-label" style="margin-top: 15px;">Max DD Historis</div>
                <div class="metric-value">{max_dd:.2f}%</div>
                <div class="metric-label" style="margin-top: 15px;">ADX</div>
                <div class="metric-value">{adx:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---------- TAB 2: BERITA & SENTIMEN ----------
    with tab2:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
            st.metric("Skor Sentimen", f"{avg_sentiment:.2f}", sentimen_status)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='section-title'>📰 5 Berita Teratas</div>", unsafe_allow_html=True)
            for i, h in enumerate(headlines):
                src = sources[i] if i < len(sources) else ""
                t = translated[i] if i < len(translated) else ""
                st.markdown(f"""
                <div class="news-item">
                    <b>{i+1}. {h}</b> <span style="color:#64748b; font-size:12px;">({src})</span>
                    {f'<br><span style="color:#94a3b8; font-size:13px;">🇮🇩 {t}</span>' if t and t != h else ''}
                </div>
                """, unsafe_allow_html=True)

    # ---------- TAB 3: TEKNIKAL & REGIME ----------
    with tab3:
        st.markdown("<div class='section-title'>🧬 Regime Pasar</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rezim", regime)
        col2.metric("Kondisi Makro", ihsg_cond)
        col3.metric("ADX", f"{adx:.1f} (Thr: {adx_threshold:.1f})")
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
            "Sideways Normal ↔️": "Sideways moderat, tunggu katalis."
        }
        st.info(REGIME_INFO.get(regime, ""))

        st.markdown("<div class='section-title' style='margin-top:20px;'>🎯 Pivot Point & Level Kunci</div>", unsafe_allow_html=True)
        pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
        pcol1.metric("R2", f"Rp {r2:,.0f}".replace(",", "."))
        pcol2.metric("R1", f"Rp {r1:,.0f}".replace(",", "."))
        pcol3.metric("Pivot", f"Rp {pp:,.0f}".replace(",", "."))
        pcol4.metric("S1", f"Rp {s1:,.0f}".replace(",", "."))
        pcol5.metric("S2", f"Rp {s2:,.0f}".replace(",", "."))
        st.caption(f"Breakout 20-hari: **{breakout}**")

        # Chart
        if PLOTLY_AVAILABLE:
            st.markdown("<div class='section-title' style='margin-top:20px;'>📈 Chart Harga & Sinyal</div>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#00ffcc', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', line=dict(color='#f59e0b', dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50', line=dict(color='#ef4444', dash='dot')))
            buy_signals = df_back[df_back['Signal'].str.contains("BUY")]
            fig.add_trace(go.Scatter(
                x=buy_signals.index, y=buy_signals['Close'],
                mode='markers', name='Buy Signal',
                marker=dict(symbol='triangle-up', size=10, color='#10b981', line=dict(width=1, color='white'))
            ))
            for level, label, col in [(r1, 'R1', 'orange'), (s1, 'S1', 'red'), (pp, 'PP', 'gray')]:
                fig.add_hline(y=level, line_dash="dash", line_color=col, annotation_text=label, annotation_position="right")
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 4: FUNDAMENTAL ----------
    with tab4:
        st.markdown("<div class='section-title'>🏢 Metrik Fundamental</div>", unsafe_allow_html=True)
        if ticker_info:
            def clean_val(val, fmt="{:.2f}"):
                return "N/A" if val is None else fmt.format(val)
            mc = ticker_info.get('marketCap')
            per = ticker_info.get('trailingPE') or ticker_info.get('forwardPE')
            pbv = ticker_info.get('priceToBook')
            roe = ticker_info.get('returnOnEquity')
            de = ticker_info.get('debtToEquity')
            st.markdown(f"""
            <table style="width:100%; border-collapse: collapse; color: #cbd5e1;">
                <tr><td style="padding:8px; border-bottom:1px solid #334155; width:200px;">Market Cap</td><td style="padding:8px; border-bottom:1px solid #334155;">{clean_val(mc, '{:,.0f} IDR')}</td></tr>
                <tr><td style="padding:8px; border-bottom:1px solid #334155;">PER</td><td style="padding:8px; border-bottom:1px solid #334155;">{clean_val(per, '{:.2f}x')}</td></tr>
                <tr><td style="padding:8px; border-bottom:1px solid #334155;">PBV</td><td style="padding:8px; border-bottom:1px solid #334155;">{clean_val(pbv, '{:.2f}x')}</td></tr>
                <tr><td style="padding:8px; border-bottom:1px solid #334155;">ROE</td><td style="padding:8px; border-bottom:1px solid #334155;">{clean_val(roe * 100 if roe else None, '{:.1f}%')}</td></tr>
                <tr><td style="padding:8px;">D/E</td><td style="padding:8px;">{clean_val(de, '{:.2f}%')}</td></tr>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.warning("Data fundamental tidak tersedia.")

    # ---------- TAB 5: BACKTEST & RISIKO ----------
    with tab5:
        st.markdown("<div class='section-title'>📊 Hasil Backtest 6 Bulan</div>", unsafe_allow_html=True)
        bcol1, bcol2, bcol3, bcol4, bcol5, bcol6 = st.columns(6)
        bcol1.metric("Win Rate", f"{win_bt:.1%}" if trades_bt else "N/A")
        bcol2.metric("Profit Factor", f"{pf_bt:.2f}" if trades_bt else "N/A")
        bcol3.metric("Rata-rata Return", f"{avg_bt:.2%}" if trades_bt else "N/A")
        bcol4.metric("Max DD", f"{max_dd_bt:.2f}%" if trades_bt else "N/A")
        bcol5.metric("Sharpe", f"{sharpe_bt:.2f}" if trades_bt else "N/A")
        bcol6.metric("Total Trades", trades_bt)
        st.markdown("<div class='section-title' style='margin-top:20px;'>🛡️ Manajemen Risiko</div>", unsafe_allow_html=True)
        st.write(f"Alokasi Maksimum Kelly: **{kelly_adj*100:.1f}%** | Beta IHSG: **{beta_ihsg:.2f}x**")
        st.write(f"Max DD Historis: **{max_dd:.2f}%** | DD 30 Hari: **{max_dd_30:.2f}%**")

    # ---------- TAB 6: MONTE CARLO ----------
    with tab6:
        st.markdown("<div class='section-title'>🎲 Proyeksi Monte Carlo 30 Hari</div>", unsafe_allow_html=True)
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Prob. Naik Besok", f"{prob_bull:.1f}%")
        pcol2.metric("Prob. Sentuh R1", f"{hit_tp:.1f}%")
        pcol3.metric("Prob. Sentuh S2", f"{hit_sl:.1f}%")
        st.markdown(f"Estimasi Harga Besok: **Rp {est_besok:,.0f}** (50% range: Rp {low_est:,.0f} - Rp {up_est:,.0f})".replace(",", "."))
