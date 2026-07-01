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
st.set_page_config(page_title="Quant Risk Engine Pro v2", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    .main { background-color: #0f1116; color: #ffffff; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; color: #00ffcc; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #8892b0; }
    .stButton>button { width: 100%; background-color: #1f2937; color: white; border: 1px solid #374151; }
    .stButton>button:hover { background-color: #374151; border-color: #00ffcc; }
    h1, h2, h3 { color: #f3f4f6; }
    div[data-testid="InputInstructions"] { display: none !important; }
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

# Kamus Referensi
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
IHSG_CONDITION_INFO = {
    "RISK-ON 🔥": "Sentimen pasar positif, cocok untuk beli saham agresif.",
    "RISK-OFF 🛑": "Sentimen pasar negatif, tahan diri atau pindah ke defensif.",
    "NEUTRAL ⚖️": "Pasar tanpa arah jelas, strategi konservatif.",
    "TRANSISI ⚠️": "Pasar dalam transisi, volatilitas tinggi, entry hati-hati."
}

# ==========================================
# 2. USER INTERFACE INPUT
# ==========================================
st.title("📊 Quant & Risk Engine Pro")
st.write("Algoritma kuantitatif + Berita + Backtest Terintegrasi + Grafik Interaktif + Analisis Fundamental Saham.")

ticker_raw = st.text_input("Masukkan Kode Saham IHSG (Contoh: BRMS, BBRI, BMRI):", value="BBRI").upper().strip()

if ticker_raw and not ticker_raw.endswith(".JK"):
    ticker_input = f"{ticker_raw}.JK"
else:
    ticker_input = ticker_raw

if st.button("JALANKAN QUANT ENGINE PRO + BACKTEST"):
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

        # DATA FUNDAMENTAL
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
        z_oversold_th = np.percentile(df_thresh['ZScore'].dropna(), 30) if not df_thresh['ZScore'].dropna().empty else -1.0
        mom_median_th = np.percentile(df_thresh['Mom5D'].dropna(), 50) if not df_thresh['Mom5D'].dropna().empty else 0.0
        
        vol_hist_th = returns_thresh.rolling(20).std().dropna() * np.sqrt(252) * 100
        
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

        # BETA IHSG COVARIANCE
        try:
            ihsg = load_ihsg_data()
            ihsg_ret = ihsg['Close'].pct_change().dropna()
            common = returns.index.intersection(ihsg_ret.index)
            if len(common) > 20:
                cov = np.cov(returns.loc[common], ihsg_ret.loc[common])
                beta_ihsg = cov[0,1]/cov[1,1] if cov[1,1] > 0 else 1.0
            else: beta_ihsg = 1.0
        except: beta_ihsg = 1.0

        # PIVOT CALCULATION
        hi, lo = float(df['High'].iloc[-1]), float(df['Low'].iloc[-1])
        pp = (hi + lo + harga_terakhir) / 3
        r1, s1 = 2*pp - lo, 2*pp - hi
        r2, s2 = pp + (hi - lo), pp - (hi - lo)
        res20 = float(df['High'].iloc[-21:-1].max())
        breakout = "YES (🔥)" if harga_terakhir > res20 else "NO"

        # GENERATE SIGNALS
        def generate_signals_vectorized(dataframe, mom_th, z_th):
            score = pd.Series(0, index=dataframe.index)
            is_uptrend = (dataframe['Close'] > dataframe['EMA20']) & (dataframe['EMA20'] > dataframe['EMA50'])
            score += is_uptrend.astype(int) * 2
            score += (dataframe['Mom3D'] > mom_th).astype(int)
            if 'Volume' in dataframe.columns:
                score += (dataframe['Volume'] > dataframe['Vol_MA20']).astype(int)
                
            sig_series = pd.Series("🚨 AVOID", index=dataframe.index)
            sig_series[score == 1] = "⏸️ HOLD / WAIT"
            sig_series[score >= 2] = "⚡ BUY (TACTICAL)"
            sig_series[score >= 3] = "🔥 STRONG BUY"
            
            is_oversold = (dataframe['ZScore'] < z_th) & (dataframe['Close'] < dataframe['EMA20'])
            sig_series[is_oversold] = "⚡ BUY (TACTICAL)"
            return sig_series

        df['Signal'] = generate_signals_vectorized(df, mom_median_th, z_oversold_th)
        signal = df['Signal'].iloc[-1]

        # BACKTEST STATEFUL TRACKING
        backtest_periods = 126
        df_back = df.iloc[-backtest_periods:].copy()
        trades = []
        in_position = False
        entry_price = 0.0
        
        for i in range(len(df_back)):
            current_sig = df_back['Signal'].iloc[i]
            current_close = float(df_back['Close'].iloc[i])
            
            if not in_position:
                if "BUY" in current_sig:
                    in_position = True
                    entry_price = current_close
            else:
                if "AVOID" in current_sig or "HOLD" in current_sig or i == len(df_back) - 1:
                    exit_price = current_close
                    trade_return = (exit_price - entry_price) / entry_price
                    trades.append(trade_return)
                    in_position = False

        if trades:
            win_bt = sum(1 for r in trades if r > 0) / len(trades)
            loss_trades = [r for r in trades if r < 0]
            profit_trades = [r for r in trades if r > 0]
            pf_bt = abs(sum(profit_trades) / sum(loss_trades)) if loss_trades else np.inf
            avg_bt = np.mean(trades)
            equity = np.cumprod([1 + r for r in trades])
            max_dd_bt = float(np.min(equity / np.maximum.accumulate(equity) - 1) * 100) if len(equity) > 0 else 0
            sharpe_bt = np.mean(trades) / np.std(trades) * np.sqrt(252) if np.std(trades) > 0 else 0
            trades_bt = len(trades)
        else:
            win_bt, pf_bt, avg_bt, max_dd_bt, sharpe_bt, trades_bt = 0, 0, 0, 0, 0, 0

        # RISK METRICS & KELLY ALLOCATION
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

        mu_ou = theta_ou * (locked_log_mean20 - np.log(harga_terakhir)) + t_loc
        est_besok = float(np.exp(np.log(harga_terakhir) + mu_ou))
        sim_h1 = student_t.rvs(df_est, loc=t_loc, scale=scale_corrected, size=2000)
        prices_besok = harga_terakhir * np.exp(sim_h1)
        low_est, up_est = float(np.percentile(prices_besok, 25)), float(np.percentile(prices_besok, 75))
        
        hit_tp = (np.any(paths >= r1, axis=0).sum() / n_sim) * 100
        hit_sl = (np.any(paths <= s1, axis=0).sum() / n_sim) * 100
        prob_bull = ((sim_h1 > 0).sum() / 2000) * 100

        # ==========================================
        # 3. RENDERING DASHBOARD STREAMLIT
        # ==========================================
        st.success(f"✅ Analisis Berhasil: {ticker_input} | Closing Price: Rp {harga_terakhir:,.0f}".replace(",", "."))
        
        # SECTION BERITA
        st.header("📰 Sentimen Berita Terbobot")
        c1, c2 = st.columns([1, 2])
        c1.metric("Sentimen Skor", f"{avg_sentiment:.2f}", sentimen_status)
        with c2:
            st.markdown("**5 Berita Utama Pasar:**")
            for i, h in enumerate(headlines):
                src = sources[i] if i < len(sources) else ""
                t = translated[i] if i < len(translated) else ""
                st.markdown(f"{i+1}. **{h}** <span class='source'>({src})</span>", unsafe_allow_html=True)
                if t and t != h: st.markdown(f"<span class='translated'>🇮🇩 {t}</span>", unsafe_allow_html=True)
        st.divider()

        # SECTION REGIME MARKET
        st.header("🧬 Regime Pasar & Volatilitas")
        m1, m2, m3 = st.columns(3)
        m1.metric("Market Regime", regime)
        m2.metric("Kondisi Makro IHSG", ihsg_cond)
        m3.metric("ADX Adaptif", f"{adx:.1f} (Thresh: {adx_threshold:.1f})")
        st.markdown(f"**Insight Regime:** {REGIME_INFO.get(regime, 'Regime tidak terdefinisi.')}")
        st.divider()

        # SECTION ANALISIS FUNDAMENTAL
        st.header("📊 Metrik Fundamental Saham (IDX)")
        if ticker_info:
            def clean_val(val, fmt="{:.2f}"):
                return "N/A" if val is None else fmt.format(val)
            mc = ticker_info.get('marketCap')
            per = ticker_info.get('trailingPE') or ticker_info.get('forwardPE')
            pbv = ticker_info.get('priceToBook')
            roe = ticker_info.get('returnOnEquity')
            de = ticker_info.get('debtToEquity')
            
            table_html = f"""
            <table class='fundamental-table'>
                <tr><td>Market Cap</td><td>{clean_val(mc, "{:,.0f} IDR")}</td></tr>
                <tr><td>Price to Earnings Ratio (PER)</td><td>{clean_val(per, "{:.2f}x")}</td></tr>
                <tr><td>Price to Book Value (PBV)</td><td>{clean_val(pbv, "{:.2f}x")}</td></tr>
                <tr><td>Return On Equity (ROE)</td><td>{clean_val(roe * 100 if roe else None, "{:.1f}%")}</td></tr>
                <tr><td>Debt to Equity Ratio (D/E)</td><td>{clean_val(de, "{:.2f}%")}</td></tr>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data fundamental finansial tidak tersedia.")
        st.divider()

        # SECTION PIVOT ANALYSIS
        st.header("🎯 Target Pivot & Support/Resistance")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Resistance 2 (R2)", f"Rp {r2:,.0f}".replace(",", "."))
        p2.metric("Resistance 1 (R1)", f"Rp {r1:,.0f}".replace(",", "."))
        p3.metric("Pivot Point (PP)", f"Rp {pp:,.0f}".replace(",", "."))
        p4.metric("Support 1 (S1)", f"Rp {s1:,.0f}".replace(",", "."))
        p5.metric("Support 2 (S2)", f"Rp {s2:,.0f}".replace(",", "."))
        st.write(f"Kondisi Breakout 20 Hari: **{breakout}**")
        st.divider()

        # SECTION SIGNAL & EXPANDING BACKTEST (MODIFIKASI: Ditambahkan Stop Loss Target di t5)
        st.header("🔮 Sinyal Kuantitatif & Hasil Backtest Realistis (6 Bulan)")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Sinyal Eksekusi", signal)
        t2.metric("Estimasi Besok", f"Rp {est_besok:,.0f}".replace(",", "."), f"Rentang 50%: {low_est:,.0f} - {up_est:,.0f}".replace(",", "."))
        t3.metric("Wilayah Entry Ideal", f"Rp {s1:,.0f} - {pp:,.0f}".replace(",", "."))
        t4.metric("Take Profit Target", f"Rp {r1:,.0f}".replace(",", "."))
        t5.metric("Stop Loss Target (S2)", f"Rp {s2:,.0f}".replace(",", "."))
        
        st.markdown("**Hasil Pengujian Algoritma (Stateful Tracking Backtest 126 Hari):**")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Win Rate", f"{win_bt:.1%}" if trades_bt else "N/A")
        b2.metric("Profit Factor", f"{pf_bt:.2f}" if trades_bt and pf_bt != np.inf else "N/A")
        b3.metric("Rata-rata Return/Trade", f"{avg_bt:.2%}" if trades_bt else "N/A")
        b4.metric("Max Drawdown Strategi", f"{max_dd_bt:.2f}%" if trades_bt else "N/A")
        b5.metric("Sharpe Ratio", f"{sharpe_bt:.2f}" if trades_bt else "N/A")
        b6.metric("Total Trades Riil", f"{trades_bt}")
        st.divider()

        # SECTION ALOKASI MANAJEMEN RISIKO
        st.header("🛡️ Kriteria Manajemen Risiko Portofolio Terkalibrasi (Kelly)")
        r_c1, r_c2 = st.columns(2)
        r_c1.metric("Rekomendasi Ukuran Posisi (Kelly)", f"{kelly_adj*100:.1f}%")
        r_c2.metric("Beta Terhadap IHSG", f"{beta_ihsg:.2f}x")
        st.markdown(f"**Interpretasi Posisi:** Berdasarkan akurasi *Win Rate* strategi kuantitatif Anda senilai **{win_bt:.1%}**, sistem menyarankan batas maksimal ukuran tunggal saham ini adalah **{kelly_adj*100:.1f}%** dari total seluruh ekuitas modal portofolio Anda.")
        st.markdown(f"Statistik Historis Sektor -> Max DD: `{max_dd:.2f}%` | Drawdown 30 Hari: `{max_dd_30:.2f}%`")
        st.divider()

        # SECTION MONTE CARLO PROYEKSI MAJU
        st.header("🎲 Simulasi Monte Carlo Ornstein-Uhlenbeck")
        pr1, pr2, pr3 = st.columns(3)
        pr1.metric("Probabilitas Naik Besok", f"{prob_bull:.1f}%")
        pr2.metric("Probabilitas Kena R1 (30 Hari)", f"{hit_tp:.1f}%")
        pr3.metric("Probabilitas Turun S1 (30 Hari)", f"{hit_sl:.1f}%")

        # CHART PLOTLY
        if PLOTLY_AVAILABLE:
            st.header("📈 Chart Harga & Pemetaan Histori Sinyal")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Harga Close', line=dict(color='#00ffcc', width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20 (Trend Pendek)', line=dict(color='#f59e0b', dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50 (Trend Menengah)', line=dict(color='#ef4444', dash='dot')))
            
            buy_signals = df_back[df_back['Signal'].str.contains("BUY")]
            fig.add_trace(go.Scatter(
                x=buy_signals.index, y=buy_signals['Close'],
                mode='markers', name='Sinyal Transaksi Buy',
                marker=dict(symbol='triangle-up', size=11, color='#10b981', line=dict(width=1, color='white'))
            ))
            
            for level, label, col in [(r1, 'R1', 'orange'), (s1, 'S1', 'red'), (pp, 'PP', 'gray')]:
                fig.add_hline(y=level, line_dash="dash", line_color=col, annotation_text=label, annotation_position="right")
                
            fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        # TRADING RECOMMENDATION EXECUTIVE SUMMARY
        st.markdown("---")
        st.header("📋 Ringkasan Eksekutif & Rekomendasi")
        
        if "STRONG BUY" in signal:
            action_color, action_icon, action_text = "#10b981", "🟢", f"Algoritma mendeteksi penguatan momentum penuh dengan konfirmasi volume tinggi. Masuk secara berkala hingga batas maksimal {kelly_adj*100:.1f}% dari portfolio, dengan pembatasan risiko ketat di area Stop Loss Rp {s2:,.0f}."
        elif "BUY" in signal:
            action_color, action_icon, action_text = "#f59e0b", "🟡", f"Sinyal beli taktis terdeteksi secara parsial. Anda bisa melakukan buy-on-weakness di dekat area Support 1, batasi risiko jika harga menembus Rp {s2:,.0f}."
        elif "HOLD" in signal:
            action_color, action_icon, action_text = "#3b82f6", "🔵", "Pasar bergerak konsolidasi tanpa arah yang dominan. Pertimbangkan Hold posisi yang ada dan batasi porsi penambahan modal."
        else:
            action_color, action_icon, action_text = "#ef4444", "🔴", "Sinyal menunjukkan risiko penurunan tinggi atau kondisi pasar jenuh beli. Amankan profit dan hindari entry baru."
            
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="summary-card">
                <div class="section-title">📌 Profil Risiko Saham Saat Ini</div>
                <div class="summary-item">🏷️ <b>Kategori Rezim:</b> {regime}</div>
                <div class="summary-item">🌐 <b>Kondisi Makro:</b> {ihsg_cond}</div>
                <div class="summary-item">📊 <b>Skor Sentimen Berita:</b> {avg_sentiment:.2f} ({sentimen_status})</div>
                <div class="summary-item">🛡️ <b>Maks. Batas Ukuran Posisi:</b> {kelly_adj*100:.1f}% dari Total Portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="action-card" style="border-left-color: {action_color};">
                <div class="section-title">{action_icon} Panduan Tindakan</div>
                <div class="summary-item" style="font-size: 16px; margin-top: 8px;">
                    {action_text}
                </div>
                <hr style="border-color: #334155; margin: 15px 0;">
                <div style="color: #94a3b8; font-size: 13px;">
                    ⚠️ <i>Disclaimer: Hasil analisis ini berbasis permodelan matematika probabilitas kuantitatif. Keputusan investasi tetap berada pada kendali dan tanggung jawab penuh Anda pribadi.</i>
                </div>
            </div>
            """, unsafe_allow_html=True)




