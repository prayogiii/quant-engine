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
        nltk.download('vader_lexicon')
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
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Quant Risk Engine Pro", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
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

st.title("📊 Quant & Risk Engine Pro (Final + Quant Score 100)")
st.write("Algoritma kuantitatif + berita + Backtest + Grafik + Fundamental + **Quant Score 100** sebagai otak utama.")

if not SENTIMENT_AVAILABLE: st.warning("⚠️ NLTK tidak terpasang")
if not RSS_AVAILABLE: st.warning("⚠️ feedparser tidak terpasang")
if not TRANSLATOR_AVAILABLE: st.info("💡 deep-translator tidak terpasang")
if not PLOTLY_AVAILABLE: st.info("📈 plotly tidak terpasang – grafik tidak akan ditampilkan. Install dengan `pip install plotly`.")

# ==========================================
# 2. INPUT
# ==========================================
ticker_raw = st.text_input("Masukkan Kode Saham IHSG (Contoh: BRMS, BBRI, BMRI):", value="").upper().strip()
total_capital = st.number_input("Total Modal Portofolio Anda (Rp):", min_value=0, value=None, step=10000, placeholder="Masukkan nominal modal anda...")
if total_capital is not None and total_capital > 0:
    st.markdown(f"✍️ *Terbaca:* **Rp {total_capital:,.0f}**".replace(",", "."))

if ticker_raw and not ticker_raw.endswith(".JK"):
    ticker_input = f"{ticker_raw}.JK"
else:
    ticker_input = ticker_raw

# ==================== FUNGSI UTILITAS ====================
def compute_adx(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]

def compute_adx_series(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
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

def get_yahoo_ticker_news(ticker, num=5):
    try:
        items = yf.Ticker(ticker).news or []
        news = []
        for item in items[:num]:
            inner = item.get('content') or item
            title = (inner.get('title') or inner.get('shortTitle') or inner.get('headline') or '')
            summary = (inner.get('summary') or inner.get('longSummary') or inner.get('description') or '')
            if title: news.append({'title': title, 'summary': summary, 'source': 'Yahoo Ticker'})
        return news, None
    except: return [], "Yahoo Ticker gagal"

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

# ==================== KAMUS ====================
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
# =====================================================

if st.button("JALANKAN QUANT ENGINE PRO + BACKTEST"):
    if not ticker_input: st.warning("⚠️ Kode saham tidak boleh kosong!")
    elif total_capital is None or total_capital <= 0: st.warning("⚠️ Modal portofolio harus diisi dan > Rp 0!")
    else:
        with st.spinner("🤖 Mengunduh data, berita, backtest, dan model kuantitatif..."):
            try:
                # DATA HARGA
                df = yf.download(ticker_input, period="1y")
                if df.empty: st.error("❌ Data tidak ditemukan."); st.stop()
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                harga_terakhir = float(df['Close'].iloc[-1])
                returns = df['Close'].pct_change().dropna()
                if len(returns) < 20: st.error("❌ Data historis kurang (minimal 20 hari)."); st.stop()

                # === DATA FUNDAMENTAL ===
                try:
                    ticker_info = yf.Ticker(ticker_input).info
                except:
                    ticker_info = {}

                # BERITA
                news_pool = []
                translator_en = GoogleTranslator(source='auto', target='en') if TRANSLATOR_AVAILABLE else None
                translator_id = GoogleTranslator(source='auto', target='id') if TRANSLATOR_AVAILABLE else None
                rss, _ = get_google_news_rss(f"{ticker_raw} saham")
                if rss: news_pool.extend(rss)
                ysearch, _ = get_yahoo_search_news(f"{ticker_raw} saham")
                if ysearch: news_pool.extend(ysearch)
                if not news_pool:
                    ytick, _ = get_yahoo_ticker_news(ticker_input)
                    if ytick: news_pool.extend(ytick)
                news_pool = filter_relevant(news_pool, ticker_raw)
                seen = set()
                unique_news = []
                for n in news_pool:
                    if n['title'] not in seen:
                        seen.add(n['title']); unique_news.append(n)
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
                sentimen_status = "Positif 🟢" if avg_sentiment >= 0.05 else ("Negatif 🔴" if avg_sentiment <= -0.05 else "Netral ⚪") if SENTIMENT_AVAILABLE else "Nonaktif"

                # VOLATILITAS
                log_hl = np.log(df['High'] / df['Low'])**2
                parkinson_vol = float(np.sqrt(log_hl.mean()/(4*np.log(2))) * np.sqrt(252)*100)
                ewma_vol = float(np.sqrt(returns.ewm(alpha=0.06).var().iloc[-1]) * np.sqrt(252)*100)

                # BETA IHSG
                try:
                    ihsg = yf.download("^JKSE", period="1y")
                    if isinstance(ihsg.columns, pd.MultiIndex): ihsg.columns = ihsg.columns.get_level_values(0)
                    ihsg_ret = ihsg['Close'].pct_change().dropna()
                    common = returns.index.intersection(ihsg_ret.index)
                    if len(common)>20:
                        cov = np.cov(returns.loc[common], ihsg_ret.loc[common])
                        beta_ihsg = cov[0,1]/cov[1,1] if cov[1,1]>0 else 1.0
                    else: beta_ihsg=1.0
                except: beta_ihsg=1.0

                # === RELATIVE STRENGTH VS IHSG (20 hari) ===
                try:
                    rs_20 = (
                        (df['Close'].iloc[-1] / df['Close'].iloc[-20])
                        /
                        (ihsg['Close'].iloc[-1] / ihsg['Close'].iloc[-20])
                    )
                except:
                    rs_20 = 1.0

                # ==================== THRESHOLD & PARAMETER DARI 6 BULAN PERTAMA ====================
                split_idx = max(126, len(df) - 126)
                df_thresh = df.iloc[:split_idx]
                returns_thresh = df_thresh['Close'].pct_change().dropna()

                adx_series = compute_adx_series(df_thresh)
                adx_threshold = np.percentile(adx_series.dropna(), 75) if len(adx_series.dropna()) > 0 else 20

                z_hist_th = (df_thresh['Close'] - df_thresh['Close'].rolling(20).mean()) / df_thresh['Close'].rolling(20).std()
                z_oversold_th = np.percentile(z_hist_th.dropna(), 30)
                mom5_hist_th = df_thresh['Close'].pct_change(5).dropna()*100
                mom_median_th = np.percentile(mom5_hist_th, 50)

                vol_hist_th = returns_thresh.rolling(20).std().dropna()*np.sqrt(252)*100
                high_vol_th = np.percentile(vol_hist_th, 70)
                low_vol_th = np.percentile(vol_hist_th, 30)

                def t_loglike(p, d):
                    if p[0]<=2 or p[2]<=0: return np.inf
                    return -np.sum(student_t.logpdf(d, p[0], p[1], p[2]))
                res_thresh = minimize(t_loglike, [5, returns_thresh.mean(), returns_thresh.std()],
                                      bounds=[(2.1,100),(-0.1,0.1),(1e-6,None)], args=(returns_thresh,), method='L-BFGS-B')
                df_est, t_loc, t_scale = res_thresh.x if res_thresh.success else (5, returns_thresh.mean(), returns_thresh.std())

                # ==================== FUNGSI REGIME ====================
                def get_regime(slice_df):
                    close = slice_df['Close']
                    h = float(close.iloc[-1])
                    ema20s = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
                    ema50s = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
                    adxs = compute_adx(slice_df)
                    zs = (h - close.tail(20).mean()) / close.tail(20).std() if close.tail(20).std() > 0 else 0
                    m5 = float((h / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0
                    if adxs > adx_threshold:
                        if h > ema20s and ema20s > ema50s:
                            if m5 > mom_median_th or zs > z_oversold_th: return "Strong Bullish 🚀", "RISK-ON 🔥"
                            else: return "Bullish 📈", "RISK-ON 🔥"
                        elif h < ema20s and ema20s < ema50s:
                            if m5 < mom_median_th or zs < z_oversold_th: return "Panic Sell 🚨", "RISK-OFF 🛑"
                            else: return "Bearish 🔻", "RISK-OFF 🛑"
                        elif h > ema20s and ema20s < ema50s: return "Early Recovery 🔄", "TRANSISI ⚠️"
                        elif h < ema20s and ema20s > ema50s: return "Distribution 📉", "TRANSISI ⚠️"
                        else: return "Konsolidasi Tren ↔️", "NEUTRAL ⚖️"
                    else:
                        if h > ema20s and ema20s > ema50s: return "Bullish Accumulation 🏗️", "NEUTRAL ⚖️"
                        elif h < ema20s and ema20s < ema50s: return "Bearish Accumulation 🧊", "NEUTRAL ⚖️"
                        elif h > ema20s and ema20s < ema50s: return "Sideways Bias Naik ↗️", "NEUTRAL ⚖️"
                        elif h < ema20s and ema20s > ema50s: return "Sideways Bias Turun ↘️", "NEUTRAL ⚖️"
                        else:
                            ret_s = close.pct_change().dropna()
                            ewma_vol_s = float(np.sqrt(ret_s.ewm(alpha=0.06).var().iloc[-1]) * np.sqrt(252)*100) if len(ret_s) >= 20 else 0
                            if ewma_vol_s > high_vol_th: return "Sideways Choppy 🌊", "NEUTRAL ⚖️"
                            elif ewma_vol_s < low_vol_th: return "Sideways Calm 😴", "NEUTRAL ⚖️"
                            else: return "Sideways Normal ↔️", "NEUTRAL ⚖️"

                regime, ihsg_cond = get_regime(df)
                adx = compute_adx(df)

                # PIVOT
                hi, lo = float(df['High'].iloc[-1]), float(df['Low'].iloc[-1])
                pp = (hi+lo+harga_terakhir)/3
                r1, s1 = 2*pp-lo, 2*pp-hi
                r2, s2 = pp+(hi-lo), pp-(hi-lo)
                res20 = float(df['High'].iloc[-21:-1].max())
                breakout = "YES (🔥)" if harga_terakhir > res20 else "NO"

                # MOMENTUM LENGKAP (3D, 5D, 10D)
                mom_3d = float((df['Close'].iloc[-1]/df['Close'].iloc[-4]-1)*100)
                mom_5d = float((df['Close'].iloc[-1]/df['Close'].iloc[-6]-1)*100)
                mom_10d = float((df['Close'].iloc[-1]/df['Close'].iloc[-11]-1)*100)
                z_score = (harga_terakhir - df['Close'].tail(20).mean()) / df['Close'].tail(20).std() if df['Close'].tail(20).std()>0 else 0
                ema20 = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[-1])

                # === VOLUME SPIKE ===
                vol_ratio = 1.0
                if 'Volume' in df.columns:
                    vol_ratio = (
                        df['Volume'].iloc[-1]
                        / df['Volume'].tail(20).mean()
                    )

                # SIGNAL LIVE
                score = 0
                if mom_3d > mom_median_th: score += 1
                if z_score < z_oversold_th: score += 1
                if harga_terakhir > ema20: score += 1
                if 'Volume' in df.columns and df['Volume'].iloc[-1] > df['Volume'].tail(20).mean(): score += 1
                if SENTIMENT_AVAILABLE and avg_sentiment > 0.2: score += 1
                elif SENTIMENT_AVAILABLE and avg_sentiment < -0.2: score -= 1
                signal = "🔥 STRONG BUY" if score >= 3 else ("⚡ BUY (TACTICAL)" if score >= 2 else ("⏸️ HOLD / WAIT" if score == 1 else "🚨 AVOID"))

                # BACKTEST EXPANDING WINDOW
                def backtest_expanding(df, periods=126):
                    df_back = df.iloc[-periods:].copy()
                    signals = []
                    for i in range(20, len(df_back)):
                        slice_df = df.iloc[:df.index.get_loc(df_back.index[i-1])+1]
                        sl_thresh = slice_df.iloc[:max(126, len(slice_df)-126)] if len(slice_df) >= 126 else slice_df
                        ret_sl = sl_thresh['Close'].pct_change().dropna()
                        z_hist_sl = (sl_thresh['Close'] - sl_thresh['Close'].rolling(20).mean()) / sl_thresh['Close'].rolling(20).std()
                        z_ov_sl = np.percentile(z_hist_sl.dropna(), 30) if len(z_hist_sl.dropna()) > 0 else -1.5
                        mom5_sl = sl_thresh['Close'].pct_change(5).dropna()*100
                        mom_med_sl = np.percentile(mom5_sl, 50) if len(mom5_sl) > 0 else 0.0
                        h = float(slice_df['Close'].iloc[-1])
                        m3 = float((slice_df['Close'].iloc[-1]/slice_df['Close'].iloc[-4]-1)*100)
                        ema20s = float(slice_df['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
                        zs = (h - slice_df['Close'].tail(20).mean())/slice_df['Close'].tail(20).std() if slice_df['Close'].tail(20).std()>0 else 0
                        vol_cond = slice_df['Volume'].iloc[-1] > slice_df['Volume'].tail(20).mean() if 'Volume' in slice_df.columns else False
                        s = 0
                        if m3 > mom_med_sl: s += 1
                        if zs < z_ov_sl: s += 1
                        if h > ema20s: s += 1
                        if vol_cond: s += 1
                        sig = "🔥 STRONG BUY" if s >= 2 else ("⚡ BUY (TACTICAL)" if s >= 1 else "🚨 AVOID")
                        signals.append((df_back.index[i], sig))
                    trades = []
                    for i in range(len(signals)-1):
                        date, sig = signals[i]
                        next_date = signals[i+1][0]
                        if "BUY" in sig:
                            entry = df.loc[date, 'Close']
                            exit_ = df.loc[next_date, 'Close']
                            trades.append((exit_ - entry) / entry)
                    if trades:
                        win_rate = sum(1 for r in trades if r > 0) / len(trades)
                        profit_factor = abs(sum(r for r in trades if r > 0) / sum(r for r in trades if r < 0)) if sum(r for r in trades if r < 0) != 0 else np.inf
                        avg_return = np.mean(trades)
                        equity = np.cumprod([1+r for r in trades])
                        max_dd_bt = float(np.min(equity / np.maximum.accumulate(equity) - 1) * 100)
                        sharpe_bt = np.mean(trades) / np.std(trades) * np.sqrt(252/30) if np.std(trades) > 0 else 0
                        return win_rate, profit_factor, avg_return, len(trades), max_dd_bt, sharpe_bt
                    return 0, 0, 0, 0, 0, 0

                win_bt, pf_bt, avg_bt, trades_bt, maxdd_bt, sharpe_bt = backtest_expanding(df)

                # RISK METRICS
                roll_max_th = df_thresh['Close'].cummax()
                drawdown_th = (df_thresh['Close'] - roll_max_th) / roll_max_th
                max_dd = float(drawdown_th.min()*100)
                max_dd_30 = float(drawdown_th.tail(30).min()*100) if len(drawdown_th) >= 30 else max_dd
                mean_ret = returns_thresh.mean()*252
                std_ret = returns_thresh.std()*np.sqrt(252)
                sharpe = mean_ret/std_ret if std_ret>0 else 0
                down_std = returns_thresh[returns_thresh<0].std()*np.sqrt(252) if len(returns_thresh[returns_thresh<0])>0 else std_ret
                sortino = mean_ret/down_std if down_std>0 else 0
                calmar = mean_ret/abs(max_dd/100) if max_dd!=0 else 0
                win_r = len(returns_thresh[returns_thresh>0])/len(returns_thresh)
                avg_g = returns_thresh[returns_thresh>0].mean() if win_r>0 else 0.01
                avg_l = abs(returns_thresh[returns_thresh<0].mean()) if len(returns_thresh[returns_thresh<0])>0 else 0.01
                wl = avg_g/avg_l if avg_l>0 else 1
                kelly_raw = win_r - (1-win_r)/wl
                ret_skew = float(skew(returns_thresh))
                ret_kurt = float(kurtosis(returns_thresh, fisher=True))
                kurt_penalty = 0.5 if ret_kurt > 3 else 1.0
                kelly_adj = min(0.25, max(0.0, kelly_raw * 0.3 * (0.5 if ret_skew < -0.5 else 1.0) * kurt_penalty))
                alloc = total_capital*kelly_adj

                # MONTE CARLO OU
                n_sim, n_days = 2000, 30
                latest_vol_daily = np.sqrt(returns.ewm(alpha=0.06).var().iloc[-1])
                scale_corrected = latest_vol_daily / np.sqrt(df_est / (df_est - 2)) if df_est > 2 else latest_vol_daily
                theta_ou = estimate_theta_ou(df['Close'])
                log_mean_series = np.log(df['Close']).rolling(20).mean().dropna()
                paths = np.zeros((n_days, n_sim))
                current_log = np.log(harga_terakhir)
                for day in range(n_days):
                    log_mean20_val = log_mean_series.iloc[-day-1] if (day % 5 == 0 and day != 0 and len(log_mean_series) > day) else np.log(df['Close']).tail(20).mean()
                    innovations = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=n_sim)
                    current_log = current_log + theta_ou * (log_mean20_val - current_log) + innovations
                    paths[day, :] = np.exp(current_log)
                mu_ou = theta_ou * (np.log(df['Close']).tail(20).mean() - np.log(harga_terakhir)) + t_loc
                est_besok = float(np.exp(np.log(harga_terakhir) + mu_ou))
                sim_h1 = student_t.rvs(df_est, loc=t_loc, scale=scale_corrected, size=2000)
                prices_besok = harga_terakhir * np.exp(sim_h1)
                low_est, up_est = float(np.percentile(prices_besok,25)), float(np.percentile(prices_besok,75))
                final_prices = paths[-1, :]
                es_95_mc = float(np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)]))
                es_95_pct = (harga_terakhir - es_95_mc) / harga_terakhir * 100
                tp, sl = r1, s1
                hit_tp = (np.any(paths >= tp, axis=0).sum() / n_sim) * 100
                hit_sl = (np.any(paths <= sl, axis=0).sum() / n_sim) * 100
                prob_bull = ((sim_h1 > 0).sum() / 2000) * 100

                # === DATA FUNDAMENTAL UNTUK QUANT SCORE ===
                per = ticker_info.get('trailingPE') or ticker_info.get('forwardPE')
                pbv = ticker_info.get('priceToBook')
                roe = ticker_info.get('returnOnEquity')

                # ==================== QUANT SCORE 100 ====================
                quant_score = 0
                # 1. REGIME (20)
                if "Strong Bullish" in regime:
                    quant_score += 20
                elif "Bullish" in regime:
                    quant_score += 16
                elif "Early Recovery" in regime:
                    quant_score += 12
                elif "Bullish Accumulation" in regime:
                    quant_score += 10
                elif "Distribution" in regime:
                    quant_score += 6
                else:
                    quant_score += 2

                # 2. ADX (15)
                if adx > 40:
                    quant_score += 15
                elif adx > 30:
                    quant_score += 12
                elif adx > 20:
                    quant_score += 8
                else:
                    quant_score += 3

                # 3. RELATIVE STRENGTH VS IHSG (15)
                if rs_20 > 1.10:
                    quant_score += 15
                elif rs_20 > 1.03:
                    quant_score += 10
                elif rs_20 > 1:
                    quant_score += 6

                # 4. VOLUME SPIKE (10)
                if vol_ratio > 2:
                    quant_score += 10
                elif vol_ratio > 1.5:
                    quant_score += 7
                elif vol_ratio > 1:
                    quant_score += 4

                # 5. SENTIMEN (10)
                if avg_sentiment > 0.30:
                    quant_score += 10
                elif avg_sentiment > 0.10:
                    quant_score += 7
                elif avg_sentiment >= 0:
                    quant_score += 4

                # 6. MONTE CARLO (15)
                if prob_bull > 70:
                    quant_score += 15
                elif prob_bull > 60:
                    quant_score += 10
                elif prob_bull > 50:
                    quant_score += 5

                # 7. FUNDAMENTAL (15)
                fund_score = 0
                if per is not None and per < 15:
                    fund_score += 5
                if pbv is not None and pbv < 2:
                    fund_score += 5
                if roe is not None and roe > 0.15:
                    fund_score += 5
                quant_score += fund_score

                # GRADE
                if quant_score >= 85:
                    grade = "A+ 🚀"
                elif quant_score >= 75:
                    grade = "A 🟢"
                elif quant_score >= 65:
                    grade = "B 🟡"
                elif quant_score >= 50:
                    grade = "C 🟠"
                else:
                    grade = "D 🔴"

                # ==================== TAMPILAN ====================
                st.success(f"✅ Analisis: {ticker_input} | Harga: Rp {harga_terakhir:,.0f}".replace(",","."))

                # --- QUANT SCORE DASHBOARD (DITAMPILKAN UTAMA) ---
                st.header("🏆 Quant Score 100")
                q1, q2 = st.columns(2)
                q1.metric("Quant Score", f"{quant_score}/100")
                q2.metric("Grade", grade)
                st.caption("""
                Interpretasi:
                85+ = Sangat menarik
                75–84 = Layak dipertimbangkan
                65–74 = Cukup baik
                50–64 = Spekulatif
                <50 = Hindari
                """)
                st.divider()

                # --- BERITA ---
                st.header("📰 Sentimen Berita")
                st.caption("Berita diambil dari Google News & Yahoo Finance, sentimen dihitung dengan bobot.")
                c1,c2=st.columns([1,2])
                c1.metric("Sentimen", f"{avg_sentiment:.2f}", sentimen_status)
                with c2:
                    st.markdown("**5 Berita Teratas:**")
                    for i,h in enumerate(headlines):
                        src = sources[i] if i<len(sources) else ""
                        t = translated[i] if i<len(translated) else ""
                        st.markdown(f"{i+1}. **{h}** <span class='source'>({src})</span>", unsafe_allow_html=True)
                        if t and t!=h: st.markdown(f"<span class='translated'>🇮🇩 {t}</span>", unsafe_allow_html=True)
                        st.markdown("")
                st.divider()

                # --- REGIME ---
                st.header("🧬 Regime & Volatility")
                m1,m2,m3=st.columns(3)
                m1.metric("Regime", regime); m2.metric("IHSG", ihsg_cond); m3.metric("ADX", f"{adx:.1f}")
                st.markdown(f"EWMA Vol: `{ewma_vol:.2f}%` | Parkinson: `{parkinson_vol:.2f}%` | T(df={df_est:.1f})")
                st.markdown(f"**Apa artinya?** {REGIME_INFO.get(regime, '')}")
                st.markdown(f"**Kondisi IHSG:** {IHSG_CONDITION_INFO.get(ihsg_cond, '')}")
                st.markdown(f"**ADX threshold adaptif:** {adx_threshold:.1f}")
                st.divider()

                # --- FUNDAMENTAL (DETAIL) ---
                st.header("📊 Analisis Fundamental")
                st.caption("Data fundamental dari laporan keuangan terbaru (jika tersedia). Klasifikasi berdasarkan PER & PBV sebagai acuan sederhana.")
                if ticker_info:
                    market_cap = ticker_info.get('marketCap')
                    eps = ticker_info.get('trailingEps')
                    debt_equity = ticker_info.get('debtToEquity')
                    dividend_yield = ticker_info.get('dividendYield')
                    table_html = "<table class='fundamental-table'>"
                    table_html += f"<tr><td>Market Cap</td><td>{market_cap:,.0f} IDR</td></tr>" if market_cap else ""
                    table_html += f"<tr><td>PER</td><td>{per:.2f}x</td></tr>" if per else ""
                    table_html += f"<tr><td>PBV</td><td>{pbv:.2f}x</td></tr>" if pbv else ""
                    table_html += f"<tr><td>EPS</td><td>{eps:.2f}</td></tr>" if eps else ""
                    table_html += f"<tr><td>ROE</td><td>{roe*100:.1f}%</td></tr>" if roe else ""
                    table_html += f"<tr><td>Debt/Equity</td><td>{debt_equity:.2f}%</td></tr>" if debt_equity else ""
                    table_html += f"<tr><td>Div Yield</td><td>{dividend_yield*100:.2f}%</td></tr>" if dividend_yield else ""
                    table_html += "</table>"
                    st.markdown(table_html, unsafe_allow_html=True)
                    interpretation = []
                    if per:
                        if per < 10: per_status, per_color = "Rendah (undervalued)", "#10b981"
                        elif per > 25: per_status, per_color = "Tinggi (overvalued)", "#ef4444"
                        else: per_status, per_color = "Moderat", "#f59e0b"
                        interpretation.append(f"PER {per:.1f}x <span style='color:{per_color}; font-weight:bold;'>{per_status}</span>.")
                    if pbv:
                        if pbv < 1: pbv_status, pbv_color = "di bawah 1 (undervalued)", "#10b981"
                        elif pbv > 3: pbv_status, pbv_color = "di atas 3 (overvalued)", "#ef4444"
                        else: pbv_status, pbv_color = "normal", "#f59e0b"
                        interpretation.append(f"PBV {pbv:.1f}x <span style='color:{pbv_color}; font-weight:bold;'>{pbv_status}</span>.")
                    if roe:
                        if roe > 0.15: roe_status, roe_color = "baik (>15%)", "#10b981"
                        elif roe < 0.05: roe_status, roe_color = "rendah", "#ef4444"
                        else: roe_status, roe_color = "cukup", "#f59e0b"
                        interpretation.append(f"ROE {roe*100:.1f}% <span style='color:{roe_color}; font-weight:bold;'>{roe_status}</span>.")
                    if debt_equity:
                        if debt_equity > 100: de_status, de_color = "tinggi", "#ef4444"
                        else: de_status, de_color = "aman", "#10b981"
                        interpretation.append(f"D/E {debt_equity:.1f}% <span style='color:{de_color}; font-weight:bold;'>{de_status}</span>.")
                    if interpretation:
                        st.markdown("**Interpretasi:** " + " ".join(interpretation), unsafe_allow_html=True)
                    else:
                        st.markdown("Data fundamental tidak mencukupi untuk interpretasi.")
                else:
                    st.markdown("Data fundamental tidak tersedia untuk saham ini.")
                st.divider()

                # --- MOMENTUM ---
                st.header("📊 Momentum & Z‑Score")
                mo1,mo2,mo3,mo4=st.columns(4)
                mo1.metric("3D", f"{mom_3d:+.2f}%"); mo2.metric("5D", f"{mom_5d:+.2f}%"); mo3.metric("10D", f"{mom_10d:+.2f}%"); mo4.metric("Z", f"{z_score:+.2f}σ")
                st.divider()

                # --- PIVOT ---
                st.header("🎯 Pivot & S/R")
                st.write(f"Breakout Res20: `{breakout}`")
                p1,p2,p3,p4,p5=st.columns(5)
                p1.metric("R2", f"Rp {r2:,.0f}".replace(",",".")); p2.metric("R1", f"Rp {r1:,.0f}".replace(",",".")); p3.metric("PP", f"Rp {pp:,.0f}".replace(",",".")); p4.metric("S1", f"Rp {s1:,.0f}".replace(",",".")); p5.metric("S2", f"Rp {s2:,.0f}".replace(",","."))
                st.divider()

                # --- SIGNAL & BACKTEST ---
                st.header("🔮 Signal & Trading Plan With Backtest")
                t1,t2,t3,t4=st.columns(4)
                t1.metric("Signal", signal)
                t2.metric("Est. Besok", f"Rp {est_besok:,.0f}".replace(",","."), f"25-75%: {low_est:,.0f} – {up_est:,.0f}".replace(",","."))
                t3.metric("Entry", f"Rp {s1:,.0f} - {pp:,.0f}".replace(",","."))
                t4.metric("Target", f"Rp {r1:,.0f}".replace(",","."))
                st.caption("💡 Interval 25%-75% adalah rentang harga besok yang paling mungkin (probabilitas 50%).")
                st.markdown("**📈 Backtest 6 Bulan (Expanding Window):**")
                b1,b2,b3,b4,b5,b6=st.columns(6)
                b1.metric("Win Rate", f"{win_bt:.1%}" if trades_bt else "N/A")
                b2.metric("Profit Factor", f"{pf_bt:.2f}" if trades_bt else "N/A")
                b3.metric("Avg Return", f"{avg_bt:.2%}" if trades_bt else "N/A")
                b4.metric("Max DD", f"{maxdd_bt:.2f}%" if trades_bt else "N/A")
                b5.metric("Sharpe", f"{sharpe_bt:.2f}" if trades_bt else "N/A")
                b6.metric("Trades", f"{trades_bt}")
                st.divider()

                # --- RISK ---
                st.header("🛡️ Risk & Portfolio Sizing")
                r1,r2,r3=st.columns(3)
                r1.metric("Kelly Adj.", f"{kelly_adj*100:.1f}%")
                r2.metric("Rekom. Modal", f"Rp {alloc:,.0f}".replace(",","."))
                r3.metric("Beta", f"{beta_ihsg:.2f}x")
                st.markdown(f"Max DD: `{max_dd:.2f}%` (30D: `{max_dd_30:.2f}%`) | Sharpe: `{sharpe:.2f}` | Sortino: `{sortino:.2f}` | Calmar: `{calmar:.2f}`")
                st.divider()

                # --- MONTE CARLO ---
                st.header("🎲 Monte Carlo (OU, Student‑t, vol adaptif)")
                pr1,pr2,pr3=st.columns(3)
                pr1.metric("Prob Bullish Besok", f"{prob_bull:.1f}%"); pr2.metric("Prob TP 30D", f"{hit_tp:.1f}%"); pr3.metric("Prob SL 30D", f"{hit_sl:.1f}%")

                # --- GRAFIK (jika plotly tersedia) ---
                if PLOTLY_AVAILABLE:
                    st.header("📈 Chart Harga & Indikator")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#00ffcc')))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=20, adjust=False).mean(), name='EMA20', line=dict(color='#f59e0b', dash='dot')))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=50, adjust=False).mean(), name='EMA50', line=dict(color='#ef4444', dash='dot')))
                    for level, label in [(r1, 'R1'), (s1, 'S1'), (pp, 'PP')]:
                        fig.add_hline(y=level, line_dash="dash", line_color="gray", annotation_text=label, annotation_position="right")
                    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=20,b=0))
                    st.plotly_chart(fig, use_container_width=True)

                # --- KESIMPULAN ---
                st.markdown("---")
                st.header("📋 Kesimpulan & Rekomendasi Trading")
                if "STRONG BUY" in signal:
                    action_color, action_icon, action_text = "#10b981", "🟢", "Pertimbangkan untuk membeli dengan ukuran posisi sesuai alokasi Kelly. Pasang stop loss di bawah S1."
                elif "BUY" in signal:
                    action_color, action_icon, action_text = "#f59e0b", "🟡", "Sinyal beli taktis muncul, tetapi belum terlalu kuat. Bisa entry dengan porsi lebih kecil atau menunggu konfirmasi tambahan."
                elif "HOLD" in signal:
                    action_color, action_icon, action_text = "#3b82f6", "🔵", "Tahan posisi jika sudah ada, hindari entry baru sampai sinyal lebih jelas."
                else:
                    action_color, action_icon, action_text = "#ef4444", "🔴", "Hindari pembelian. Pertimbangkan untuk keluar dari posisi atau menunggu pullback ke support kuat."
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"""
                    <div class="summary-card">
                        <div class="section-title">📌 Ringkasan Kondisi</div>
                        <div class="summary-item">🏷️ <b>Regime:</b> {regime}</div>
                        <div class="summary-item">🌐 <b>IHSG Condition:</b> {ihsg_cond}</div>
                        <div class="summary-item">📊 <b>ADX:</b> {adx:.1f}</div>
                        <div class="summary-item">📰 <b>Sentimen:</b> {sentimen_status} ({avg_sentiment:.2f})</div>
                        <div class="summary-item">🔮 <b>Sinyal:</b> {signal}</div>
                        <div class="summary-item">💰 <b>Estimasi Besok:</b> Rp {est_besok:,.0f}<br><span style="font-size:13px; color:#94a3b8;">Range 25-75%: {low_est:,.0f} – {up_est:,.0f}</span></div>
                        <div class="summary-item">🛡️ <b>Alokasi:</b> {kelly_adj*100:.1f}% (Rp {alloc:,.0f})</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="action-card" style="border-left-color: {action_color};">
                        <div class="section-title">{action_icon} Rekomendasi Aksi</div>
                        <div class="summary-item" style="font-size: 16px; margin-top: 8px;">
                            {action_text}
                        </div>
                        <hr style="border-color: #334155; margin: 15px 0;">
                        <div style="color: #94a3b8; font-size: 14px;">
                            ⚠️ <i>Keputusan trading sepenuhnya ada di tangan Anda. Gunakan analisis ini sebagai konfirmasi tambahan.</i>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"🚨 Kesalahan: {str(e)}")
