import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, t as student_t
from scipy.optimize import minimize
import warnings
import urllib.parse
import re

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
# 1. KONFIGURASI HALAMAN & STYLING
# ==========================================
st.set_page_config(
    page_title="QuantRisk Pro v3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0a0e1a 0%, #121a2f 100%); }
    
    .stButton>button {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #00ffcc; border: 1px solid #334155; border-radius: 12px;
        font-weight: 600; padding: 10px 20px; transition: all 0.2s;
    }
    .stButton>button:hover { border-color: #00ffcc; box-shadow: 0 0 15px rgba(0,255,204,0.3); transform: scale(1.02); }
    .stTextInput>div>div>input {
        background: rgba(15, 23, 42, 0.8); border: 1px solid #334155;
        border-radius: 12px; color: white; padding: 12px;
    }
    .neo-card {
        background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(20px);
        border: 1px solid rgba(51, 65, 85, 0.5); border-radius: 20px;
        padding: 20px; margin: 10px 0;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    .action-card {
        background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(20px);
        border-radius: 20px; padding: 24px; margin: 10px 0;
        border-left: 5px solid; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.4);
    }
    .section-title {
        font-size: 20px; font-weight: 600; margin-bottom: 15px;
        color: #f1f5f9; display: flex; align-items: center; gap: 10px;
    }
    hr { border-color: #334155; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 📊 QuantRisk Pro")
    st.markdown("Engine analisis kuantitatif saham Indonesia.")
    ticker_raw = st.text_input(
        "🔍 Kode Saham",
        value="BBRI",
        placeholder="Contoh: BBRI, TLKM, BMRI"
    ).upper().strip()
    if ticker_raw and not ticker_raw.endswith(".JK"):
        ticker_input = f"{ticker_raw}.JK"
    else:
        ticker_input = ticker_raw
    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("🚀 ANALISIS", use_container_width=True)
    with col2:
        if st.button("🗑️ Reset", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache dibersihkan!")
    st.markdown("---")
    st.caption("Data: Yahoo Finance. Bukan rekomendasi investasi.")

if not run_btn:
    st.info("👈 Masukkan kode saham di sidebar lalu klik **ANALISIS**")
    st.stop()

if not ticker_input:
    st.warning("⚠️ Kode saham tidak boleh kosong!")
    st.stop()

# ==================== FUNGSI DATA & INDIKATOR ====================
@st.cache_data(ttl=3600)
def load_stock_data(ticker):
    df = yf.download(ticker, period="2y")
    if df.empty: return pd.DataFrame()
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
    up, down = high.diff(), -low.diff()
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

# ==================== PROSES ANALISIS ====================
with st.spinner("🔄 Mengunduh dan menganalisis data..."):
    df = load_stock_data(ticker_input)
    if df.empty:
        st.error("❌ Data tidak ditemukan. Periksa kode atau coba lagi.")
        st.stop()
    harga_terakhir = float(df['Close'].iloc[-1])
    returns = df['Close'].pct_change().dropna()
    if len(returns) < 50:
        st.error("❌ Data historis kurang.")
        st.stop()

    # Indikator
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ADX'] = compute_adx_series(df)
    df['Mom5D'] = df['Close'].pct_change(5) * 100
    df['ZScore'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean() if 'Volume' in df.columns else 0

    # Fundamental
    try: ticker_info = yf.Ticker(ticker_input).info
    except: ticker_info = {}

    # Berita & Sentimen (fungsi sama persis, diringkas panggilannya)
    def get_google_news_rss(query, num=5):
        if not RSS_AVAILABLE: return [], ""
        try:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=id&gl=ID&ceid=ID:id"
            feed = feedparser.parse(url)
            return [{'title': e.get('title','').strip(), 'summary': re.sub('<[^<]+?>', '', e.get('summary','')),
                     'source': 'Google News'} for e in feed.entries[:num]], None
        except Exception as e: return [], str(e)

    def get_yahoo_search_news(query, num=5):
        try:
            items = (yf.Search(query).news or [])[:num]
            return [{'title': (c.get('title') or c.get('shortTitle') or c.get('headline') or ''),
                     'summary': (c.get('summary') or c.get('longSummary') or c.get('description') or ''),
                     'source': 'Yahoo Search'} for item in items if (c:=item.get('content') or item)], None
        except: return [], "Yahoo Search gagal"

    news_pool = []
    translator_en = GoogleTranslator(source='auto', target='en') if TRANSLATOR_AVAILABLE else None
    translator_id = GoogleTranslator(source='auto', target='id') if TRANSLATOR_AVAILABLE else None
    rss, _ = get_google_news_rss(f"{ticker_raw} saham")
    if rss: news_pool.extend(rss)
    ysearch, _ = get_yahoo_search_news(f"{ticker_raw} saham")
    if ysearch: news_pool.extend(ysearch)
    keywords = [ticker_raw.lower(), 'saham', 'ihsg', 'bei', 'idx']
    news_pool = [n for n in news_pool if any(k in (n['title']+n['summary']).lower() for k in keywords)] or news_pool
    seen = set()
    unique_news = []
    for n in news_pool:
        if n['title'] not in seen:
            seen.add(n['title']); unique_news.append(n)
        if len(unique_news) >= 5: break

    if SENTIMENT_AVAILABLE and unique_news:
        analyzer = SentimentIntensityAnalyzer()
        total_w, w_sum = 0, 0
        for i, item in enumerate(unique_news):
            text = f"{item['title']}. {item['summary']}" if item['summary'] else item['title']
            if any(ord(c) > 127 for c in text) and translator_en:
                try: text = translator_en.translate(text)
                except: pass
            score = analyzer.polarity_scores(text)['compound']
            weight = 1/(i+1)
            w_sum += score*weight; total_w += weight
        avg_sentiment = w_sum/total_w if total_w else 0.0
    else:
        avg_sentiment = 0.0

    sentimen_status = "Positif 🟢" if avg_sentiment >= 0.05 else ("Negatif 🔴" if avg_sentiment <= -0.05 else "Netral ⚪")

    # Threshold & Distribusi
    split_idx = max(126, len(df)-126)
    df_thresh = df.iloc[:split_idx]
    returns_thresh = df_thresh['Close'].pct_change().dropna()
    adx_threshold = np.percentile(df_thresh['ADX'].dropna(), 75) if not df_thresh['ADX'].dropna().empty else 20
    z_oversold_th = -1.5
    mom_median_th = np.percentile(df_thresh['Mom5D'].dropna(), 50) if not df_thresh['Mom5D'].dropna().empty else 0.0

    def t_loglike(p, d):
        if p[0] <= 2 or p[2] <= 0: return np.inf
        return -np.sum(student_t.logpdf(d, p[0], p[1], p[2]))
    res = minimize(t_loglike, [5, returns_thresh.mean(), returns_thresh.std()],
                   bounds=[(2.1,100), (-0.1,0.1), (1e-6, None)], args=(returns_thresh,), method='L-BFGS-B')
    df_est, t_loc, t_scale = res.x if res.success else (5, returns_thresh.mean(), returns_thresh.std())

    # Regime
    def get_regime_row(row):
        h, e20, e50, a, z, m = row['Close'], row['EMA20'], row['EMA50'], row['ADX'], row['ZScore'], row['Mom5D']
        if a > adx_threshold:
            if h > e20 and e20 > e50:
                return ("Strong Bullish 🚀","RISK-ON 🔥") if (m > mom_median_th or z > z_oversold_th) else ("Bullish 📈","RISK-ON 🔥")
            elif h < e20 and e20 < e50:
                return ("Panic Sell 🚨","RISK-OFF 🛑") if (m < mom_median_th or z < z_oversold_th) else ("Bearish 🔻","RISK-OFF 🛑")
            elif h > e20 and e20 < e50: return ("Early Recovery 🔄","TRANSISI ⚠️")
            elif h < e20 and e20 > e50: return ("Distribution 📉","TRANSISI ⚠️")
            else: return ("Konsolidasi Tren ↔️","NEUTRAL ⚖️")
        else:
            if h > e20 and e20 > e50: return ("Bullish Accumulation 🏗️","NEUTRAL ⚖️")
            elif h < e20 and e20 < e50: return ("Bearish Accumulation 🧊","NEUTRAL ⚖️")
            elif h > e20 and e20 < e50: return ("Sideways Bias Naik ↗️","NEUTRAL ⚖️")
            elif h < e20 and e20 > e50: return ("Sideways Bias Turun ↘️","NEUTRAL ⚖️")
            else: return ("Sideways Normal ↔️","NEUTRAL ⚖️")

    regime, ihsg_cond = get_regime_row(df.iloc[-1])
    adx = df['ADX'].iloc[-1]

    # Beta
    try:
        ihsg = load_ihsg_data()
        ihsg_ret = ihsg['Close'].pct_change().dropna()
        common = returns.index.intersection(ihsg_ret.index)
        beta_ihsg = (np.cov(returns.loc[common], ihsg_ret.loc[common])[0,1] / np.var(ihsg_ret.loc[common])) if len(common)>20 else 1.0
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
        score += ((dataframe['Close'] > dataframe['EMA20']) & (dataframe['EMA20'] > dataframe['EMA50'])).astype(int)*2
        score += (dataframe['Mom5D'] > mom_th).astype(int)
        if 'Volume' in dataframe.columns:
            score += (dataframe['Volume'] > dataframe['Vol_MA20']).astype(int)
        sig = pd.Series("🚨 AVOID", index=dataframe.index)
        sig[score == 1] = "⏸️ HOLD / WAIT"
        sig[score >= 2] = "⚡ BUY (TACTICAL)"
        sig[score >= 3] = "🔥 STRONG BUY"
        sig[(dataframe['ADX'] < 20) & sig.str.contains("BUY")] = "⏸️ HOLD / WAIT"
        sig[(dataframe['ZScore'] < -1.5) & (dataframe['Close'] < dataframe['EMA20'])] = "⚡ BUY (TACTICAL)"
        return sig

    df['Signal'] = generate_signals_vectorized(df, mom_median_th)
    signal = df['Signal'].iloc[-1]

    # Backtest
    df_back = df.iloc[-126:].copy()
    trades, daily_returns = [], []
    in_pos, entry_price = False, 0.0
    for i in range(len(df_back)):
        curr_sig = df_back['Signal'].iloc[i]
        curr_close = float(df_back['Close'].iloc[i])
        prev_close = float(df_back['Close'].iloc[i-1]) if i>0 else curr_close
        if in_pos:
            daily_returns.append((curr_close - prev_close)/prev_close if prev_close else 0)
            if "AVOID" in curr_sig or i == len(df_back)-1:
                trades.append((curr_close - entry_price)/entry_price)
                in_pos = False
        else:
            daily_returns.append(0.0)
            if "BUY" in curr_sig:
                in_pos, entry_price = True, curr_close

    if trades:
        win_bt = sum(1 for r in trades if r>0)/len(trades)
        loss = [r for r in trades if r<0]; profit = [r for r in trades if r>0]
        pf_bt = abs(sum(profit)/sum(loss)) if loss else np.inf
        avg_bt = np.mean(trades)
        equity = np.cumprod([1+r for r in trades])
        max_dd_bt = float(np.min(equity/np.maximum.accumulate(equity)-1)*100) if len(equity) else 0
        daily_ret = np.array(daily_returns)
        sharpe_bt = (daily_ret.mean()/daily_ret.std())*np.sqrt(252) if daily_ret.std() else 0
        trades_bt = len(trades)
    else:
        win_bt = pf_bt = avg_bt = max_dd_bt = sharpe_bt = trades_bt = 0

    # Kelly
    roll_max_th = df_thresh['Close'].cummax()
    dd_th = (df_thresh['Close'] - roll_max_th)/roll_max_th
    max_dd = float(dd_th.min()*100)
    max_dd_30 = float(dd_th.tail(30).min()*100) if len(dd_th)>=30 else max_dd
    if trades_bt >= 2:
        win_r, avg_g, avg_l = win_bt, np.mean(profit) if profit else 0.01, abs(np.mean(loss)) if loss else 0.01
    else:
        win_r = len(returns_thresh[returns_thresh>0])/len(returns_thresh)
        avg_g = returns_thresh[returns_thresh>0].mean() if win_r else 0.01
        avg_l = abs(returns_thresh[returns_thresh<0].mean()) if len(returns_thresh[returns_thresh<0]) else 0.01
    wl = avg_g/avg_l if avg_l else 1
    kelly_raw = win_r - (1-win_r)/wl
    ret_skew = float(skew(returns_thresh))
    ret_kurt = float(kurtosis(returns_thresh, fisher=True))
    kelly_adj = min(0.25, max(0.0, kelly_raw*0.3*(0.5 if ret_skew<-0.5 else 1)*(0.5 if ret_kurt>3 else 1)))

    # Monte Carlo OU
    n_sim, n_days = 2000, 30
    latest_vol = np.sqrt(df['Close'].pct_change().ewm(alpha=0.06).var().iloc[-1])
    scale_corrected = latest_vol / np.sqrt(df_est/(df_est-2)) if df_est>2 else latest_vol
    def estimate_theta_ou(close):
        lp = np.log(close.dropna()); diff = lp.diff().dropna(); lag = lp.shift(1).dropna()
        idx = diff.index.intersection(lag.index)
        if len(idx)<20: return 0.05
        X = np.vstack([np.ones(len(idx)), lag.loc[idx]]).T
        theta = -np.linalg.lstsq(X, diff.loc[idx], rcond=None)[0][1]
        return theta if theta>0 else 0.05
    theta_ou = estimate_theta_ou(df['Close'])
    locked_log_mean20 = np.log(df['Close']).tail(20).mean()
    paths = np.zeros((n_days, n_sim))
    current_log = np.ones((1, n_sim))*np.log(harga_terakhir)
    for day in range(n_days):
        inov = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=n_sim)
        next_log = current_log[-1] + theta_ou*(locked_log_mean20 - current_log[-1]) + inov
        current_log = np.vstack([current_log, next_log])
        paths[day] = np.exp(next_log)
    mu_ou = theta_ou * (locked_log_mean20 - np.log(harga_terakhir))
    est_besok = float(np.exp(np.log(harga_terakhir) + mu_ou))
    sim_h1 = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=2000)
    prices_besok = harga_terakhir * np.exp(mu_ou + sim_h1)
    low_est, up_est = float(np.percentile(prices_besok, 25)), float(np.percentile(prices_besok, 75))
    hit_tp = (np.any(paths >= r1, axis=0).sum()/n_sim)*100
    hit_sl = (np.any(paths <= s2, axis=0).sum()/n_sim)*100
    prob_bull = ((mu_ou + sim_h1 > 0).sum()/2000)*100

    # TP/SL persentase
    tp_pct = ((r1-harga_terakhir)/harga_terakhir)*100 if harga_terakhir else 0
    sl_pct = ((harga_terakhir-s2)/harga_terakhir)*100 if harga_terakhir else 0
    rrr = tp_pct/sl_pct if sl_pct else 0
    rrr_status = "Ideal 🟢" if rrr>=1.5 else ("Cukup 🟡" if rrr>=1 else "Buruk 🔴")

# ==================== TAMPILAN UTAMA ====================
st.title("📊 Quant & Risk Intelligence Engine")
st.caption(f"Analisis terintegrasi untuk **{ticker_input}** | Harga terakhir: **Rp {harga_terakhir:,.0f}**".replace(",", "."))

# ---------- ROW 1: SINYAL & METRIK UTAMA ----------
col_signal, col_est, col_prob = st.columns([2, 1, 1])
with col_signal:
    signal_color = {"🔥 STRONG BUY": "#10b981", "⚡ BUY (TACTICAL)": "#f59e0b",
                    "⏸️ HOLD / WAIT": "#3b82f6", "🚨 AVOID": "#ef4444"}.get(signal, "#ef4444")
    st.markdown(f"""
    <div class="neo-card" style="border-left: 5px solid {signal_color};">
        <h2 style="color:{signal_color}; margin:0;">{signal}</h2>
        <p style="color:#cbd5e1; margin:5px 0 0 0;">
            Rezim: <b>{regime}</b> | Makro: <b>{ihsg_cond}</b> | ADX: {adx:.1f}
        </p>
    </div>
    """, unsafe_allow_html=True)
with col_est:
    st.markdown(f"""
    <div class="neo-card">
        <div class="section-title">📆 Estimasi Besok</div>
        <h3 style="color:#00ffcc; margin:0;">Rp {est_besok:,.0f}</h3>
        <p style="color:#94a3b8;">50% range: Rp {low_est:,.0f} - {up_est:,.0f}</p>
    </div>
    """.replace(",", "."), unsafe_allow_html=True)
with col_prob:
    st.markdown(f"""
    <div class="neo-card">
        <div class="section-title">🎲 Prob 30 Hari</div>
        <p style="color:#10b981;">Naik: {prob_bull:.1f}%</p>
        <p style="color:#f59e0b;">Sentuh R1: {hit_tp:.1f}%</p>
        <p style="color:#ef4444;">Sentuh S2: {hit_sl:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- ROW 2: TARGET & RRR ----------
col_tp, col_sl, col_rrr = st.columns(3)
col_tp.metric("Take Profit (R1)", f"Rp {r1:,.0f} (+{tp_pct:.1f}%)".replace(",", "."))
col_sl.metric("Stop Loss (S2)", f"Rp {s2:,.0f} (-{sl_pct:.1f}%)".replace(",", "."))
col_rrr.metric("Risk:Reward Ratio", f"1:{rrr:.2f}", rrr_status)

# ---------- ROW 3: REKOMENDASI AKSI ----------
if rrr < 1.0 and "BUY" in signal:
    action_color, action_icon, action_text = "#ef4444", "⚠️", "Tunda entry. RRR buruk, tunggu koreksi ke S1."
elif "STRONG BUY" in signal:
    action_color, action_icon, action_text = "#10b981", "🟢", f"Pembelian agresif, maks {kelly_adj*100:.1f}% portofolio. SL di Rp {s2:,.0f}."
elif "BUY" in signal:
    action_color, action_icon, action_text = "#f59e0b", "🟡", f"Buy on weakness di Rp {s1:,.0f}-{pp:,.0f}. SL Rp {s2:,.0f}."
elif "HOLD" in signal:
    action_color, action_icon, action_text = "#3b82f6", "🔵", "Tahan posisi, jangan tambah. Pantau batas SL."
else:
    action_color, action_icon, action_text = "#ef4444", "🔴", "Hindari / Likuidasi. Amankan modal."

st.markdown(f"""
<div class="action-card" style="border-left-color: {action_color};">
    <h3 style="margin:0;">{action_icon} Panduan Eksekusi</h3>
    <p style="color:#cbd5e1; margin:10px 0 0 0;">{action_text}</p>
</div>
""", unsafe_allow_html=True)

# ---------- CHART ----------
if PLOTLY_AVAILABLE:
    st.markdown("<div class='section-title'>📈 Chart Harga & Sinyal</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#00ffcc', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', line=dict(color='#f59e0b', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50', line=dict(color='#ef4444', dash='dot')))
    buy_df = df_back[df_back['Signal'].str.contains("BUY")]
    fig.add_trace(go.Scatter(x=buy_df.index, y=buy_df['Close'], mode='markers',
                             marker=dict(symbol='triangle-up', size=10, color='#10b981'),
                             name='Buy Signal'))
    for lvl, name, clr in [(r1,'R1','orange'), (s1,'S1','red'), (pp,'PP','gray')]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=clr, annotation_text=name, annotation_position="right")
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ---------- DETAIL DALAM EXPANDER ----------
with st.expander("📰 Berita & Sentimen", expanded=False):
    col1, col2 = st.columns([1,2])
    col1.metric("Sentimen", f"{avg_sentiment:.2f}", sentimen_status)
    for i, h in enumerate(unique_news):
        src = h['source']
        trans = translated[i] if i < len(translated) else ""
        st.markdown(f"{i+1}. **{h['title']}** ({src})")
        if trans and trans != h['title']:
            st.caption(f"🇮🇩 {trans}")

with st.expander("🏢 Fundamental", expanded=False):
    if ticker_info:
        mc = ticker_info.get('marketCap')
        per = ticker_info.get('trailingPE') or ticker_info.get('forwardPE')
        pbv = ticker_info.get('priceToBook')
        roe = ticker_info.get('returnOnEquity')
        de = ticker_info.get('debtToEquity')
        st.markdown(f"""
        | Metrik | Nilai |
        |--------|-------|
        | Market Cap | {mc:,.0f} IDR |
        | PER | {per:.2f}x |
        | PBV | {pbv:.2f}x |
        | ROE | {roe*100:.1f}% |
        | D/E | {de:.2f}% |
        """)
    else:
        st.warning("Data tidak tersedia.")

with st.expander("⚖️ Backtest & Risiko", expanded=False):
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Win Rate", f"{win_bt:.1%}" if trades_bt else "N/A")
    c2.metric("Profit Factor", f"{pf_bt:.2f}" if trades_bt else "N/A")
    c3.metric("Avg Return/Trade", f"{avg_bt:.2%}" if trades_bt else "N/A")
    c4.metric("Max DD Strat", f"{max_dd_bt:.2f}%" if trades_bt else "N/A")
    c5.metric("Sharpe", f"{sharpe_bt:.2f}" if trades_bt else "N/A")
    c6.metric("Total Trades", trades_bt)
    st.write(f"**Alokasi Kelly:** {kelly_adj*100:.1f}% | **Beta IHSG:** {beta_ihsg:.2f}x")
    st.write(f"Max DD Historis: {max_dd:.2f}% | DD 30 Hari: {max_dd_30:.2f}%")
