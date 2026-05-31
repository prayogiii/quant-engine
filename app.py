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

# ====================== FALLBACK RSS PARSER ======================
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
# 1. KONFIGURASI HALAMAN & TEMA
# ==========================================
st.set_page_config(
    page_title="Quant Risk Engine Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    .positive { color: #10b981; }
    .negative { color: #ef4444; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Quant & Risk Engine Pro (Akurasi Tinggi)")
st.write("Algoritma kuantitatif + berita multi‑sumber + Backtesting. Distribusi Student‑t, ADX filter, Monte Carlo.")

if not SENTIMENT_AVAILABLE:
    st.warning("⚠️ NLTK tidak terpasang → sentimen tidak aktif.")
if not RSS_AVAILABLE:
    st.warning("⚠️ `feedparser` tidak terpasang → RSS tidak tersedia.")
if not TRANSLATOR_AVAILABLE:
    st.info("💡 `deep-translator` tidak terpasang → terjemahan tidak aktif.")

# ==========================================
# 2. PANEL INPUT
# ==========================================
ticker_raw = st.text_input(
    "Masukkan Kode Saham IHSG (Contoh: BRMS, BBRI, BMRI):",
    value=""
).upper().strip()

total_capital = st.number_input(
    "Total Modal Portofolio Anda (Rp):",
    min_value=0,
    value=None,
    step=10000,
    placeholder="Masukkan nominal modal anda..."
)

if total_capital is not None and total_capital > 0:
    rupiah_format = f"Rp {total_capital:,.0f}".replace(",", ".")
    st.markdown(f"✍️ *Terbaca:* **{rupiah_format}**")

# Proses ticker
if ticker_raw and not ticker_raw.endswith(".JK"):
    ticker_input = f"{ticker_raw}.JK"
else:
    ticker_input = ticker_raw

# ==================== FUNGSI ADX ====================
def compute_adx(df, period=14):
    """Hitung ADX untuk mengukur kekuatan tren."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.iloc[-1]

# ==================== FUNGSI BACKTEST ====================
def backtest_signal(df, signal_func, periods=126):
    """Backtest sinyal pada 6 bulan terakhir (~126 hari)."""
    df_back = df.iloc[-periods:].copy()
    signals = []
    for i in range(20, len(df_back)):
        slice_df = df.iloc[:df_back.index[i-1]+1]  # data sampai hari sebelumnya
        try:
            sig = signal_func(slice_df)
            signals.append((df_back.index[i], sig))
        except:
            continue
    # Hitung return jika mengikuti sinyal
    trades = []
    for i in range(len(signals)-1):
        date, sig = signals[i]
        next_date = signals[i+1][0]
        if sig == "🔥 STRONG BUY" or sig == "⚡ BUY (TACTICAL)":
            entry = df.loc[date, 'Close']
            exit_ = df.loc[next_date, 'Close']
            ret = (exit_ - entry) / entry
            trades.append(ret)
    if trades:
        win_rate = sum(1 for r in trades if r > 0) / len(trades)
        profit_factor = abs(sum(r for r in trades if r > 0) / sum(r for r in trades if r < 0)) if sum(r for r in trades if r < 0) != 0 else np.inf
        avg_return = np.mean(trades)
        return win_rate, profit_factor, avg_return, len(trades)
    return 0, 0, 0, 0

# ==================== FUNGSI BERITA ====================
def get_google_news_rss(query_str, num=5):
    if not RSS_AVAILABLE:
        return [], "RSS tidak tersedia"
    try:
        encoded = urllib.parse.quote(query_str)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)
        entries = feed.entries[:num]
        news = []
        for e in entries:
            title = e.get('title', '').strip()
            summary = e.get('summary', '').strip()
            summary = re.sub('<[^<]+?>', '', summary)
            news.append({'title': title, 'summary': summary, 'source': 'Google News'})
        return news, None
    except Exception as e:
        return [], str(e)

def get_yahoo_search_news(query_str, num=5):
    try:
        search = yf.Search(query_str)
        items = search.news or []
        news = []
        for item in items[:num]:
            inner = item.get('content') or item
            title = (inner.get('title') or inner.get('shortTitle') or 
                     inner.get('headline') or inner.get('summary') or '')
            summary = (inner.get('summary') or inner.get('longSummary') or 
                       inner.get('description') or '')
            if title:
                news.append({'title': title, 'summary': summary, 'source': 'Yahoo Search'})
        return news, None
    except Exception as e:
        return [], str(e)

def get_yahoo_ticker_news(ticker, num=5):
    try:
        t = yf.Ticker(ticker)
        items = t.news or []
        news = []
        for item in items[:num]:
            inner = item.get('content') or item
            title = (inner.get('title') or inner.get('shortTitle') or 
                     inner.get('headline') or inner.get('summary') or '')
            summary = (inner.get('summary') or inner.get('longSummary') or 
                       inner.get('description') or '')
            if title:
                news.append({'title': title, 'summary': summary, 'source': 'Yahoo Ticker'})
        return news, None
    except Exception as e:
        return [], str(e)

def filter_relevant(news_list, ticker):
    keywords = [ticker.lower(), 'saham', 'ihsg', 'bei', 'idx']
    filtered = []
    for n in news_list:
        text = (n['title'] + ' ' + n['summary']).lower()
        if any(k in text for k in keywords):
            filtered.append(n)
    return filtered if filtered else news_list

def analyze_sentiment_weighted(news_items, translator):
    """Sentimen berbobot: berita lebih baru (indeks kecil) berbobot lebih tinggi."""
    if not SENTIMENT_AVAILABLE or not news_items:
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    total_weight = 0
    weighted_sum = 0
    for i, item in enumerate(news_items):
        weight = 1 / (i + 1)  # berita pertama bobot 1, kedua 0.5, dst
        text = f"{item['title']}. {item['summary']}" if item['summary'] else item['title']
        # Terjemahkan ke Inggris jika perlu
        if any(ord(c) > 127 for c in text) and translator:
            try:
                text = translator.translate(text)
            except:
                pass
        score = analyzer.polarity_scores(text)['compound']
        weighted_sum += score * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else 0.0

# =====================================================

if st.button("JALANKAN QUANT ENGINE PRO + BACKTEST"):
    if not ticker_input:
        st.warning("⚠️ Kode saham tidak boleh kosong!")
    elif total_capital is None or total_capital <= 0:
        st.warning("⚠️ Modal portofolio harus diisi dan > Rp 0!")
    else:
        with st.spinner("🤖 Mengunduh data, berita, backtest, dan model kuantitatif..."):
            try:
                # ==========================================
                # 3. DATA HARGA
                # ==========================================
                df = yf.download(ticker_input, period="1y")
                if df.empty:
                    st.error("❌ Data tidak ditemukan.")
                    st.stop()

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                harga_terakhir = float(df['Close'].iloc[-1])
                returns = df['Close'].pct_change().dropna()
                if len(returns) < 20:
                    st.error("❌ Data historis kurang (minimal 20 hari).")
                    st.stop()

                # ==========================================
                # 4. BERITA MULTI‑SUMBER + SENTIMEN WEIGHTED
                # ==========================================
                news_pool = []
                translator_en = GoogleTranslator(source='auto', target='en') if TRANSLATOR_AVAILABLE else None
                translator_id = GoogleTranslator(source='auto', target='id') if TRANSLATOR_AVAILABLE else None

                rss_news, _ = get_google_news_rss(f"{ticker_raw} saham")
                if rss_news: news_pool.extend(rss_news)

                ysearch_news, _ = get_yahoo_search_news(f"{ticker_raw} saham")
                if ysearch_news: news_pool.extend(ysearch_news)

                if not news_pool:
                    yticker_news, _ = get_yahoo_ticker_news(ticker_input)
                    if yticker_news: news_pool.extend(yticker_news)

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
                translated_headlines = []
                for n in unique_news:
                    if TRANSLATOR_AVAILABLE and translator_id:
                        try:
                            translated_headlines.append(translator_id.translate(n['title']))
                        except:
                            translated_headlines.append("")
                    else:
                        translated_headlines.append("")

                if SENTIMENT_AVAILABLE:
                    if avg_sentiment >= 0.05:
                        sentimen_status = "Positif 🟢"
                    elif avg_sentiment <= -0.05:
                        sentimen_status = "Negatif 🔴"
                    else:
                        sentimen_status = "Netral ⚪"
                else:
                    sentimen_status = "Netral ⚪ (nonaktif)"

                # ==========================================
                # 5. VOLATILITAS ADAPTIF
                # ==========================================
                log_hl = np.log(df['High'] / df['Low']) ** 2
                parkinson_daily = np.sqrt(log_hl.mean() / (4 * np.log(2)))
                parkinson_vol = float(parkinson_daily * np.sqrt(252) * 100)

                ewma_lambda = 0.94
                ewma_var = returns.ewm(alpha=(1 - ewma_lambda), adjust=False).var().iloc[-1]
                ewma_vol = float(np.sqrt(ewma_var) * np.sqrt(252) * 100)

                # ==========================================
                # 6. DISTRIBUSI STUDENT‑T
                # ==========================================
                def t_loglike(params, data):
                    df, loc, scale = params
                    if df <= 2 or scale <= 0:
                        return np.inf
                    return -np.sum(student_t.logpdf(data, df, loc, scale))

                init_params = [5.0, returns.mean(), returns.std()]
                bounds = [(2.1, 100), (-0.1, 0.1), (1e-6, None)]
                res = minimize(t_loglike, init_params, args=(returns,), bounds=bounds, method='L-BFGS-B')
                if res.success:
                    df_est, t_loc, t_scale = res.x
                else:
                    df_est, t_loc, t_scale = 5.0, returns.mean(), returns.std()

                var_95_t = float(student_t.ppf(0.05, df_est, t_loc, t_scale) * 100)
                cvar_95_t = float(t_loc + t_scale * (student_t.pdf(student_t.ppf(0.05, df_est), df_est) / 0.05) *
                                  (df_est + student_t.ppf(0.05, df_est)**2) / (df_est - 1)) * 100

                ret_skew = float(skew(returns))
                ret_kurt = float(kurtosis(returns, fisher=True))

                # ==========================================
                # 7. BETA IHSG
                # ==========================================
                try:
                    df_ihsg = yf.download("^JKSE", period="1y")
                    if isinstance(df_ihsg.columns, pd.MultiIndex):
                        df_ihsg.columns = df_ihsg.columns.get_level_values(0)
                    ihsg_ret = df_ihsg['Close'].pct_change().dropna()
                    common_idx = returns.index.intersection(ihsg_ret.index)
                    if len(common_idx) > 20:
                        cov_mat = np.cov(returns.loc[common_idx], ihsg_ret.loc[common_idx])
                        beta_ihsg = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else 1.0
                    else:
                        beta_ihsg = 1.0
                except:
                    beta_ihsg = 1.0

                # ==========================================
                # 8. MOMENTUM & Z‑SCORE
                # ==========================================
                mom_3d = float((df['Close'].iloc[-1] / df['Close'].iloc[-4] - 1) * 100)
                mom_5d = float((df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100)
                mom_10d = float((df['Close'].iloc[-1] / df['Close'].iloc[-11] - 1) * 100)

                ma_20_close = float(df['Close'].tail(20).mean())
                std_20_close = float(df['Close'].tail(20).std())
                z_score = (harga_terakhir - ma_20_close) / std_20_close if std_20_close > 0 else 0.0

                # ==========================================
                # 9. REGIME + ADX FILTER
                # ==========================================
                ema_20 = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
                ema_50 = float(df['Close'].ewm(span=50, adjust=False).mean().iloc[-1])
                adx_val = compute_adx(df)

                mom_5d_hist = df['Close'].pct_change(5).dropna() * 100
                bullish_thresh = np.percentile(mom_5d_hist, 70)
                bearish_thresh = np.percentile(mom_5d_hist, 30)

                z_hist = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
                z_up = np.percentile(z_hist.dropna(), 80)
                z_down = np.percentile(z_hist.dropna(), 20)

                # Klasifikasi regime dengan ADX > 20 sebagai filter tren
                if adx_val > 20:
                    if harga_terakhir > ema_20 and ema_20 > ema_50:
                        if mom_5d > bullish_thresh or z_score > z_up:
                            regime_status = "Strong Bullish 🚀"
                            ihsg_status = "RISK-ON 🔥"
                        else:
                            regime_status = "Bullish 📈"
                            ihsg_status = "RISK-ON 🔥"
                    elif harga_terakhir < ema_20 and ema_20 < ema_50:
                        if mom_5d < bearish_thresh or z_score < z_down:
                            regime_status = "Panic Sell ⚠️"
                            ihsg_status = "RISK-OFF 🚨"
                        else:
                            regime_status = "Bearish 🔻"
                            ihsg_status = "RISK-OFF 🚨"
                    else:
                        regime_status = "Konsolidasi ↔️"
                        ihsg_status = "NEUTRAL ⚖️"
                else:
                    regime_status = "Sideways (ADX rendah) ↔️"
                    ihsg_status = "NEUTRAL ⚖️"

                # ==========================================
                # 10. PIVOT POINT
                # ==========================================
                last_high = float(df['High'].iloc[-1])
                last_low = float(df['Low'].iloc[-1])
                pp = (last_high + last_low + harga_terakhir) / 3
                r1 = (2 * pp) - last_low
                s1 = (2 * pp) - last_high
                r2 = pp + (last_high - last_low)
                s2 = pp - (last_high - last_low)

                res20 = float(df['High'].iloc[-21:-1].max())
                breakout_status = "YES (🔥)" if harga_terakhir > res20 else "NO"

                # ==========================================
                # 11. SIGNAL + BACKTEST
                # ==========================================
                score = 0
                if mom_3d > 0: score += 1
                if z_score < -1.5: score += 1
                if harga_terakhir > ema_20: score += 1
                if 'Volume' in df.columns and df['Volume'].iloc[-1] > df['Volume'].tail(20).mean():
                    score += 1
                if SENTIMENT_AVAILABLE and avg_sentiment > 0.2:
                    score += 1
                elif SENTIMENT_AVAILABLE and avg_sentiment < -0.2:
                    score -= 1

                if score >= 3:
                    signal = "🔥 STRONG BUY"
                elif score >= 2:
                    signal = "⚡ BUY (TACTICAL)"
                elif score == 1:
                    signal = "⏸️ HOLD / WAIT"
                else:
                    signal = "🚨 AVOID"

                # Backtest function inline
                def signal_func_backtest(df_slice):
                    # Simplifikasi untuk backtest
                    h = float(df_slice['Close'].iloc[-1])
                    r = df_slice['Close'].pct_change().dropna()
                    m3 = float((df_slice['Close'].iloc[-1] / df_slice['Close'].iloc[-4] - 1) * 100)
                    ema20 = float(df_slice['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
                    ema50 = float(df_slice['Close'].ewm(span=50, adjust=False).mean().iloc[-1])
                    z = (h - df_slice['Close'].tail(20).mean()) / df_slice['Close'].tail(20).std() if df_slice['Close'].tail(20).std() > 0 else 0
                    s = 0
                    if m3 > 0: s += 1
                    if z < -1.5: s += 1
                    if h > ema20: s += 1
                    if s >= 2: return "🔥 STRONG BUY"
                    elif s >= 1: return "⚡ BUY (TACTICAL)"
                    else: return "🚨 AVOID"

                win_rate_bt, profit_factor_bt, avg_ret_bt, trades_count = backtest_signal(df, signal_func_backtest)

                # ==========================================
                # 12. RISK METRICS
                # ==========================================
                roll_max = df['Close'].cummax()
                drawdown = (df['Close'] - roll_max) / roll_max
                max_dd = float(drawdown.min() * 100)
                max_dd_30d = float(drawdown.tail(30).min() * 100)

                mean_ret = returns.mean() * 252
                std_ret = returns.std() * np.sqrt(252)
                sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
                downside_ret = returns[returns < 0]
                downside_std = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 0 else std_ret
                sortino = mean_ret / downside_std if downside_std > 0 else 0.0
                calmar = mean_ret / abs(max_dd / 100) if max_dd != 0 else 0.0

                win_rate = len(returns[returns > 0]) / len(returns)
                avg_gain = returns[returns > 0].mean() if win_rate > 0 else 0.01
                avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.01
                win_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else 1.0
                kelly_raw = win_rate - ((1 - win_rate) / win_loss_ratio)
                skew_adj = 0.5 if ret_skew < -0.5 else 1.0
                kelly_adj = min(0.25, max(0.0, kelly_raw * 0.3 * skew_adj))  # capped 25%
                allocated_capital = total_capital * kelly_adj

                # ==========================================
                # 13. MONTE CARLO + EXPECTED SHORTFALL
                # ==========================================
                n_sim = 2000
                n_days = 30
                sim_innov = student_t.rvs(df_est, loc=t_loc, scale=t_scale, size=(n_days, n_sim))
                price_paths = harga_terakhir * np.exp(np.cumsum(sim_innov, axis=0))

                log_harga = np.log(df['Close'])
                log_lt_mean = log_harga.tail(20).mean()
                log_terakhir = np.log(harga_terakhir)
                theta = 1/20
                mu_ou_log = theta * (log_lt_mean - log_terakhir) + t_loc
                estimasi_close_besok = float(np.exp(log_terakhir + mu_ou_log))

                sim_h1 = student_t.rvs(df_est, loc=t_loc, scale=t_scale, size=2000)
                sim_prices_besok = harga_terakhir * np.exp(sim_h1)
                lower_est = float(np.percentile(sim_prices_besok, 25))
                upper_est = float(np.percentile(sim_prices_besok, 75))

                # Expected Shortfall 95% dari simulasi 30 hari
                final_prices = price_paths[-1, :]
                es_95_mc = float(np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)]))
                es_95_pct = (harga_terakhir - es_95_mc) / harga_terakhir * 100

                tp = r1
                sl = s1
                hit_tp_30d = (np.any(price_paths >= tp, axis=0).sum() / n_sim) * 100
                hit_sl_30d = (np.any(price_paths <= sl, axis=0).sum() / n_sim) * 100
                prob_bullish_besok = ((student_t.rvs(df_est, loc=t_loc, scale=t_scale, size=1000) > 0).sum() / 1000) * 100

                # ==========================================
                # 14. TAMPILAN DASHBOARD
                # ==========================================
                st.success(f"✅ Analisis Akurat: {ticker_input} | Harga: Rp {harga_terakhir:,.0f}".replace(",", "."))

                # --- BERITA ---
                st.header("📰 Sentimen Berita (Weighted)")
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Sentimen Agregat", f"{avg_sentiment:.2f}", sentimen_status)
                with c2:
                    st.markdown("**5 Berita Teratas:**")
                    if headlines:
                        for i, h in enumerate(headlines[:5]):
                            src = sources[i] if i < len(sources) else ""
                            t = translated_headlines[i] if i < len(translated_headlines) else ""
                            st.markdown(f"{i+1}. **{h}** <span class='source'>({src})</span>", unsafe_allow_html=True)
                            if t and t != h:
                                st.markdown(f"<span class='translated'>🇮🇩 {t}</span>", unsafe_allow_html=True)
                            st.markdown("")
                st.divider()

                # --- REGIME ---
                st.header("🧬 Regime & Volatility (ADX Filter)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Regime", regime_status)
                m2.metric("IHSG", ihsg_status)
                m3.metric("ADX", f"{adx_val:.1f}")
                st.markdown(f"EWMA Vol: `{ewma_vol:.2f}%` | Parkinson: `{parkinson_vol:.2f}%` | T (df={df_est:.1f}) | Skew: `{ret_skew:.2f}`")
                st.divider()

                # --- MOMENTUM ---
                st.header("📊 Momentum & Z‑Score")
                mo1, mo2, mo3, mo4 = st.columns(4)
                mo1.metric("Mom 3D", f"{mom_3d:+.2f}%")
                mo2.metric("Mom 5D", f"{mom_5d:+.2f}%")
                mo3.metric("Mom 10D", f"{mom_10d:+.2f}%")
                mo4.metric("Z‑Score", f"{z_score:+.2f}σ")
                st.divider()

                # --- PIVOT ---
                st.header("🎯 Pivot & S/R")
                st.write(f"Breakout Res20: `{breakout_status}`")
                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric("R2", f"Rp {r2:,.0f}".replace(",", "."))
                p2.metric("R1", f"Rp {r1:,.0f}".replace(",", "."))
                p3.metric("PP", f"Rp {pp:,.0f}".replace(",", "."))
                p4.metric("S1", f"Rp {s1:,.0f}".replace(",", "."))
                p5.metric("S2", f"Rp {s2:,.0f}".replace(",", "."))
                st.divider()

                # --- SIGNAL + BACKTEST ---
                st.header("🔮 Signal & Backtest 6 Bulan")
                tp1, tp2, tp3, tp4 = st.columns(4)
                tp1.metric("Signal", signal)
                tp2.metric("Est. Close Besok", f"Rp {estimasi_close_besok:,.0f}".replace(",", "."),
                           f"25-75%: {lower_est:,.0f} – {upper_est:,.0f}".replace(",", "."))
                tp3.metric("Entry (S1-PP)", f"Rp {s1:,.0f} - {pp:,.0f}".replace(",", "."))
                tp4.metric("Target (R1)", f"Rp {r1:,.0f}".replace(",", "."))
                st.caption("💡 Interval 25%–75% = rentang harga besok yang paling mungkin (50% probabilitas)")

                # Backtest result
                st.markdown("**📈 Backtest 6 Bulan Terakhir:**")
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Win Rate", f"{win_rate_bt:.1%}" if trades_count > 0 else "N/A")
                b2.metric("Profit Factor", f"{profit_factor_bt:.2f}" if trades_count > 0 else "N/A")
                b3.metric("Avg Return/Trade", f"{avg_ret_bt:.2%}" if trades_count > 0 else "N/A")
                b4.metric("Total Trades", f"{trades_count}")
                st.divider()

                # --- RISK ---
                st.header("🛡️ Risk & Portfolio Sizing")
                r1, r2, r3 = st.columns(3)
                r1.metric("Kelly Adj. (capped)", f"{kelly_adj*100:.1f}%")
                r2.metric("Rekom. Modal", f"Rp {allocated_capital:,.0f}".replace(",", "."))
                r3.metric("Beta vs IHSG", f"{beta_ihsg:.2f}x")
                st.markdown(f"Max DD: `{max_dd:.2f}%` | Sharpe: `{sharpe:.2f}` | Sortino: `{sortino:.2f}` | Calmar: `{calmar:.2f}`")
                st.markdown(f"VaR 95% (t): `{var_95_t:.2f}%` | CVaR 95% (t): `{cvar_95_t:.2f}%` | MC ES 95%: `{es_95_pct:.2f}%`")
                st.divider()

                # --- PROBABILITY ---
                st.header("🎲 Monte Carlo (Student‑t 2000)")
                pr1, pr2, pr3 = st.columns(3)
                pr1.metric("Prob Bullish Besok", f"{prob_bullish_besok:.1f}%")
                pr2.metric("Prob TP 30D", f"{hit_tp_30d:.1f}%")
                pr3.metric("Prob SL 30D", f"{hit_sl_30d:.1f}%")

            except Exception as e:
                st.error(f"🚨 Kesalahan: {str(e)}")
