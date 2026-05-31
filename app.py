import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, t as student_t
from scipy.optimize import minimize
import warnings
import urllib.parse
import re

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
    </style>
""", unsafe_allow_html=True)

st.title("📊 Quant & Risk Engine Pro (Akurasi Tinggi)")
st.write("Algoritma kuantitatif + berita multi‑sumber + Backtest. Distribusi Student‑t, ADX filter, Monte Carlo.")
if not SENTIMENT_AVAILABLE: st.warning("⚠️ NLTK tidak terpasang")
if not RSS_AVAILABLE: st.warning("⚠️ feedparser tidak terpasang")
if not TRANSLATOR_AVAILABLE: st.info("💡 deep-translator tidak terpasang")

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

def backtest_signal(df, signal_func, periods=126):
    """Backtest dengan slicing integer, aman untuk semua versi pandas."""
    df_back = df.iloc[-periods:].copy()
    signals = []
    # Kita butuh minimal 20 hari untuk indikator, mulai dari indeks ke-20
    for i in range(20, len(df_back)):
        # Data hingga hari ke-i (exclusive untuk sinyal, pakai data sampai i-1)
        slice_df = df.iloc[:df.index.get_loc(df_back.index[i-1])+1]  # aman, integer
        try:
            sig = signal_func(slice_df)
            signals.append((df_back.index[i], sig))
        except:
            continue
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
        return win_rate, profit_factor, avg_return, len(trades)
    return 0, 0, 0, 0
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

                # BERITA MULTI‑SUMBER
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

                # DISTRIBUSI T
                def t_loglike(p, d):
                    if p[0]<=2 or p[2]<=0: return np.inf
                    return -np.sum(student_t.logpdf(d, p[0], p[1], p[2]))
                res = minimize(t_loglike, [5, returns.mean(), returns.std()], bounds=[(2.1,100),(-0.1,0.1),(1e-6,None)], args=(returns,), method='L-BFGS-B')
                df_est, t_loc, t_scale = res.x if res.success else (5, returns.mean(), returns.std())
                var_95_t = float(student_t.ppf(0.05, df_est, t_loc, t_scale)*100)
                cvar_95_t = float(t_loc + t_scale * (student_t.pdf(student_t.ppf(0.05, df_est), df_est)/0.05) * (df_est+student_t.ppf(0.05, df_est)**2)/(df_est-1))*100
                ret_skew = float(skew(returns))
                ret_kurt = float(kurtosis(returns, fisher=True))

                # BETA
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

                # MOMENTUM
                mom_3d = float((df['Close'].iloc[-1]/df['Close'].iloc[-4]-1)*100)
                mom_5d = float((df['Close'].iloc[-1]/df['Close'].iloc[-6]-1)*100)
                mom_10d = float((df['Close'].iloc[-1]/df['Close'].iloc[-11]-1)*100)
                ma20 = float(df['Close'].tail(20).mean())
                std20 = float(df['Close'].tail(20).std())
                z_score = (harga_terakhir-ma20)/std20 if std20>0 else 0.0

                # REGIME + ADX
                ema20 = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
                ema50 = float(df['Close'].ewm(span=50, adjust=False).mean().iloc[-1])
                adx = compute_adx(df)
                mom5_hist = df['Close'].pct_change(5).dropna()*100
                bull_th = np.percentile(mom5_hist, 70)
                bear_th = np.percentile(mom5_hist, 30)
                z_hist = (df['Close']-df['Close'].rolling(20).mean())/df['Close'].rolling(20).std()
                z_up = np.percentile(z_hist.dropna(), 80)
                z_down = np.percentile(z_hist.dropna(), 20)
                if adx > 20:
                    if harga_terakhir > ema20 and ema20 > ema50:
                        regime = "Strong Bullish 🚀" if (mom_5d > bull_th or z_score > z_up) else "Bullish 📈"
                        ihsg_cond = "RISK-ON 🔥"
                    elif harga_terakhir < ema20 and ema20 < ema50:
                        regime = "Panic Sell ⚠️" if (mom_5d < bear_th or z_score < z_down) else "Bearish 🔻"
                        ihsg_cond = "RISK-OFF 🚨"
                    else:
                        regime = "Konsolidasi ↔️"; ihsg_cond = "NEUTRAL ⚖️"
                else:
                    regime = "Sideways (ADX rendah) ↔️"; ihsg_cond = "NEUTRAL ⚖️"

                # PIVOT
                hi, lo = float(df['High'].iloc[-1]), float(df['Low'].iloc[-1])
                pp = (hi+lo+harga_terakhir)/3
                r1, s1 = 2*pp-lo, 2*pp-hi
                r2, s2 = pp+(hi-lo), pp-(hi-lo)
                res20 = float(df['High'].iloc[-21:-1].max())
                breakout = "YES (🔥)" if harga_terakhir > res20 else "NO"

                # SIGNAL
                score = 0
                if mom_3d>0: score+=1
                if z_score<-1.5: score+=1
                if harga_terakhir>ema20: score+=1
                if 'Volume' in df.columns and df['Volume'].iloc[-1] > df['Volume'].tail(20).mean(): score+=1
                if SENTIMENT_AVAILABLE and avg_sentiment>0.2: score+=1
                elif SENTIMENT_AVAILABLE and avg_sentiment<-0.2: score-=1
                signal = "🔥 STRONG BUY" if score>=3 else ("⚡ BUY (TACTICAL)" if score>=2 else ("⏸️ HOLD / WAIT" if score==1 else "🚨 AVOID"))

                # BACKTEST (FUNGSI SIGNAL SEDERHANA)
                def sig_func(sl):
                    h = float(sl['Close'].iloc[-1])
                    m3 = float((sl['Close'].iloc[-1]/sl['Close'].iloc[-4]-1)*100)
                    ema20s = float(sl['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
                    zs = (h-sl['Close'].tail(20).mean())/sl['Close'].tail(20).std() if sl['Close'].tail(20).std()>0 else 0
                    s = 0
                    if m3>0: s+=1
                    if zs<-1.5: s+=1
                    if h>ema20s: s+=1
                    return "🔥 STRONG BUY" if s>=2 else ("⚡ BUY (TACTICAL)" if s>=1 else "🚨 AVOID")
                win_bt, pf_bt, avg_bt, trades_bt = backtest_signal(df, sig_func)

                # RISK METRICS
                roll_max = df['Close'].cummax()
                drawdown = (df['Close']-roll_max)/roll_max
                max_dd = float(drawdown.min()*100)
                max_dd_30 = float(drawdown.tail(30).min()*100)
                mean_ret = returns.mean()*252
                std_ret = returns.std()*np.sqrt(252)
                sharpe = mean_ret/std_ret if std_ret>0 else 0
                down_std = returns[returns<0].std()*np.sqrt(252) if len(returns[returns<0])>0 else std_ret
                sortino = mean_ret/down_std if down_std>0 else 0
                calmar = mean_ret/abs(max_dd/100) if max_dd!=0 else 0
                win_r = len(returns[returns>0])/len(returns)
                avg_g = returns[returns>0].mean() if win_r>0 else 0.01
                avg_l = abs(returns[returns<0].mean()) if len(returns[returns<0])>0 else 0.01
                wl = avg_g/avg_l if avg_l>0 else 1
                kelly_raw = win_r - (1-win_r)/wl
                kelly_adj = min(0.25, max(0.0, kelly_raw*0.3*(0.5 if ret_skew<-0.5 else 1.0)))
                alloc = total_capital*kelly_adj

                # MONTE CARLO
                n_sim, n_days = 2000, 30
                sim_innov = student_t.rvs(df_est, loc=t_loc, scale=t_scale, size=(n_days, n_sim))
                paths = harga_terakhir * np.exp(np.cumsum(sim_innov, axis=0))
                log_close = np.log(df['Close'])
                log_mean20 = log_close.tail(20).mean()
                log_last = np.log(harga_terakhir)
                mu_ou = (1/20)*(log_mean20-log_last) + t_loc
                est_besok = float(np.exp(log_last+mu_ou))
                sim_h1 = student_t.rvs(df_est, loc=t_loc, scale=t_scale, size=2000)
                prices_besok = harga_terakhir * np.exp(sim_h1)
                low_est, up_est = float(np.percentile(prices_besok,25)), float(np.percentile(prices_besok,75))
                final_prices = paths[-1,:]
                es_95_mc = float(np.mean(final_prices[final_prices<=np.percentile(final_prices,5)]))
                es_95_pct = (harga_terakhir-es_95_mc)/harga_terakhir*100
                tp, sl = r1, s1
                hit_tp = (np.any(paths>=tp, axis=0).sum()/n_sim)*100
                hit_sl = (np.any(paths<=sl, axis=0).sum()/n_sim)*100
                prob_bull = ((sim_h1>0).sum()/2000)*100

                # TAMPILAN
                st.success(f"✅ Analisis: {ticker_input} | Harga: Rp {harga_terakhir:,.0f}".replace(",","."))
                st.header("📰 Sentimen Berita (Weighted)")
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
                st.header("🧬 Regime & Volatility (ADX)")
                m1,m2,m3=st.columns(3)
                m1.metric("Regime", regime); m2.metric("IHSG", ihsg_cond); m3.metric("ADX", f"{adx:.1f}")
                st.markdown(f"EWMA Vol: `{ewma_vol:.2f}%` | Parkinson: `{parkinson_vol:.2f}%` | T(df={df_est:.1f}) | Skew: `{ret_skew:.2f}`")
                st.divider()
                st.header("📊 Momentum & Z‑Score")
                mo1,mo2,mo3,mo4=st.columns(4)
                mo1.metric("3D", f"{mom_3d:+.2f}%"); mo2.metric("5D", f"{mom_5d:+.2f}%"); mo3.metric("10D", f"{mom_10d:+.2f}%"); mo4.metric("Z", f"{z_score:+.2f}σ")
                st.divider()
                st.header("🎯 Pivot & S/R")
                st.write(f"Breakout Res20: `{breakout}`")
                p1,p2,p3,p4,p5=st.columns(5)
                p1.metric("R2", f"Rp {r2:,.0f}".replace(",",".")); p2.metric("R1", f"Rp {r1:,.0f}".replace(",",".")); p3.metric("PP", f"Rp {pp:,.0f}".replace(",",".")); p4.metric("S1", f"Rp {s1:,.0f}".replace(",",".")); p5.metric("S2", f"Rp {s2:,.0f}".replace(",","."))
                st.divider()
                st.header("🔮 Signal & Backtest 6 Bulan")
                t1,t2,t3,t4=st.columns(4)
                t1.metric("Signal", signal)
                t2.metric("Est. Besok", f"Rp {est_besok:,.0f}".replace(",","."), f"25-75%: {low_est:,.0f} – {up_est:,.0f}".replace(",","."))
                t3.metric("Entry", f"Rp {s1:,.0f} - {pp:,.0f}".replace(",","."))
                t4.metric("Target", f"Rp {r1:,.0f}".replace(",","."))
                st.caption("💡 Interval 25%-75% = rentang paling mungkin (50% probabilitas)")
                st.markdown("**📈 Backtest 6 Bulan:**")
                b1,b2,b3,b4=st.columns(4)
                b1.metric("Win Rate", f"{win_bt:.1%}" if trades_bt else "N/A")
                b2.metric("Profit Factor", f"{pf_bt:.2f}" if trades_bt else "N/A")
                b3.metric("Avg Return", f"{avg_bt:.2%}" if trades_bt else "N/A")
                b4.metric("Trades", f"{trades_bt}")
                st.divider()
                st.header("🛡️ Risk & Portfolio Sizing")
                r1,r2,r3=st.columns(3)
                r1.metric("Kelly Adj.", f"{kelly_adj*100:.1f}%")
                r2.metric("Rekom. Modal", f"Rp {alloc:,.0f}".replace(",","."))
                r3.metric("Beta", f"{beta_ihsg:.2f}x")
                st.markdown(f"Max DD: `{max_dd:.2f}%` (30D: `{max_dd_30:.2f}%`) | Sharpe: `{sharpe:.2f}` | Sortino: `{sortino:.2f}` | Calmar: `{calmar:.2f}`")
                st.markdown(f"VaR 95% (t): `{var_95_t:.2f}%` | CVaR 95% (t): `{cvar_95_t:.2f}%` | MC ES 95%: `{es_95_pct:.2f}%`")
                st.divider()
                st.header("🎲 Monte Carlo (Student‑t 2000)")
                pr1,pr2,pr3=st.columns(3)
                pr1.metric("Prob Bullish Besok", f"{prob_bull:.1f}%"); pr2.metric("Prob TP 30D", f"{hit_tp:.1f}%"); pr3.metric("Prob SL 30D", f"{hit_sl:.1f}%")

            except Exception as e:
                st.error(f"🚨 Kesalahan: {str(e)}")
