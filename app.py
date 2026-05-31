import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, t as student_t
from scipy.optimize import minimize
import warnings

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
# ===============================================================

warnings.filterwarnings("ignore")

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA (UI FIX)
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
    </style>
""", unsafe_allow_html=True)

st.title("📊 Quant & Risk Engine Pro (dengan Sentimen Berita)")
st.write("Algoritma kuantitatif + berita terkini. Distribusi Student‑t, volatilitas adaptif, Monte Carlo.")

if not SENTIMENT_AVAILABLE:
    st.warning("⚠️ NLTK tidak terpasang → fitur sentimen berita dinonaktifkan. "
               "Install dengan `pip install nltk` untuk mengaktifkan kembali.")

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

if st.button("JALANKAN QUANT ENGINE PRO + BERITA"):
    # Validasi
    if not ticker_input:
        st.warning("⚠️ Kode saham tidak boleh kosong!")
    elif total_capital is None or total_capital <= 0:
        st.warning("⚠️ Modal portofolio harus diisi dan > Rp 0!")
    else:
        with st.spinner("🤖 Mengunduh data harga, berita, dan menjalankan model kuantitatif..."):
            try:
                # ==========================================
                # 3. AMBIL DATA HARGA
                # ==========================================
                df = yf.download(ticker_input, period="1y")
                if df.empty:
                    st.error("❌ Data tidak ditemukan. Periksa kode saham.")
                    st.stop()

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                harga_terakhir = float(df['Close'].iloc[-1])
                returns = df['Close'].pct_change().dropna()
                if len(returns) < 20:
                    st.error("❌ Data historis kurang (minimal 20 hari).")
                    st.stop()

                # ==========================================
                # 4. SENTIMEN BERITA (YAHOO FINANCE) – DENGAN FALLBACK
                # ==========================================
                avg_sentiment = 0.0
                headlines = []
                sentimen_status = "Netral ⚪ (nonaktif)" if not SENTIMENT_AVAILABLE else "Netral ⚪"

                if SENTIMENT_AVAILABLE:
                    try:
                        ticker_obj = yf.Ticker(ticker_input)
                        news_list = ticker_obj.news
                        analyzer = SentimentIntensityAnalyzer()
                        sentiments = []
                        if news_list:
                            for item in news_list[:5]:
                                title = item.get('title', '')
                                summary = item.get('summary', '') if 'summary' in item else ''
                                text = f"{title}. {summary}"
                                vs = analyzer.polarity_scores(text)
                                sentiments.append(vs['compound'])
                                headlines.append(title)
                            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
                        else:
                            headlines = ["Tidak ada berita tersedia"]
                    except Exception as e:
                        avg_sentiment = 0.0
                        headlines = [f"Gagal mengambil berita: {str(e)}"]
                else:
                    headlines = ["Fitur sentimen tidak aktif (NLTK tidak terpasang)"]

                # Kategori sentimen (hanya jika tersedia)
                if SENTIMENT_AVAILABLE:
                    if avg_sentiment >= 0.05:
                        sentimen_status = "Positif 🟢"
                    elif avg_sentiment <= -0.05:
                        sentimen_status = "Negatif 🔴"
                    else:
                        sentimen_status = "Netral ⚪"

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
                # 8. MOMENTUM & Z‑SCORE ADAPTIF
                # ==========================================
                mom_3d = float((df['Close'].iloc[-1] / df['Close'].iloc[-4] - 1) * 100)
                mom_5d = float((df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100)
                mom_10d = float((df['Close'].iloc[-1] / df['Close'].iloc[-11] - 1) * 100)

                ma_20_close = float(df['Close'].tail(20).mean())
                std_20_close = float(df['Close'].tail(20).std())
                z_score = (harga_terakhir - ma_20_close) / std_20_close if std_20_close > 0 else 0.0

                # ==========================================
                # 9. REGIME CLASSIFICATION ADAPTIF
                # ==========================================
                ema_20 = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
                ema_50 = float(df['Close'].ewm(span=50, adjust=False).mean().iloc[-1])

                mom_5d_hist = df['Close'].pct_change(5).dropna() * 100
                bullish_thresh = np.percentile(mom_5d_hist, 70)
                bearish_thresh = np.percentile(mom_5d_hist, 30)

                z_hist = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
                z_up = np.percentile(z_hist.dropna(), 80)
                z_down = np.percentile(z_hist.dropna(), 20)

                if harga_terakhir > ema_20 and ema_20 > ema_50:
                    if mom_5d > bullish_thresh or z_score > z_up:
                        regime_status = "Strong Bullish Momentum 🚀"
                        ihsg_status = "RISK-ON 🔥"
                    else:
                        regime_status = "Bullish Momentum 📈"
                        ihsg_status = "RISK-ON 🔥"
                elif harga_terakhir < ema_20 and ema_20 < ema_50:
                    if mom_5d < bearish_thresh or z_score < z_down:
                        regime_status = "Panic Sell ⚠️"
                        ihsg_status = "RISK-OFF 🚨"
                    else:
                        regime_status = "Bearish Momentum 🔻"
                        ihsg_status = "RISK-OFF 🚨"
                elif harga_terakhir < ema_20 and ema_20 > ema_50:
                    regime_status = "Bearish Distribution 📉"
                    ihsg_status = "NEUTRAL ⚖️"
                elif harga_terakhir > ema_20 and ema_20 < ema_50:
                    if mom_3d > np.percentile(mom_5d_hist, 75):
                        regime_status = "Recovery 🔄"
                        ihsg_status = "NEUTRAL ⚖️"
                    else:
                        regime_status = "Bullish Accumulation 🏗️"
                        ihsg_status = "NEUTRAL ⚖️"
                else:
                    regime_status = "Sideways/Choppy ↔️"
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
                # 11. SIGNAL ENGINE DENGAN SENTIMEN (JIKA ADA)
                # ==========================================
                score = 0
                if mom_3d > 0: score += 1
                if z_score < -1.5: score += 1
                if harga_terakhir > ema_20: score += 1
                if 'Volume' in df.columns and df['Volume'].iloc[-1] > df['Volume'].tail(20).mean():
                    score += 1

                # *** INTEGRASI SENTIMEN HANYA JIKA TERSEDIA ***
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

                # ==========================================
                # 12. METRIK RISIKO LANJUTAN
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
                kelly_adj = max(0.0, kelly_raw * 0.3 * skew_adj)
                allocated_capital = total_capital * kelly_adj

                # ==========================================
                # 13. MONTE CARLO STUDENT‑T & ESTIMASI HARGA BESOK (LOG‑OU)
                # ==========================================
                n_sim = 2000
                n_days = 30
                sim_innov = student_t.rvs(df_est, loc=t_loc, scale=t_scale, size=(n_days, n_sim))
                price_paths = harga_terakhir * np.exp(np.cumsum(sim_innov, axis=0))

                # --- ESTIMASI CLOSE BESOK (log‑Ornstein‑Uhlenbeck) ---
                log_harga = np.log(df['Close'])
                log_lt_mean = log_harga.tail(20).mean()          # rata‑rata log harga 20 hari
                log_terakhir = np.log(harga_terakhir)
                theta = 1/20                                     # kecepatan mean‑reversion
                # Drift gabungan: mean‑reversion + ekspektasi harian Student‑t
                mu_ou_log = theta * (log_lt_mean - log_terakhir) + t_loc
                estimasi_close_besok = float(np.exp(log_terakhir + mu_ou_log))
                # -------------------------------------------------

                tp = r1
                sl = s1
                hit_tp_30d = (np.any(price_paths >= tp, axis=0).sum() / n_sim) * 100
                hit_sl_30d = (np.any(price_paths <= sl, axis=0).sum() / n_sim) * 100
                prob_bullish_besok = ((student_t.rvs(df_est, loc=t_loc, scale=t_scale, size=1000) > 0).sum() / 1000) * 100

                # ==========================================
                # 14. TAMPILAN DASHBOARD
                # ==========================================
                st.success(f"✅ Analisis Akurat: {ticker_input} | Harga: Rp {harga_terakhir:,.0f}".replace(",", "."))

                # --- BERITA & SENTIMEN ---
                st.header("📰 Sentimen Berita Terkini")
                col_sent1, col_sent2 = st.columns([1, 2])
                with col_sent1:
                    st.metric("Sentimen Agregat", f"{avg_sentiment:.2f}", sentimen_status)
                with col_sent2:
                    st.markdown("**5 Berita Teratas:**")
                    for i, h in enumerate(headlines[:5]):
                        st.markdown(f"{i+1}. {h}")
                st.divider()

                # --- SECTION 1: REGIME & VOLATILITY ---
                st.header("🧬 Market Regime & Volatility Engine")
                m1, m2, m3 = st.columns(3)
                m1.metric("Regime Status", regime_status)
                m2.metric("IHSG Condition", ihsg_status)
                m3.metric("EWMA Vol (λ=0.94)", f"{ewma_vol:.2f}%")
                st.markdown(f"Parkinson Vol: `{parkinson_vol:.2f}%` | Distribusi T (df={df_est:.1f}) | Skew: `{ret_skew:.2f}` | Kurt: `{ret_kurt:.2f}`")
                st.divider()

                # --- SECTION 2: MOMENTUM & MEAN‑REVERSION ---
                st.header("📊 Momentum & Mean‑Reversion")
                mo1, mo2, mo3, mo4 = st.columns(4)
                mo1.metric("Mom 3D", f"{mom_3d:+.2f}%")
                mo2.metric("Mom 5D", f"{mom_5d:+.2f}%")
                mo3.metric("Mom 10D", f"{mom_10d:+.2f}%")
                mo4.metric("Z‑Score (20D)", f"{z_score:+.2f}σ")
                st.divider()

                # --- SECTION 3: PIVOT & S/R ---
                st.header("🎯 Pivot & S/R")
                st.write(f"**Breakout Res20:** `{breakout_status}`")
                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric("R2", f"Rp {r2:,.0f}".replace(",", "."))
                p2.metric("R1", f"Rp {r1:,.0f}".replace(",", "."))
                p3.metric("PP", f"Rp {pp:,.0f}".replace(",", "."))
                p4.metric("S1", f"Rp {s1:,.0f}".replace(",", "."))
                p5.metric("S2", f"Rp {s2:,.0f}".replace(",", "."))
                st.divider()

                # --- SECTION 4: SIGNAL & ESTIMASI ---
                st.header("🔮 Signal & Estimasi Harga")
                tp1, tp2, tp3, tp4 = st.columns(4)
                tp1.metric("Signal V2 (dgn Sentimen)", signal)
                tp2.metric("Est. Close Besok (OU)", f"Rp {estimasi_close_besok:,.0f}".replace(",", "."))
                tp3.metric("Area Entry (S1-PP)", f"Rp {s1:,.0f} - {pp:,.0f}".replace(",", "."))
                tp4.metric("Target Profit (R1)", f"Rp {r1:,.0f}".replace(",", "."))
                st.divider()

                # --- SECTION 5: RISK ENGINE & PORTFOLIO SIZING ---
                st.header("🛡️ Risk Engine & Portfolio Sizing")
                r1, r2, r3 = st.columns(3)
                r1.metric("Kelly Adj. Allocation", f"{kelly_adj*100:.1f}%")
                r2.metric("Rekom. Modal (Rp)", f"Rp {allocated_capital:,.0f}".replace(",", "."))
                r3.metric("Beta vs IHSG", f"{beta_ihsg:.2f}x")
                st.markdown(
                    f"Max DD: `{max_dd:.2f}%` (30D: `{max_dd_30d:.2f}%`) | "
                    f"Sharpe: `{sharpe:.2f}` | Sortino: `{sortino:.2f}` | Calmar: `{calmar:.2f}`"
                )
                st.markdown(
                    f"VaR 95% (t): `{var_95_t:.2f}%` | CVaR 95% (t): `{cvar_95_t:.2f}%`"
                )
                st.divider()

                # --- SECTION 6: PROBABILITY ENGINE (MONTE CARLO T) ---
                st.header("🎲 Probability Engine (Monte Carlo Student‑t 2000 runs)")
                pr1, pr2, pr3 = st.columns(3)
                pr1.metric("Prob Bullish Besok", f"{prob_bullish_besok:.1f}%")
                pr2.metric("Prob Hit TP (30D)", f"{hit_tp_30d:.1f}%")
                pr3.metric("Prob Hit SL (30D)", f"{hit_sl_30d:.1f}%")

            except Exception as e:
                st.error(f"🚨 Kesalahan pemrosesan: {str(e)}")
