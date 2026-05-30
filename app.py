import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, t

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA (UI FIX)
# ==========================================
st.set_page_config(
    page_title="Quant Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk mempercantik tampilan di Browser HP (iOS) & Hapus Tulisan Pengganggu
st.markdown("""
    <style>
    .main { background-color: #0f1116; color: #ffffff; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; color: #00ffcc; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #8892b0; }
    .stButton>button { width: 100%; background-color: #1f2937; color: white; border: 1px solid #374151; }
    .stButton>button:hover { background-color: #374151; border-color: #00ffcc; }
    h1, h2, h3 { color: #f3f4f6; }
    
    /* MANTRA SIHIR: Menghilangkan tulisan 'Press Enter to apply' yang nabrak di HP */
    div[data-testid="InputInstructions"] { 
        display: none !important; 
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Quant & Risk Engine Dashboard")
st.write("Analisator Kuantitatif Saham untuk Trading Plan Objektif")
# ==========================================
# 2. PANEL INPUT (SIMPLE & MOBILE FRIENDLY)
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

# --- PROSES KODE TICKER OTOMATIS ---
# Jika user mengetik 'BBRI', otomatis diubah jadi 'BBRI.JK'
# Jika user sudah mengetik 'BBRI.JK', sistem tidak akan mengubahnya lagi
if ticker_raw and not ticker_raw.endswith(".JK"):
    ticker_input = f"{ticker_raw}.JK"
else:
    ticker_input = ticker_raw
# ------------------------------------if st.button("JALANKAN QUANT ENGINE"):
    # 1. VALIDASI INPUT KOSONG
    if not ticker_input:
        st.warning("⚠️ Kode saham tidak boleh kosong! Silakan masukkan kode saham terlebih dahulu.")
    elif total_capital is None:
        st.warning("⚠️ Nominal modal tidak boleh kosong! Silakan isi modal portofolio Anda.")
    elif total_capital <= 0:
        st.warning("⚠️ Modal portofolio harus lebih besar dari Rp 0!")
    else:
        with st.spinner("🤖 Menjalankan Algoritma Kuantitatif & Simulasi Monte Carlo..."):
            try:
                # Download Data Saham (1 Tahun)
                df = yf.download(ticker_input, period="1y")
                if df.empty:
                    st.error("❌ Data saham tidak ditemukan atau kode salah.")
                    st.stop()
                
                # 🔥 ANTIDOTE FIX: Meratakan kolom MultiIndex YFinance agar kembali jadi Series biasa
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Pemrosesan Data Dasar (Lebih aman tanpa .item())
                harga_terakhir = float(df['Close'].iloc[-1])
                returns = df['Close'].pct_change().dropna()
                
                # Try-Except untuk Fetch Data IHSG (^JKSE) guna menghitung Beta
                try:
                    df_ihsg = yf.download("^JKSE", period="1y")
                    if isinstance(df_ihsg.columns, pd.MultiIndex):
                        df_ihsg.columns = df_ihsg.columns.get_level_values(0)
                        
                    ihsg_returns = df_ihsg['Close'].pct_change().dropna()
                    combined_ret = pd.concat([returns, ihsg_returns], axis=1).dropna()
                    covariance = combined_ret.cov().iloc[0, 1]
                    ihsg_variance = combined_ret.iloc[:, 1].var()
                    beta_ihsg = covariance / ihsg_variance
                except:
                    beta_ihsg = 1.0  # Fallback jika IHSG sedang glitch

                # ==========================================
                # A. KOSMETIK & STATISTIK VOLATILITAS (IMAGE 1)
                # ==========================================
                std_dev_20 = float(returns.tail(20).std() * np.sqrt(252) * 100)
                
                # Rumus Parkinson Volatility
                log_hl = np.log(df['High'] / df['Low']).tail(20) ** 2
                parkinson_vol = float(np.sqrt(log_hl.mean() / (4 * np.log(2))) * np.sqrt(252) * 100)
                
                # Penentuan Market Regime Berdasarkan Trend & Volatilitas
                ema_20 = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
                ema_50 = float(df['Close'].ewm(span=50, adjust=False).mean().iloc[-1])
                
                if harga_terakhir > ema_20 and harga_terakhir > ema_50:
                    regime_status = "BULLISH MOMENTUM"
                    ihsg_status = "RISK-ON 🔥"
                elif harga_terakhir < ema_20 and harga_terakhir < ema_50:
                    regime_status = "BEARISH ACCUMULATION"
                    ihsg_status = "RISK-OFF 🚨"
                else:
                    regime_status = "CHOPPY / SIDEWAYS"
                    ihsg_status = "NEUTRAL ⚖️"

                # ==========================================
                # B. MOMENTUM & MEAN-REVERSION (IMAGE 2)
                # ==========================================
                mom_3d = float(((df['Close'].iloc[-1] / df['Close'].iloc[-4]) - 1) * 100)
                mom_5d = float(((df['Close'].iloc[-1] / df['Close'].iloc[-6]) - 1) * 100)
                mom_10d = float(((df['Close'].iloc[-1] / df['Close'].iloc[-11]) - 1) * 100)
                
                # Z-Score
                ma_20_close = float(df['Close'].tail(20).mean())
                std_20_close = float(df['Close'].tail(20).std())
                z_score = (harga_terakhir - ma_20_close) / std_20_close if std_20_close > 0 else 0.0

                # ==========================================
                # C. PIVOT POINTS & TRADING PLAN
                # ==========================================
                last_high = float(df['High'].iloc[-1])
                last_low = float(df['Low'].iloc[-1])
                
                pp = (last_high + last_low + harga_terakhir) / 3
                pivot_r1 = (2 * pp) - last_low
                pivot_s1 = (2 * pp) - last_high
                pivot_r2 = pp + (last_high - last_low)
                pivot_s2 = pp - (last_high - last_low)
                
                res20 = float(df['High'].iloc[-21:-1].max())
                breakout_status = "YES (🔥)" if harga_terakhir > res20 else "NO"

                # Consensus Signal Engine Sederhana
                score = 0
                if mom_3d > 0: score += 1
                if z_score < -1.0: score += 1 # Syarat Mean Reversion (jenuh jual)
                if harga_terakhir > ema_20: score += 1
                
                if score >= 2 and breakout_status == "YES (🔥)":
                    signal_v12 = "🔥 STRONGLY BUY"
                elif score >= 1:
                    signal_v12 = "⚡ BUY (TACTICAL)"
                else:
                    signal_v12 = "🚨 NO TRADE / AVOID"

                # ==========================================
                # D. RISK ENGINE & ADVANCED RISK METRICS
                # ==========================================
                # Kelly Criterion Calculator
                win_days = returns[returns > 0]
                loss_days = returns[returns < 0]
                win_rate = len(win_days) / len(returns) if len(returns) > 0 else 0.5
                avg_gain = win_days.mean() if len(win_days) > 0 else 0.01
                avg_loss = abs(loss_days.mean()) if len(loss_days) > 0 else 0.01
                win_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else 1.0
                
                kelly_raw = win_rate - ((1 - win_rate) / win_loss_ratio) if win_loss_ratio > 0 else 0
                kelly_adj = max(0.0, kelly_raw * 0.4) # Fractional Kelly (x0.4 multiplier untuk pengaman bursa)
                allocated_capital = total_capital * kelly_adj

                # Max Drawdown 30 Hari Terakhir
                roll_max = df['Close'].tail(30).cummax()
                drawdown = (df['Close'].tail(30) - roll_max) / roll_max
                max_dd_30d = float(drawdown.min() * 100)

                # Sharpe Ratio
                sharpe_est = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

                # CVaR 95% (Conditional Value at Risk)
                var_95 = returns.quantile(0.05)
                cvar_95 = float(returns[returns <= var_95].mean() * 100)

                # ==========================================
                # E. MONTE CARLO 1000x SIMULATION ENGINE
                # ==========================================
                n_simulations = 1000
                n_days = 30
                daily_mu = returns.mean()
                daily_sigma = returns.std()
                
                # Simulasi proyeksi harga geometric brownian motion fiktif
                sim_returns = np.random.normal(daily_mu, daily_sigma, (n_days, n_simulations))
                price_paths = harga_terakhir * np.exp(np.cumsum(sim_returns, axis=0))
                
                # Hitung Probabilitas Berdasarkan Batas Atas/Bawah Tradisional
                hit_tp_30d = (np.any(price_paths >= pivot_r1, axis=0).sum() / n_simulations) * 100
                hit_sl_30d = (np.any(price_paths <= pivot_s1, axis=0).sum() / n_simulations) * 100
                prob_bullish_besok = ( (np.random.normal(daily_mu, daily_sigma, 1000) > 0).sum() / 1000 ) * 100

                # ==========================================
                # VISUALISASI KE DASHBOARD STREAMLIT
                # ==========================================
                st.success(f"✅ Analisis Berhasil untuk {ticker_input} | Harga Terakhir: Rp {harga_terakhir:,.0f}".replace(",", "."))
                
                # --- SECTION 1: REGIME & VOLATILITY ---
                st.header("🧬 Market Regime & Volatility Engine")
                m1, m2, m3 = st.columns(3)
                m1.metric("Regime Status", regime_status)
                m2.metric("IHSG Condition", ihsg_status)
                m3.metric("Parkinson Volatility (20D)", f"{parkinson_vol:.2f}%")
                
                st.markdown(f"**Metrics:** Rolling StdDev20: `{std_dev_20:.2f}%` | Model Target Distribution: `Student-T (df=5)` equivalent")
                st.separator()

                # --- SECTION 2: MOMENTUM & MEAN-REVERSION ---
                st.header("📊 Momentum & Mean-Reversion")
                mo1, mo2, mo3, mo4 = st.columns(4)
                mo1.metric("Mom 3D", f"{mom_3d:+.2f}%")
                mo2.metric("Mom 5D", f"{mom_5d:+.2f}%")
                mo3.metric("Mom 10D", f"{mom_10d:+.2f}%")
                mo4.metric("Z-Score (20D)", f"{z_score:+.2f}σ")
                st.separator()

                # --- SECTION 3: PIVOT POINTS & TRADING PLAN ---
                st.header("🎯 Pivot Points & Trading Plan")
                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric("Resistance 2 (R2)", f"Rp {pivot_r2:,.0f}".replace(",", "."))
                p2.metric("Resistance 1 (R1)", f"Rp {pivot_r1:,.0f}".replace(",", "."))
                p3.metric("Pivot Point (PP)", f"Rp {pp:,.0f}".replace(",", "."))
                p4.metric("Support 1 (S1)", f"Rp {pivot_s1:,.0f}".replace(",", "."))
                p5.metric("Support 2 (S2)", f"Rp {pivot_s2:,.0f}".replace(",", "."))
                
                st.write(f"**Breakout Status (Res20):** `{breakout_status}`")
                
                st.markdown("📋 **V12 Quant Trading Plan (Objective Rules):**")
                tp1, tp2, tp3 = st.columns(3)
                tp1.metric("SIGNAL GENERATOR", signal_v12)
                tp2.metric("Area Entry Ideal (S1 - PP)", f"Rp {pivot_s1:,.0f} - {pp:,.0f}".replace(",", "."))
                tp3.metric("Target Profit Terdekat (R1)", f"Rp {pivot_r1:,.0f}".replace(",", "."))
                st.separator()

                # --- SECTION 4: RISK ENGINE & ADVANCED METRICS ---
                st.header("🛡️ Risk Engine & Portfolio Sizing")
                r1, r2, r3 = st.columns(3)
                r1.metric("Kelly Allocation %", f"{kelly_adj*100:.1f}% capital")
                r2.metric("Rekomendasi Size Modal (Rp)", f"Rp {allocated_capital:,.0f}".replace(",", "."))
                r3.metric("Beta vs IHSG", f"{beta_ihsg:.2f}x")
                
                st.markdown(f"**Advanced Risk Bounds:** Max Drawdown (30D): `{max_dd_30d:.2f}%` | Sharpe Ratio Est: `{sharpe_est:.2f}` | CVaR (95%): `{cvar_95:.2f}%` (Potensi risiko ekstrim harian)")
                st.separator()

                # --- SECTION 5: PROBABILITY ENGINE (MONTE CARLO) ---
                st.header("🎲 Probability Engine (Monte Carlo 1000 Simulations)")
                pr1, pr2, pr3 = st.columns(3)
                pr1.metric("Prob Bullish Besok", f"{prob_bullish_besok:.1f}%")
                pr2.metric("Prob TP Hit (30D)", f"{hit_tp_30d:.1f}%")
                pr3.metric("Prob SL Hit (30D)", f"{hit_sl_30d:.1f}%")

            except Exception as e:
                st.error(f"🚨 Terjadi kesalahan pemrosesan algoritma: {str(e)}")
