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
# ------------------------------------
if st.button("JALANKAN QUANT ENGINE"):
    with st.spinner("Mengunduh data historis & memproses algoritma statistik..."):
        try:
            df = yf.download(ticker_input, period="1y")
            
            if df.empty:
                st.error("Saham tidak ditemukan! Pastikan kode benar dan menggunakan akhiran '.JK'")
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df['Return'] = df['Close'].pct_change()
                df = df.dropna()

                # METRIK 1: STATISTIK KUANTITATIF & VOLATILITAS
                daily_vol_raw = df['Return'].tail(20).std()
                
                # Antisipasi jika deviasi nilainya 0 atau NaN karena data feed error
                if pd.isna(daily_vol_raw) or daily_vol_raw == 0:
                    daily_vol_raw = 0.02  # Fallback standar 2% per hari
                
                fast_std = daily_vol_raw * np.sqrt(252) * 100
                
                def calc_mad(x):
                    return np.median(np.abs(x - np.median(x)))
                rolling_mad = calc_mad(df['Return'].tail(20))
                robust_std = rolling_mad * 1.4826 * np.sqrt(252) * 100
                
                hl_term = np.log(df['High'] / df['Low']) ** 2
                parkinson_var = hl_term.tail(20).sum() / (4 * np.log(2) * 20)
                parkinson_vol = np.sqrt(parkinson_var * 252) * 100

                # METRIK 2: MOMENTUM & MEAN-REVERSION
                ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
                std20 = df['Close'].rolling(window=20).std().iloc[-1]
                harga_terakhir = df['Close'].iloc[-1].item()
                z_score = (harga_terakhir - ma20) / std20
                val_skew = skew(df['Return'].tail(20))

                # ==========================================================
                # METRIK 3: STATISTIK SUPPORT & RESISTANCE (ANTI-GLITCH)
                # ==========================================================
                # Menggunakan deviasi pergerakan harga riil untuk menentukan benteng pertahanan harga
                r1 = harga_terakhir * (1 + daily_vol_raw)
                s1 = harga_terakhir * (1 - daily_vol_raw)
                r2 = harga_terakhir * (1 + (2 * daily_vol_raw))
                s2 = harga_terakhir * (1 - (2 * daily_vol_raw))

                # --- TAMBAHAN FITUR: RES 20 & BREAKOUT DETECTOR ---
                # Mengambil harga tertinggi dari 20 hari sebelum hari ini
                res20 = float(df['High'].iloc[-21:-1].max())
                
                # Logika penentuan status breakout
                if harga_terakhir > res20:
                    breakout_status = "🔥 BREAKOUT!"
                elif float(df['High'].iloc[-1].item()) > res20:
                    breakout_status = "⚡ Intraday Breakout (Ekor)"
                else:
                    breakout_status = "❌ Belum Breakout"
                # --------------------------------------------------

                # METRIK 4: SIGNAL ENGINE & PROBABILITAS
                df_student_t = len(df['Return'].tail(20)) - 1
                prob_bullish = (1 - t.cdf(0, df_student_t, loc=df['Return'].tail(20).mean(), scale=df['Return'].tail(20).std())) * 100
                
                if z_score < -1.5 and prob_bullish > 52.0:
                    signal = "BUY / LONG"
                    tp = harga_terakhir * (1 + (fast_std / 100 / np.sqrt(252)) * 1.5)
                    sl = harga_terakhir * (1 - (fast_std / 100 / np.sqrt(252)) * 1.0)
                elif z_score > 1.5:
                    signal = "TAKE PROFIT / SELL"
                    tp, sl = 0, 0
                else:
                    signal = "NO TRADE (Sinyal Lemah)"
                    tp, sl = 0, 0

                # METRIK 5: RISK ENGINE (KELLY CRITERION)
                w = prob_bullish / 100
                r_ratio = (tp - harga_terakhir) / (harga_terakhir - sl) if (signal == "BUY / LONG" and (harga_terakhir - sl) != 0) else 1.5
                
                kelly_raw = w - ((1 - w) / r_ratio)
                kelly_adj = max(0.0, kelly_raw * 0.4) 
                porsi_modal = total_capital * kelly_adj
                jumlah_lot = int((porsi_modal / (harga_terakhir * 100)))

                # DISPLAY HASIL
                st.success(f"Analisis Saham {ticker_input} Berhasil Dieksekusi!")
                
                st.subheader("🤖 Signal Engine Output")
                c1, c2, c3 = st.columns(3)
                c1.metric("Rekomendasi Sinyal", signal)
                c2.metric("Harga Saat Ini (Close)", f"Rp {harga_terakhir:,.0f}")
                c3.metric("Probabilitas Bullish", f"{prob_bullish:.1f}%")
                
                if signal == "BUY / LONG":
                    c1_tp, c2_sl = st.columns(2)
                    c1_tp.metric("Target Price (TP)", f"Rp {tp:,.0f}")
                    c2_sl.metric("Stop Loss (SL)", f"Rp {sl:,.0f}")

                st.subheader("📊 Statistik Kuantitatif & Volatilitas")
                col1, col2, col3 = st.columns(3)
                col1.metric("σ Fast (Std Dev Annualized)", f"{fast_std:.2f}%")
                col2.metric("Robust Std (MAD)", f"{robust_std:.2f}%")
                col3.metric("Parkinson Volatility", f"{parkinson_vol:.2f}%")
                
                col4, col5 = st.columns(2)
                col4.metric("Z-Score (20-day MA)", f"{z_score:.2f} σ")
                col5.metric("Skewness (Asimetri Harga)", f"{val_skew:.2f}")

                st.subheader("📍 Statistical Support & Resistance (Volatility Bands)")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Resistance 2 (R2 - 2σ)", f"Rp {r2:,.0f}".replace(",", "."))
                p2.metric("Resistance 1 (R1 - 1σ)", f"Rp {r1:,.0f}".replace(",", "."))
                p3.metric("Support 1 (S1 - 1σ)", f"Rp {s1:,.0f}".replace(",", "."))
                p4.metric("Support 2 (S2 - 2σ)", f"Rp {s2:,.0f}".replace(",", "."))

                # --- TAMPILAN BARU UNTUK BREAKOUT ENGINE ---
                st.markdown(" ") # Kasih sedikit jarak kosong
                b1, b2 = st.columns(2)
                b1.metric("20-Day High Ceiling (Res 20)", f"Rp {res20:,.0f}".replace(",", "."))
                b2.metric("Kondisi Saham Saat Ini", breakout_status)
                # -------------------------------------------

                st.subheader("🛡️ Risk Engine (Kelly Criterion)")
                r_col1, r_col2, r_col3 = st.columns(3)
                r_col1.metric("Kelly Allocation (Adj x0.4)", f"{kelly_adj*100:.1f}% Modal")
                r_col2.metric("Rekomendasi Alokasi Uang", f"Rp {porsi_modal:,.0f}")
                r_col3.metric("Maksimal Pembelian", f"{jumlah_lot:,} Lot")

                st.subheader("📈 Grafik Tren Harga (1 Tahun Terakhir)")
                st.line_chart(df['Close'])

        except Exception as e:
            st.error(f"Terjadi error pada sistem kalkulasi: {e}")
