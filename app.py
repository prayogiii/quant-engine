import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, t as student_t
from scipy.optimize import minimize
import warnings
import urllib.parse
import re
import csv
import os
from datetime import datetime, timedelta
import pytz
import math
import google.generativeai as genai

# ====================== FALLBACK HANDLERS ======================
PLOTLY_AVAILABLE = True
try: import plotly.graph_objects as go
except ImportError: PLOTLY_AVAILABLE = False

SENTIMENT_AVAILABLE = True
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try: nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError: nltk.download('vader_lexicon', quiet=True)
except ImportError: SENTIMENT_AVAILABLE = False

RSS_AVAILABLE = True
try: import feedparser
except ImportError: RSS_AVAILABLE = False

TRANSLATOR_AVAILABLE = True
try: from deep_translator import GoogleTranslator
except ImportError: TRANSLATOR_AVAILABLE = False

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# V12 ADAPTIVE ENGINE – KONSTANTA & STATE
# ═══════════════════════════════════════════════════════════════
FACTOR_KEYS   = ["Momentum","AI_Senti","MeanRev","Beta_IHSG","Coppock"]
WEIGHT_MIN    = 0.08
WEIGHT_MAX    = 0.40
SOFTMAX_TEMP  = 2.5
AI_SIGNAL_CAP = 0.30
MC_PESSIMISM  = 0.82
V12_MEM_FILE  = "adaptive_memory.csv"
V12_PRED_FILE = "v12_predictions.csv"

def load_v12_memory():
    mem = {}
    if not os.path.isfile(V12_MEM_FILE): return mem
    with open(V12_MEM_FILE,'r',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = row['ticker']
            if t not in mem: mem[t] = {'weights':{},'accuracy':{},'error_ema':{}}
            for k in FACTOR_KEYS:
                if row.get(f'W_{k}'): mem[t]['weights'][k] = float(row[f'W_{k}'])
                if row.get(f'ACC_{k}'): mem[t]['accuracy'][k] = float(row[f'ACC_{k}'])
                if row.get(f'ERR_{k}'): mem[t]['error_ema'][k] = float(row[f'ERR_{k}'])
    return mem

def save_v12_memory(mem):
    rows = []
    for t,d in mem.items():
        row = {'ticker':t}
        for k in FACTOR_KEYS:
            row[f'W_{k}']   = d['weights'].get(k, default_weight(k,'SIDEWAYS'))
            row[f'ACC_{k}'] = d['accuracy'].get(k,0.5)
            row[f'ERR_{k}'] = d['error_ema'].get(k,1.0)
        rows.append(row)
    with open(V12_MEM_FILE,'w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['ticker']+[f'W_{k}' for k in FACTOR_KEYS]+[f'ACC_{k}' for k in FACTOR_KEYS]+[f'ERR_{k}' for k in FACTOR_KEYS])
        w.writeheader(); w.writerows(rows)

def default_weight(factor, regime):
    defaults = {
        "STABLE BULLISH": {"Momentum":0.25,"AI_Senti":0.18,"MeanRev":0.12,"Beta_IHSG":0.15,"Coppock":0.30},
        "VOLATILE UPTREND": {"Momentum":0.28,"AI_Senti":0.14,"MeanRev":0.12,"Beta_IHSG":0.16,"Coppock":0.30},
        "HIGH-STRESS PANIC": {"Momentum":0.15,"AI_Senti":0.18,"MeanRev":0.22,"Beta_IHSG":0.15,"Coppock":0.30},
        "SIDEWAYS / CONSOLIDATION": {"Momentum":0.15,"AI_Senti":0.18,"MeanRev":0.27,"Beta_IHSG":0.12,"Coppock":0.28},
        "BEARISH ACCUMULATION": {"Momentum":0.20,"AI_Senti":0.18,"MeanRev":0.20,"Beta_IHSG":0.15,"Coppock":0.27}
    }
    return defaults.get(regime, {"Momentum":0.23,"AI_Senti":0.17,"MeanRev":0.15,"Beta_IHSG":0.15,"Coppock":0.30}).get(factor,0.15)

if 'v12_memory' not in st.session_state:
    st.session_state.v12_memory = load_v12_memory()

# ---------- Coppock Curve ----------
def coppock_curve(prices, rP1=14, rP2=11, wP=10):
    if len(prices) < max(rP1,rP2)+wP+2: return 0.0,0.0
    roc1 = [(prices[i]-prices[i-rP1])/prices[i-rP1]*100 for i in range(rP1,len(prices))]
    roc2 = [(prices[i]-prices[i-rP2])/prices[i-rP2]*100 for i in range(rP2,len(prices))]
    mn = min(len(roc1),len(roc2))
    combined = [roc1[i]+roc2[i] for i in range(-mn,0)]
    def wma(data,per):
        if len(data)<per: return 0.0
        w = np.arange(1,per+1)
        vals = [np.dot(data[i:i+per],w)/w.sum() for i in range(len(data)-per+1)]
        return vals[-1]
    curr = wma(combined,wP)
    prev = wma(combined[:-1],wP) if len(combined)>wP else 0.0
    return curr,prev

# ---------- Adaptive Weights ----------
def get_adaptive_weights(ticker, regime):
    mem = st.session_state.v12_memory.get(ticker, {})
    defs = {k: default_weight(k, regime) for k in FACTOR_KEYS}
    w_pri = {}
    for k in FACTOR_KEYS:
        w = mem.get('weights',{}).get(k, defs[k])
        acc = mem.get('accuracy',{}).get(k,0.5)
        if acc>=0.65: w = min(w*1.15, WEIGHT_MAX)
        elif acc>=0.45: pass
        elif acc>=0.35: w *= 0.5
        else: w = max(w*0.2, WEIGHT_MIN/2)
        w_pri[k] = max(WEIGHT_MIN, min(WEIGHT_MAX, w))
    err = {k: mem.get('error_ema',{}).get(k,1.0) for k in FACTOR_KEYS}
    scores = {k: 1.0/(err[k]+1e-6) for k in FACTOR_KEYS}
    exp_s = {k: math.exp(v/SOFTMAX_TEMP) for k,v in scores.items()}
    sum_exp = sum(exp_s.values())
    sm = {k: v/sum_exp for k,v in exp_s.items()}
    final = {}
    for k in FACTOR_KEYS:
        bw = w_pri[k]; sw = max(0.10, sm[k])
        final[k] = max(WEIGHT_MIN, min(WEIGHT_MAX, 0.6*bw + 0.4*bw*sw*len(FACTOR_KEYS)))
    tot = sum(final.values())
    return {k: v/tot for k,v in final.items()}

def update_v12_memory(ticker, factor_signals, actual_return, volatility=0.02):
    if ticker not in st.session_state.v12_memory:
        st.session_state.v12_memory[ticker] = {'weights':{},'accuracy':{},'error_ema':{}}
    mem = st.session_state.v12_memory[ticker]
    alpha = 0.20 if volatility>0.04 else (0.10 if volatility>0.02 else 0.05)
    ac = max(-1.0, min(1.0, actual_return))
    for k in FACTOR_KEYS:
        sv = max(-1.0, min(1.0, factor_signals.get(k,0.0)))
        err = abs(sv - ac)
        old = mem['error_ema'].get(k,1.0)
        mem['error_ema'][k] = old*(1-alpha) + err*alpha
    for k in FACTOR_KEYS:
        hit = 1.0 if factor_signals.get(k,0.0)*actual_return>0 else 0.0
        old_acc = mem['accuracy'].get(k,0.5)
        mem['accuracy'][k] = old_acc*0.97 + hit*0.03
    for k in FACTOR_KEYS:
        acc = mem['accuracy'][k]
        old_w = mem['weights'].get(k, default_weight(k,'SIDEWAYS'))
        if acc>=0.65: new_w = min(old_w*1.01, WEIGHT_MAX)
        elif acc<0.35: new_w = max(old_w*0.99, WEIGHT_MIN)
        else: new_w = old_w
        mem['weights'][k] = new_w
    st.session_state.v12_memory[ticker] = mem
    save_v12_memory(st.session_state.v12_memory)

# ==========================================
# KONFIGURASI FILE RIWAYAT & SESSION STATE
# ==========================================
RIWAYAT_FILE = "riwayat_analisis.csv"

def simpan_riwayat(ringkasan):
    file_exists = os.path.isfile(RIWAYAT_FILE)
    with open(RIWAYAT_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ringkasan.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(ringkasan)

def muat_riwayat_dari_csv():
    if not os.path.isfile(RIWAYAT_FILE):
        return []
    with open(RIWAYAT_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        riwayat = list(reader)
    riwayat.sort(key=lambda x: x.get('Waktu', ''), reverse=True)
    return riwayat

if "riwayat" not in st.session_state:
    st.session_state.riwayat = muat_riwayat_dari_csv()

# ==========================================
# FUNGSI AI GEMINI
# ==========================================
def dapatkan_model_gemini(api_key):
    if not api_key:
        return None, "API key belum diisi."
    try:
        genai.configure(api_key=api_key)
        available = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_id = m.name.split('/')[-1]
                available.append(model_id)
        if not available:
            return None, "Tidak ada model Gemini."
        for model_id in available:
            try:
                model = genai.GenerativeModel(model_id)
                model.generate_content("test", generation_config={"max_output_tokens": 1})
                return model, None
            except Exception:
                continue
        return None, "Model gagal digunakan."
    except Exception as e:
        return None, f"Error: {str(e)}"

def analisis_saham_dengan_ai(data_saham, riwayat, api_key):
    model, error = dapatkan_model_gemini(api_key)
    if error:
        return None, error

    riwayat_text = ""
    if riwayat:
        riwayat_text = "Riwayat analisis sebelumnya:\n"
        for r in riwayat[:10]:
            base = f"- {r['Waktu']} | {r['Saham']} | Sinyal: {r['Sinyal']} | RRR: {r['RRR']} | Rezim: {r['Rezim']}"
            ai_insight = r.get("AI_Insight", "").strip()
            if ai_insight:
                short_insight = (ai_insight[:120] + "...") if len(ai_insight) > 120 else ai_insight
                base += f" | AI Insight: {short_insight}"
            riwayat_text += base + "\n"
    else:
        riwayat_text = "Belum ada riwayat sebelumnya."

    prompt = f"""
Anda adalah asisten analis saham profesional. Berikut data analisis teknikal dan fundamental saham {data_saham['Saham']}:

- Harga terakhir: Rp {data_saham['Harga']}
- Sinyal saat ini: {data_saham['Sinyal']}
- Rezim Pasar: {data_saham['Rezim']}
- Sentimen Berita: {data_saham['Sentimen']}
- Risk/Reward Ratio (RRR): {data_saham['RRR']}
- Probabilitas Naik Besok: {data_saham['Prob Naik']}
- Take Profit: +{data_saham['TP%']}%
- Stop Loss: -{data_saham['SL%']}%
- Estimasi Harga Besok: Rp {data_saham['Estimasi']}
- Beta terhadap IHSG: {data_saham.get('Beta', 'N/A')}
- Win Rate Backtest: {data_saham.get('WinRate', 'N/A')}
- Profit Factor Backtest: {data_saham.get('ProfitFactor', 'N/A')}
- Max Drawdown Backtest: {data_saham.get('MaxDD', 'N/A')}
- Alokasi Kelly Maks: {data_saham.get('Kelly', 'N/A')}%
- Fundamental: Market Cap: {data_saham.get('Fundamental_MC', 'N/A')}, PER: {data_saham.get('Fundamental_PER', 'N/A')}, PBV: {data_saham.get('Fundamental_PBV', 'N/A')}, ROE: {data_saham.get('Fundamental_ROE', 'N/A')}, D/E: {data_saham.get('Fundamental_DE', 'N/A')}

{riwayat_text}

Berdasarkan data di atas, berikan analisis ringkas (Bahasa Indonesia) yang mencakup:
- Makna sinyal dalam konteks saat ini
- Kekuatan dan kelemahan saham
- Risiko utama
- Rekomendasi langkah selanjutnya (buy/hold/sell) dengan alasan singkat
- Jika ada pola dari riwayat, sebutkan.
Gunakan bahasa mudah dipahami trader, maksimal 4 paragraf pendek.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), None
    except Exception as e:
        return None, f"Gagal menghasilkan insight AI: {str(e)}"

def analisis_riwayat_global(riwayat_data, api_key):
    model, error = dapatkan_model_gemini(api_key)
    if error:
        return None, error
    if not riwayat_data:
        return None, "Belum ada riwayat."

    prompt = "Berikut adalah riwayat analisis saham yang telah dilakukan:\n\n"
    for r in riwayat_data[:30]:
        prompt += (
            f"- {r['Waktu']} | {r['Saham']} | Sinyal: {r['Sinyal']} | "
            f"Harga: {r['Harga']} | RRR: {r['RRR']} | Sentimen: {r['Sentimen']} | "
            f"Rezim: {r['Rezim']} | TP%: {r['TP%']}% | SL%: {r['SL%']}%\n"
        )
    prompt += (
        "\nBerdasarkan data di atas, berikan analisis ringkas (Bahasa Indonesia):\n"
        "- Pola sinyal yang sering muncul\n"
        "- Saham dengan peluang terbaik menurut data\n"
        "- Rekomendasi perbaikan strategi\n"
        "- Insight tambahan yang berguna untuk trader"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), None
    except Exception as e:
        return None, f"Gagal menghasilkan insight: {str(e)}"

def bersihkan_teks_ai(teks):
    if not teks:
        return teks
    teks = re.sub(r'^#{1,3}\s*', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'\*\*', '', teks)
    teks = re.sub(r'\*', '', teks)
    teks = teks.replace('\n', '<br>')
    return teks

# ==========================================
# KONFIGURASI HALAMAN & STYLING
# ==========================================
st.set_page_config(page_title="Quant Risk Engine Pro v2", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0f1116; color: #ffffff; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; color: #00ffcc; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #8892b0; }
    .stButton>button { width: 100%; background-color: #1f2937; color: white; border: 1px solid #374151; }
    .stButton>button:hover { background-color: #374151; border-color: #00ffcc; }
    h1, h2, h3 { color: #f3f4f6; }
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
    
    .ai-insight-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #8b5cf6;
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.6;
    }
    .ai-insight-card h3 {
        color: #a78bfa;
        margin-top: 0;
        font-size: 20px;
    }
    .ai-insight-card p {
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 📊 QuantRisk Pro")
    
    # ---------- PILIH TIMEFRAME ----------
    trading_style = st.radio(
        "⏱️ Gaya Trading:",
        ["Swing Trade (Mingguan)", "Day Trade (Harian)"],
        index=0,
        key="trading_style"
    )
    if "Day Trade" in trading_style:
        timeframe_label = "5m"
        period = "5d"
        st.caption("Menggunakan data 5-menit (5 hari terakhir)")
    else:
        timeframe_label = "1d"
        period = "2y"
        st.caption("Menggunakan data harian (2 tahun terakhir)")
    
    st.markdown("Masukkan kode saham IHSG untuk analisis lengkap.")
    ticker_raw = st.text_input("🔍 Kode Saham", value="BBRI", placeholder="Contoh: BBRI, TLKM, BMRI").upper().strip()
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

    # ---------- RIWAYAT ANALISIS (FORMAT BARU) ----------
    st.subheader("📜 Riwayat Analisis")
    if st.session_state.riwayat:
        for r in st.session_state.riwayat[:10]:
            if "STRONG BUY" in r.get('Sinyal',''):
                sig_icon = "🔥"
            elif "BUY" in r.get('Sinyal',''):
                sig_icon = "⚡"
            elif "HOLD" in r.get('Sinyal',''):
                sig_icon = "⏸️"
            else:
                sig_icon = "🚨"
            
            conf_str = r.get('Confidence', '0%')
            try:
                conf_val = float(conf_str.replace('%',''))
            except:
                conf_val = 0
            if conf_val >= 70: conf_text = "Tinggi ▲"
            elif conf_val >= 50: conf_text = "Sedang ►"
            else: conf_text = "Rendah ▼"
            
            est_ret_str = r.get('Est_Return', '0')
            try:
                est_ret = float(est_ret_str.replace(',',''))
            except:
                est_ret = 0
            if est_ret > 1: ret_color = "🟢"
            elif est_ret > 0: ret_color = "🟡"
            else: ret_color = "🔴"
            
            gaya = r.get('Gaya','?')
            if "Day Trade" in gaya:
                gaya_label = "⏱️DT"
            elif "Swing Trade" in gaya:
                gaya_label = "📆SW"
            else:
                gaya_label = ""
            expander_title = f"{r.get('Saham','?')} {r.get('Harga','?')} {sig_icon} {r.get('Sinyal','?')} {gaya_label} Score: {r.get('Score','?')}"
            with st.expander(expander_title):
                st.markdown(f"**{sig_icon} {r.get('Sinyal','?')}**")
                st.caption(f"Score: {r.get('Score','?')} | Confidence: {r.get('Confidence','?')} ({conf_text}) | Risk-Adj: {r.get('RRR','?')}")
                st.divider()
                
                c1, c2 = st.columns(2)
                c1.metric("Coppock", r.get('Coppock','?'))
                c2.metric("Est. Return", f"{r.get('Est_Return','?')} {ret_color}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"""
                        <div style="margin-top: 0px;">
                            <label data-testid="stMetricLabel" style="color:rgb(255, 255, 255); font-size:14px; margin:0 0 4px 0; display:block;">Est. TP Besok</label>
                            <div data-testid="stMetricValue" style="color:rgb(0, 255, 204); font-size:24px; font-weight:700; line-height:1.2;">Rp {r.get('TP_Harga','?')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with c2:
                    st.markdown(
                        f"""
                        <div style="margin-top: 0px;">
                            <label data-testid="stMetricLabel" style="color:rgb(255, 255, 255); font-size:14px; margin:0 0 4px 0; display:block;">Est. SL Besok</label>
                            <div data-testid="stMetricValue" style="color:rgb(239, 68, 68); font-size:24px; font-weight:700; line-height:1.2;">Rp {r.get('SL_Harga','?')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.metric("Likuiditas", r.get('Likuiditas','?'), delta="/hari")

                ind1, ind2, ind3, ind4 = st.columns(4)
                ind1.metric("RSI-14", r.get('RSI','?'), delta=r.get('RSI_Status',''))
                ind2.metric("Vol Surge", r.get('Vol_Surge','?'), delta=r.get('VS_Status',''))
                ind3.metric("Z-Score", r.get('ZScore','?'), delta=r.get('ZS_Status',''))
                ind4.metric("Trend Cons.", r.get('Trend_Consistency','?'))
                
                b1, b2 = st.columns(2)
                b1.metric("Beta", r.get('Beta','?'))
                b2.metric("Momentum (5D)", r.get('Momentum','?'))
                
                st.caption(f"Regime: **{r.get('Rezim','?')}**")
                
                ai = r.get("AI_Insight", "").strip()
                if ai:
                    st.caption(f"💡 {ai[:150]}")
        if len(st.session_state.riwayat) > 10:
            st.caption(f"Menampilkan 10 dari {len(st.session_state.riwayat)} riwayat.")
    else:
        st.caption("Belum ada riwayat.")

    st.markdown("---")
    st.subheader("🧠 AI (Gemini)")

    def get_api_key():
        try: return st.secrets["GEMINI_API_KEY"]
        except KeyError: pass
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key: return env_key
        return st.session_state.get("gemini_api_key", "")

    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = get_api_key()

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key,
        placeholder="AIza...",
        help="Kunci API Gemini. Disimpan di secrets atau env."
    )
    if api_key: st.session_state.gemini_api_key = api_key

    ai_riwayat_btn = st.button("📊 Analisis Riwayat dgn AI", use_container_width=True)

    if st.button("🗑️ Hapus Semua Riwayat"):
        if os.path.isfile(RIWAYAT_FILE): os.remove(RIWAYAT_FILE)
        st.session_state.riwayat = []
        st.success("Riwayat dihapus!")

    # ---------- KALENDER BURSA (TIMEZONE WIB + JAM OPERASIONAL) ----------
    st.markdown("---")
    now_jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
    today_str = now_jkt.strftime("%Y-%m-%d")
    today_day = now_jkt.strftime("%A")
    current_hour = now_jkt.hour
    current_minute = now_jkt.minute
    current_year = now_jkt.strftime("%Y")

    st.subheader(f"📅 Kalender Bursa {current_year}")
    libur_bursa = {
        "2025-01-01": "Tahun Baru Masehi",
        "2025-01-29": "Tahun Baru Imlek 2576 Kongzili",
        "2025-03-14": "Hari Suci Nyepi (Tahun Baru Saka 1947)",
        "2025-04-18": "Wafat Yesus Kristus",
        "2025-05-01": "Hari Buruh Internasional",
        "2025-05-29": "Kenaikan Yesus Kristus",
        "2025-05-30": "Hari Raya Waisak 2569",
        "2025-06-06": "Idul Adha 1446 H",
        "2025-06-27": "Tahun Baru Islam 1447 H",
        "2025-08-17": "Hari Kemerdekaan Republik Indonesia",
        "2025-09-05": "Maulid Nabi Muhammad SAW 1447 H",
        "2025-12-25": "Hari Raya Natal",
        "2026-01-01": "Tahun Baru Masehi",
        "2026-02-17": "Tahun Baru Imlek 2577 Kongzili",
        "2026-03-03": "Hari Suci Nyepi (Tahun Baru Saka 1948)",
        "2026-04-03": "Wafat Yesus Kristus",
        "2026-05-01": "Hari Buruh Internasional",
        "2026-05-14": "Kenaikan Yesus Kristus",
        "2026-05-15": "Hari Raya Waisak 2570",
        "2026-05-25": "Idul Adha 1447 H",
        "2026-06-15": "Tahun Baru Islam 1448 H",
        "2026-08-17": "Hari Kemerdekaan Republik Indonesia",
        "2026-08-24": "Maulid Nabi Muhammad SAW 1448 H",
        "2026-12-25": "Hari Raya Natal",
    }

    def dalam_jam_perdagangan(hour, minute):
        sesi1 = (hour == 9 and minute >= 0) or (10 <= hour < 12) or (hour == 12 and minute == 0)
        sesi2 = (hour == 13 and minute >= 30) or (hour == 14) or (hour == 15 and minute == 0)
        return sesi1 or sesi2

    if today_str in libur_bursa:
        st.warning(f"Hari ini bursa **TUTUP**: {libur_bursa[today_str]}")
    elif today_day in ["Saturday", "Sunday"]:
        st.warning("Hari ini **AKHIR PEKAN**, bursa tutup.")
    elif dalam_jam_perdagangan(current_hour, current_minute):
        st.success("Bursa **TERBUKA** (Sesi 1: 09:00-12:00, Sesi 2: 13:30-15:00 WIB)")
    else:
        st.info("Bursa **TUTUP** (di luar jam perdagangan).")

    st.caption("Libur dalam 2 minggu ke depan:")
    future_libur = []
    for date_str, desc in libur_bursa.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        delta = (dt.date() - now_jkt.date()).days
        if 0 < delta <= 14:
            future_libur.append(f"- {dt.strftime('%d %b')}: {desc}")
    if future_libur:
        for item in future_libur:
            st.caption(item)
    else:
        st.caption("Tidak ada libur dalam 2 minggu.")

    st.markdown("---")
    st.caption("Data dari Yahoo Finance. Bukan rekomendasi investasi.")

# ==================== FUNGSI DATA & INDIKATOR ====================
@st.cache_data(ttl=3600)
def load_stock_data(ticker, period="2y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=3600)
def load_ihsg_data(period="2y", interval="1d"):
    df = yf.download("^JKSE", period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df

def compute_adx_series(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    up = high.diff(); down = -low.diff()
    plus_dm = np.where((up>down)&(up>0), up, 0.0)
    minus_dm = np.where((down>up)&(down>0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = (abs(plus_di-minus_di)/(plus_di+minus_di))*100
    return dx.ewm(alpha=1/period, adjust=False).mean()

def get_google_news_rss(query_str, num=5):
    if not RSS_AVAILABLE: return [], "RSS tidak tersedia"
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query_str)}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)
        news = []
        for e in feed.entries[:num]:
            title = e.get('title','').strip()
            summary = re.sub('<[^<]+?>','',e.get('summary',''))
            news.append({'title':title,'summary':summary,'source':'Google News'})
        return news, None
    except Exception as e: return [], str(e)

def get_yahoo_search_news(query_str, num=5):
    try:
        items = yf.Search(query_str).news or []
        news = []
        for item in items[:num]:
            inner = item.get('content') or item
            title = (inner.get('title') or inner.get('shortTitle') or inner.get('headline') or '')
            summary = (inner.get('summary') or inner.get('longSummary') or inner.get('description') or '')
            if title: news.append({'title':title,'summary':summary,'source':'Yahoo Search'})
        return news, None
    except: return [], "Yahoo Search gagal"

def filter_relevant(news_list, ticker):
    keywords = [ticker.lower(),'saham','ihsg','bei','idx']
    filtered = [n for n in news_list if any(k in (n['title']+n['summary']).lower() for k in keywords)]
    return filtered if filtered else news_list

def analyze_sentiment_weighted(news_items, translator):
    if not SENTIMENT_AVAILABLE or not news_items: return 0.0
    analyzer = SentimentIntensityAnalyzer()
    total_w, w_sum = 0, 0
    for i, item in enumerate(news_items):
        text = f"{item['title']}. {item['summary']}" if item['summary'] else item['title']
        if any(ord(c)>127 for c in text) and translator:
            try: text = translator.translate(text)
            except: pass
        score = analyzer.polarity_scores(text)['compound']
        weight = 1/(i+1)
        w_sum += score*weight; total_w += weight
    return w_sum/total_w if total_w>0 else 0.0

def estimate_theta_ou(close_series):
    log_price = np.log(close_series.dropna())
    log_lag = log_price.shift(1).dropna()
    diff = log_price.diff().dropna()
    common_idx = diff.index.intersection(log_lag.index)
    if len(common_idx)<20: return 0.05
    y = diff.loc[common_idx].values
    X = np.vstack([np.ones(len(common_idx)), log_lag.loc[common_idx].values]).T
    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    theta = -coeff[1] if coeff[1]<0 else 0.05
    return theta

REGIME_INFO = {
    "Strong Bullish 🚀": "Tren naik kuat dengan momentum tinggi.",
    "Bullish 📈": "Tren naik stabil. Kondisi sehat untuk akumulasi.",
    "Panic Sell 🚨": "Penurunan tajam, sering oversold.",
    "Bearish 🔻": "Tren turun terkendali.",
    "Early Recovery 🔄": "Harga di atas EMA20 tapi EMA20 < EMA50.",
    "Distribution 📉": "Harga di bawah EMA20, EMA20 > EMA50.",
    "Konsolidasi Tren ↔️": "Trending namun harga bolak-balik di EMA.",
    "Bullish Accumulation 🏗️": "Sideways dengan harga > EMA.",
    "Bearish Accumulation 🧊": "Sideways di bawah EMA.",
    "Sideways Bias Naik ↗️": "Sideways cenderung naik.",
    "Sideways Bias Turun ↘️": "Sideways cenderung turun.",
    "Sideways Normal ↔️": "Sideways moderat, tunggu katalis."
}

# ==================== PROSES ANALISIS ====================
if run_btn:
    if not ticker_input:
        st.warning("⚠️ Kode saham tidak boleh kosong!"); st.stop()

    with st.spinner("🤖 Mengunduh data dan memproses analitika kuantitatif..."):
        if "Day Trade" in trading_style:
            df = load_stock_data(ticker_input, period="5d", interval="5m")
            # Untuk day trade, fallback interval jika data 5m tidak tersedia
            if df.empty or len(df) < 20:
                st.warning("Data 5 menit tidak lengkap, mencoba interval 15 menit...")
                df = load_stock_data(ticker_input, period="5d", interval="15m")
            if df.empty or len(df) < 20:
                st.warning("Data 15 menit tidak lengkap, mencoba interval 30 menit...")
                df = load_stock_data(ticker_input, period="5d", interval="30m")
            if df.empty or len(df) < 20:
                st.warning("Data 30 menit tidak lengkap, menggunakan interval 60 menit...")
                df = load_stock_data(ticker_input, period="5d", interval="60m")
            df_ihsg = load_ihsg_data(period="5d", interval="5m")
        else:
            df = load_stock_data(ticker_input, period="2y", interval="1d")
            df_ihsg = load_ihsg_data(period="2y", interval="1d")
        
        if df.empty: st.error("❌ Data tidak ditemukan untuk ticker tersebut."); st.stop()

        harga_terakhir = float(df['Close'].iloc[-1])
        returns = df['Close'].pct_change().dropna()
        if len(returns)<20: st.error("❌ Data historis kurang untuk analisa kuantitatif."); st.stop()

        # ============ INDIKATOR ============
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['ADX'] = compute_adx_series(df)
        df['Mom5D'] = df['Close'].pct_change(5)*100
        df['ZScore'] = (df['Close']-df['Close'].rolling(20).mean())/df['Close'].rolling(20).std()
        df['Vol_MA20'] = df['Volume'].rolling(20).mean() if 'Volume' in df.columns else 0

        # ============ FUNDAMENTAL ============
        try: ticker_info = yf.Ticker(ticker_input).info
        except: ticker_info = {}
        mc = ticker_info.get('marketCap')
        per = ticker_info.get('trailingPE') or ticker_info.get('forwardPE')
        pbv = ticker_info.get('priceToBook')
        roe = ticker_info.get('returnOnEquity')
        de = ticker_info.get('debtToEquity')

        # ============ BERITA & SENTIMEN ============
        news_pool = []
        translator_en = GoogleTranslator(source='auto', target='en') if TRANSLATOR_AVAILABLE else None
        translator_id = GoogleTranslator(source='auto', target='id') if TRANSLATOR_AVAILABLE else None
        rss, _ = get_google_news_rss(f"{ticker_raw} saham")
        if rss: news_pool.extend(rss)
        ysearch, _ = get_yahoo_search_news(f"{ticker_raw} saham")
        if ysearch: news_pool.extend(ysearch)
        news_pool = filter_relevant(news_pool, ticker_raw)
        seen = set(); unique_news = []
        for n in news_pool:
            if n['title'] not in seen: seen.add(n['title']); unique_news.append(n)
            if len(unique_news)>=5: break
        avg_sentiment = analyze_sentiment_weighted(unique_news, translator_en)
        headlines = [n['title'] for n in unique_news]
        sources = [n['source'] for n in unique_news]
        translated = []
        for n in unique_news:
            if TRANSLATOR_AVAILABLE and translator_id:
                try: translated.append(translator_id.translate(n['title']))
                except: translated.append("")
            else: translated.append("")
        sentimen_status = "Positif 🟢" if avg_sentiment>=0.05 else ("Negatif 🔴" if avg_sentiment<=-0.05 else "Netral ⚪")

        # ============ THRESHOLD HISTORIS ============
        split_idx = max(126, len(df)-126)
        df_thresh = df.iloc[:split_idx]
        returns_thresh = df_thresh['Close'].pct_change().dropna()
        adx_threshold = np.percentile(df_thresh['ADX'].dropna(),75) if not df_thresh['ADX'].dropna().empty else 20
        z_oversold_th = -1.5
        mom_median_th = np.percentile(df_thresh['Mom5D'].dropna(),50) if not df_thresh['Mom5D'].dropna().empty else 0.0

        def t_loglike(p,d):
            if p[0]<=2 or p[2]<=0: return np.inf
            return -np.sum(student_t.logpdf(d,p[0],p[1],p[2]))
        res = minimize(t_loglike, [5, returns_thresh.mean(), returns_thresh.std()],
                       bounds=[(2.1,100),(-0.1,0.1),(1e-6,None)], args=(returns_thresh,), method='L-BFGS-B')
        df_est, t_loc, t_scale = res.x if res.success else (5, returns_thresh.mean(), returns_thresh.std())

        # ============ REGIME ============
        def get_regime_row(row):
            h,e20,e50,a,z,m = row['Close'],row['EMA20'],row['EMA50'],row['ADX'],row['ZScore'],row['Mom5D']
            if a>adx_threshold:
                if h>e20 and e20>e50:
                    return ("Strong Bullish 🚀","RISK-ON 🔥") if (m>mom_median_th or z>z_oversold_th) else ("Bullish 📈","RISK-ON 🔥")
                elif h<e20 and e20<e50:
                    return ("Panic Sell 🚨","RISK-OFF 🛑") if (m<mom_median_th or z<z_oversold_th) else ("Bearish 🔻","RISK-OFF 🛑")
                elif h>e20 and e20<e50: return ("Early Recovery 🔄","TRANSISI ⚠️")
                elif h<e20 and e20>e50: return ("Distribution 📉","TRANSISI ⚠️")
                else: return ("Konsolidasi Tren ↔️","NEUTRAL ⚖️")
            else:
                if h>e20 and e20>e50: return ("Bullish Accumulation 🏗️","NEUTRAL ⚖️")
                elif h<e20 and e20<e50: return ("Bearish Accumulation 🧊","NEUTRAL ⚖️")
                elif h>e20 and e20<e50: return ("Sideways Bias Naik ↗️","NEUTRAL ⚖️")
                elif h<e20 and e20>e50: return ("Sideways Bias Turun ↘️","NEUTRAL ⚖️")
                else: return ("Sideways Normal ↔️","NEUTRAL ⚖️")
        regime, ihsg_cond = get_regime_row(df.iloc[-1])
        adx = df['ADX'].iloc[-1]

        # ============ BETA ============
        try:
            if not df_ihsg.empty:
                ihsg_ret = df_ihsg['Close'].pct_change().dropna()
                common = returns.index.intersection(ihsg_ret.index)
                if len(common) > 20:
                    beta_ihsg = np.cov(returns.loc[common], ihsg_ret.loc[common])[0,1] / np.var(ihsg_ret.loc[common])
                else:
                    beta_ihsg = 1.0
            else:
                beta_ihsg = 1.0
        except: beta_ihsg = 1.0

        # ============ ATR & RSI ============
        df['TR'] = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift()).abs(),
            (df['Low'] - df['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr14 = df['TR'].rolling(14).mean().iloc[-1]
        atr_pct = (atr14 / harga_terakhir) * 100

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean().iloc[-1]
        avg_loss = loss.rolling(14).mean().iloc[-1]
        if avg_loss is None or avg_loss == 0:
            rsi14 = 100.0
        else:
            rsi14 = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))

        # === MULTIPLIER TP/SL ===
        tp_mult = 2.0
        sl_mult = 1.0
        if adx > 30 and 30 < rsi14 < 70:
            tp_mult = 2.5; sl_mult = 1.0
        elif adx < 20:
            tp_mult = 1.5; sl_mult = 0.75
        if rsi14 > 70 or rsi14 < 30:
            tp_mult = min(tp_mult, 1.5)

        sl_pct = sl_mult * atr_pct
        tp_pct = tp_mult * atr_pct
        sl_harga = harga_terakhir * (1 - sl_pct/100)
        tp_harga = harga_terakhir * (1 + tp_pct/100)

        rrr = tp_pct / sl_pct if sl_pct > 0 else 0
        if rrr >= 2.0: rrr_status = "Sangat Baik (≥ 2.0) 🟢"
        elif rrr >= 1.5: rrr_status = "Baik (1.5 - 2.0) 🟢"
        elif rrr >= 1.0: rrr_status = "Cukup (1.0 - 1.5) 🟡"
        else: rrr_status = "Buruk (< 1.0) 🔴"

        # ============ PIVOT (DENGAN VALIDASI) ============
        hi = lo = cl = None
        for i in range(1, min(6, len(df))):
            row = df.iloc[-i]
            h_val = float(row['High']); l_val = float(row['Low']); c_val = float(row['Close'])
            if h_val != l_val and h_val > 0 and l_val > 0:
                hi, lo, cl = h_val, l_val, c_val
                break
        if hi is None or lo is None:
            hi = float(df['High'].iloc[-1]); lo = float(df['Low'].iloc[-1]); cl = float(df['Close'].iloc[-1])
        if hi == lo:
            pp = r1 = s1 = r2 = s2 = cl
        else:
            pp = (hi + lo + cl) / 3
            r1 = 2 * pp - lo
            s1 = 2 * pp - hi
            r2 = pp + (hi - lo)
            s2 = pp - (hi - lo)
        if len(df) >= 21:
            res20 = float(df['High'].iloc[-21:-1].max())
        else:
            res20 = float(df['High'].max())
        breakout = "YES (🔥)" if cl > res20 else "NO"

        # ============ SINYAL ============
        def generate_signals_vectorized(dataframe, mom_th):
            score = pd.Series(0, index=dataframe.index)
            is_uptrend = (dataframe['Close']>dataframe['EMA20']) & (dataframe['EMA20']>dataframe['EMA50'])
            score += is_uptrend.astype(int)*2
            score += (dataframe['Mom5D']>mom_th).astype(int)
            if 'Volume' in dataframe.columns: score += (dataframe['Volume']>dataframe['Vol_MA20']).astype(int)
            sig = pd.Series("🚨 AVOID", index=dataframe.index)
            sig[score==1] = "⏸️ HOLD / WAIT"; sig[score>=2] = "⚡ BUY (TACTICAL)"; sig[score>=3] = "🔥 STRONG BUY"
            sig[(dataframe['ADX']<20) & sig.str.contains("BUY")] = "⏸️ HOLD / WAIT"
            sig[(dataframe['ZScore']<-1.5) & (dataframe['Close']<dataframe['EMA20'])] = "⚡ BUY (TACTICAL)"
            return sig
        df['Signal'] = generate_signals_vectorized(df, mom_median_th)
        signal = df['Signal'].iloc[-1]

        # ============ ENTRY ZONE ADAPTIF ============
        if s1 >= harga_terakhir * 0.95:
            entry_low = s1
        else:
            entry_low = harga_terakhir * (1 - atr_pct/100)
        if "STRONG BUY" in signal:
            entry_high = harga_terakhir
        else:
            entry_high = harga_terakhir * (1 - 0.5 * atr_pct/100)
        entry_zone = f"Rp {entry_low:,.0f} - {entry_high:,.0f}"

        # ============ BACKTEST ============
        df_back = df.iloc[-126:].copy()
        trades, daily_returns = [], []
        in_position, entry_price = False, 0.0
        for i in range(len(df_back)):
            curr_sig = df_back['Signal'].iloc[i]
            curr_close = float(df_back['Close'].iloc[i])
            prev_close = float(df_back['Close'].iloc[i-1]) if i>0 else curr_close
            if in_position:
                daily_returns.append((curr_close-prev_close)/prev_close if prev_close else 0)
                if "AVOID" in curr_sig or i==len(df_back)-1:
                    trades.append((curr_close-entry_price)/entry_price); in_position=False
            else:
                daily_returns.append(0.0)
                if "BUY" in curr_sig: in_position, entry_price = True, curr_close
        if trades:
            win_bt = sum(1 for r in trades if r>0)/len(trades)
            loss_trades = [r for r in trades if r<0]; profit_trades = [r for r in trades if r>0]
            pf_bt = abs(sum(profit_trades)/sum(loss_trades)) if loss_trades else np.inf
            avg_bt = np.mean(trades)
            equity = np.cumprod([1+r for r in trades])
            max_dd_bt = float(np.min(equity/np.maximum.accumulate(equity)-1)*100) if len(equity) else 0
            daily_ret = np.array(daily_returns)
            sharpe_bt = (daily_ret.mean()/daily_ret.std())*np.sqrt(252) if daily_ret.std() else 0
            trades_bt = len(trades)
        else: win_bt=pf_bt=avg_bt=max_dd_bt=sharpe_bt=trades_bt=0

        # ============ KELLY ============
        roll_max_th = df_thresh['Close'].cummax()
        drawdown_th = (df_thresh['Close']-roll_max_th)/roll_max_th
        max_dd = float(drawdown_th.min()*100)
        max_dd_30 = float(drawdown_th.tail(30).min()*100) if len(drawdown_th)>=30 else max_dd
        if trades_bt>=2: win_r,avg_g,avg_l = win_bt, np.mean(profit_trades) if profit_trades else 0.01, abs(np.mean(loss_trades)) if loss_trades else 0.01
        else:
            win_r = len(returns_thresh[returns_thresh>0])/len(returns_thresh)
            avg_g = returns_thresh[returns_thresh>0].mean() if win_r>0 else 0.01
            avg_l = abs(returns_thresh[returns_thresh<0].mean()) if len(returns_thresh[returns_thresh<0]) else 0.01
        wl = avg_g/avg_l if avg_l else 1
        kelly_raw = win_r - (1-win_r)/wl
        ret_skew = float(skew(returns_thresh)); ret_kurt = float(kurtosis(returns_thresh, fisher=True))
        kurt_penalty = 0.5 if ret_kurt>3 else 1.0
        kelly_adj = min(0.25, max(0.0, kelly_raw*0.3*(0.5 if ret_skew<-0.5 else 1)*kurt_penalty))

        # ============ MONTE CARLO ============
        n_sim,n_days = 2000,30
        latest_vol = np.sqrt(df['Close'].pct_change().ewm(alpha=0.06).var().iloc[-1])
        scale_corrected = latest_vol/np.sqrt(df_est/(df_est-2)) if df_est>2 else latest_vol
        theta_ou = estimate_theta_ou(df['Close'])
        locked_log_mean20 = np.log(df['Close']).tail(20).mean()
        paths = np.zeros((n_days,n_sim)); current_log = np.ones((1,n_sim))*np.log(harga_terakhir)
        for day in range(n_days):
            inov = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=n_sim)
            next_log = current_log[-1] + theta_ou*(locked_log_mean20-current_log[-1]) + inov
            current_log = np.vstack([current_log, next_log]); paths[day] = np.exp(next_log)
        mu_ou = theta_ou*(locked_log_mean20-np.log(harga_terakhir))
        est_besok = float(np.exp(np.log(harga_terakhir)+mu_ou))
        sim_h1 = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=2000)
        prices_besok = harga_terakhir*np.exp(mu_ou+sim_h1)
        low_est,up_est = float(np.percentile(prices_besok,25)), float(np.percentile(prices_besok,75))
        hit_tp = (np.any(paths>=r1,axis=0).sum()/n_sim)*100
        hit_sl = (np.any(paths<=s2,axis=0).sum()/n_sim)*100
        prob_bull = ((mu_ou+sim_h1>0).sum()/2000)*100

        # ============ METRIK TAMBAHAN UNTUK RIWAYAT ============
        if "STRONG BUY" in signal: signal_score = 0.7 + (prob_bull / 200)
        elif "BUY" in signal: signal_score = 0.4 + (prob_bull / 200)
        elif "HOLD" in signal: signal_score = 0.2 + (prob_bull / 300)
        else: signal_score = max(0, (prob_bull - 30) / 100)
        signal_score = min(1.0, max(0.0, signal_score))
        confidence = min(0.99, 0.5 + (signal_score * 0.5) + (win_bt - 0.5) * 0.1)
        if confidence is None or np.isnan(confidence): confidence = 0.5
        
        trend_consistency = np.mean([1 if (df['Close'].iloc[-i] > df['Close'].iloc[-i-1]) == (df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]) else 0 for i in range(1, 11)]) * 100
        if np.isnan(trend_consistency): trend_consistency = 50.0
        
        avg_vol_5 = df['Volume'].iloc[-5:].mean()
        avg_vol_20 = df['Volume'].iloc[-20:].mean()
        if avg_vol_20 > 0: vol_surge_pct = ((avg_vol_5 / avg_vol_20) - 1) * 100
        else: vol_surge_pct = 0.0
        
        avg_value = (df['Volume'].iloc[-5:] * df['Close'].iloc[-5:]).mean()
        if np.isnan(avg_value): avg_value = 0.0
        if avg_value >= 1e9: likuiditas_str = f"Rp {avg_value/1e9:.2f} M"
        elif avg_value >= 1e6: likuiditas_str = f"Rp {avg_value/1e6:.0f} Jt"
        elif avg_value >= 1e3: likuiditas_str = f"Rp {avg_value/1e3:.0f} rb"
        else: likuiditas_str = f"Rp {avg_value:,.0f}"
            
        if rsi14 > 70: rsi_status = "Overbought"
        elif rsi14 < 30: rsi_status = "Oversold"
        else: rsi_status = "Normal"
        
        zscore_val = df['ZScore'].iloc[-1]
        if pd.isna(zscore_val): zscore_val = 0.0
        if zscore_val > 2: zs_status = "Overbought"
        elif zscore_val < -2: zs_status = "Oversold"
        else: zs_status = "Normal"
        
        if vol_surge_pct > 50: vs_status = "Tinggi"
        elif vol_surge_pct < -30: vs_status = "Rendah"
        else: vs_status = "Normal"
        
        coppock_val, coppock_prev = coppock_curve(df['Close'].values)
        coppock_rising = coppock_val > coppock_prev
        coppock_turning_up = coppock_rising and coppock_prev <= 0
        if coppock_turning_up: coppock_status = "Turning Up"
        elif coppock_rising: coppock_status = "Rising"
        else: coppock_status = "Falling"

        ringkasan = {
            "Waktu": datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M"),
            "Saham": ticker_raw,
            "Harga": f"{harga_terakhir:,.0f}",
            "Sinyal": signal,
            "Estimasi": f"{est_besok:,.0f}",
            "Prob Naik": f"{prob_bull:.1f}%",
            "RRR": f"{rrr:.2f}",
            "Sentimen": f"{avg_sentiment:.2f} ({sentimen_status})",
            "Rezim": regime,
            "TP%": f"{tp_pct:.1f}",
            "SL%": f"{sl_pct:.1f}",
            "AI_Insight": "",
            "Score": f"{signal_score:.3f}",
            "Confidence": f"{confidence:.0%}",
            "Coppock": coppock_status,
            "Est_Return": f"{est_besok:,.0f}",
            "TP_Harga": f"{tp_harga:,.0f}",
            "SL_Harga": f"{sl_harga:,.0f}",
            "Likuiditas": likuiditas_str,
            "RSI": f"{rsi14:.1f}",
            "RSI_Status": rsi_status,
            "Vol_Surge": f"{vol_surge_pct:+.0f}%",
            "VS_Status": vs_status,
            "ZScore": f"{zscore_val:.2f}",
            "ZS_Status": zs_status,
            "Trend_Consistency": f"{trend_consistency:.0f}%",
            "Beta": f"{beta_ihsg:.2f}",
            "Momentum": f"{df['Mom5D'].iloc[-1]:.2f}%",
            "Entry_Zone": entry_zone,
            "Gaya": trading_style
        }

    # ==================== TAMPILAN UTAMA ====================
    st.title("📊 Quant & Risk Engine Pro")
    st.write("Algoritma kuantitatif + Berita + Backtest + AI + Grafik Interaktif + Fundamental")
    st.success(f"✅ Analisis Berhasil: {ticker_input} | Closing Price: Rp {harga_terakhir:,.0f}".replace(",","."))

    col1,col2,col3 = st.columns(3)
    col1.metric("Sinyal Eksekusi", signal)
    col2.metric("Estimasi Besok", f"Rp {est_besok:,.0f}".replace(",","."),
                f"50% range: Rp {low_est:,.0f} - {up_est:,.0f}".replace(",","."))
    col3.metric("Prob. Naik Besok", f"{prob_bull:.1f}%")

    if PLOTLY_AVAILABLE:
        st.header("📈 Chart Harga & Sinyal")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#00ffcc')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', line=dict(color='#f59e0b', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50', line=dict(color='#ef4444', dash='dot')))
        buy_signals = df_back[df_back['Signal'].str.contains("BUY")]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers',
                                 marker=dict(symbol='triangle-up', size=10, color='#10b981'), name='Buy Signal'))
        for lvl,lbl,clr in [(r1,'R1','orange'),(s1,'S1','red'),(pp,'PP','gray')]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=clr, annotation_text=lbl, annotation_position="right")
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # --- RINGKASAN EKSEKUTIF ---
    st.markdown("---"); st.header("📋 Ringkasan Eksekutif & Rekomendasi")
    if rrr < 1.0 and ("BUY" in signal):
        ac,ai = "#ef4444","⚠️"
        at = f"• <b>KONDISI:</b> Sinyal {signal} tapi <b>RRR Buruk ({rrr:.2f})</b><br>• <b>REKOMENDASI:</b> WAIT & SEE<br>• <b>LANGKAH:</b> Tunda entry, tunggu setup lebih baik."
    elif "STRONG BUY" in signal:
        ac,ai = "#10b981","🟢"
        at = f"• <b>KONDISI:</b> Tren Kuat & Akumulasi Volume<br>• <b>REKOMENDASI:</b> AGGRESSIVE BUY<br>• <b>LANGKAH:</b> Entry zone {entry_zone}, SL {sl_harga:,.0f} (-{sl_pct:.1f}%), TP {tp_harga:,.0f} (+{tp_pct:.1f}%)."
    elif "BUY" in signal:
        ac,ai = "#f59e0b","🟡"
        at = f"• <b>KONDISI:</b> Tren Valid, RRR {rrr:.2f} ({rrr_status})<br>• <b>REKOMENDASI:</b> BUY ON WEAKNESS<br>• <b>LANGKAH:</b> Entry di zona {entry_zone}, SL {sl_harga:,.0f}, TP {tp_harga:,.0f}."
    elif "HOLD" in signal:
        ac,ai = "#3b82f6","🔵"
        at = f"• <b>KONDISI:</b> Konsolidasi / Transisi<br>• <b>REKOMENDASI:</b> HOLD<br>• <b>LANGKAH:</b> Jangan tambah posisi, pantau SL."
    else:
        ac,ai = "#ef4444","🔴"
        at = f"• <b>KONDISI:</b> Risiko Penurunan / Distribusi<br>• <b>REKOMENDASI:</b> AVOID / LIQUIDATE<br>• <b>LANGKAH:</b> Amankan modal."
    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown(f'''
            <div class="summary-card">
                <div class="section-title">📌 Profil Risiko (Kontekstual)</div>
                <div class="summary-item">🛡️ <b>Stop Loss:</b> -{sl_pct:.1f}% (Rp {sl_harga:,.0f})</div>
                <div class="summary-item">🎯 <b>Take Profit:</b> +{tp_pct:.1f}% (Rp {tp_harga:,.0f})</div>
                <div class="summary-item">⚖️ <b>Risk:Reward:</b> 1 : {rrr:.2f} ({rrr_status})</div>
                <div class="summary-item" style="color:#8892b0;">📊 ADX {adx:.1f} | RSI {rsi14:.1f} | ATR {atr_pct:.2f}%</div>
                <div class="summary-item">🏷️ <b>Rezim:</b> {regime} | {ihsg_cond}</div>
                <div class="summary-item">🛡️ <b>Alokasi Maks (Kelly):</b> {kelly_adj*100:.1f}% dari Total Ekuitas</div>
            </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="action-card" style="border-left-color: {ac};"><div class="section-title">{ai} Panduan Eksekusi Trader</div><div class="summary-item" style="font-size:15px;margin-top:8px;line-height:1.6;">{at}</div><hr style="border-color:#334155;margin:15px 0;"><div style="color:#94a3b8;font-size:13px;">⚠️ <i>Disclaimer: Hasil pengujian berbasis permodelan matematika probabilitas kuantitatif historis. Keputusan akhir eksekusi modal tetap merupakan tanggung jawab penuh masing-masing investor.</i></div></div>', unsafe_allow_html=True)

    # --- DETAIL EXPANDER ---
    with st.expander("🔍 Lihat Detail Analisis (Berita, Fundamental, Backtest, dll)"):
        st.subheader("📰 Sentimen Berita Terbobot"); c1,c2=st.columns([1,2])
        c1.metric("Sentimen Skor",f"{avg_sentiment:.2f}",sentimen_status)
        with c2:
            st.markdown("**5 Berita Utama Pasar:**")
            for i,h in enumerate(headlines):
                src=sources[i] if i<len(sources) else ""; t=translated[i] if i<len(translated) else ""
                st.markdown(f"{i+1}. **{h}** <span class='source'>({src})</span>",unsafe_allow_html=True)
                if t and t!=h: st.markdown(f"<span class='translated'>🇮🇩 {t}</span>",unsafe_allow_html=True)
        st.divider()
        st.subheader("🧬 Regime Pasar & Volatilitas"); m1,m2,m3=st.columns(3)
        m1.metric("Market Regime",regime); m2.metric("Kondisi Makro IHSG",ihsg_cond); m3.metric("ADX Adaptif",f"{adx:.1f} (Thresh: {adx_threshold:.1f})")
        st.markdown(f"**Insight Regime:** {REGIME_INFO.get(regime,'Regime tidak terdefinisi.')}")
        st.divider()
        st.subheader("📊 Metrik Fundamental Saham (IDX)")
        if ticker_info:
            def clean_val(v,f="{:.2f}"): return "N/A" if v is None else f.format(v)
            table_html=f"<table class='fundamental-table'><tr><td>Market Cap</td><td>{clean_val(mc,'{:,.0f} IDR')}</td></tr><tr><td>PER</td><td>{clean_val(per,'{:.2f}x')}</td></tr><tr><td>PBV</td><td>{clean_val(pbv,'{:.2f}x')}</td></tr><tr><td>ROE</td><td>{clean_val(roe*100 if roe else None,'{:.1f}%')}</td></tr><tr><td>D/E</td><td>{clean_val(de,'{:.2f}%')}</td></tr></table>"
            st.markdown(table_html,unsafe_allow_html=True)
            interpretation_items=[]
            if mc:
                if mc>=1e13: mct=f"Market Cap Rp {mc:,.0f} tergolong sangat besar (Mega Cap)."
                elif mc>=1e12: mct=f"Market Cap Rp {mc:,.0f} tergolong besar (Blue Chip)."
                elif mc>=1e10: mct=f"Market Cap Rp {mc:,.0f} tergolong menengah (Mid Cap)."
                else: mct=f"Market Cap Rp {mc:,.0f} tergolong kecil (Small Cap)."
            else: mct="Market Cap tidak tersedia."
            interpretation_items.append(f"<li><b>Market Cap:</b> {mct}</li>")
            if per:
                if per<10: pt=f"PER {per:.2f}x tergolong rendah (potensi undervalue)."
                elif per<20: pt=f"PER {per:.2f}x moderat."
                else: pt=f"PER {per:.2f}x tergolong tinggi (premium)."
            else: pt="PER tidak tersedia."
            interpretation_items.append(f"<li><b>PER:</b> {pt}</li>")
            if pbv:
                if pbv<1: pbt=f"PBV {pbv:.2f}x di bawah 1 (di bawah nilai buku, bisa undervalue)."
                elif pbv<3: pbt=f"PBV {pbv:.2f}x moderat."
                else: pbt=f"PBV {pbv:.2f}x tinggi (premium)."
            else: pbt="PBV tidak tersedia."
            interpretation_items.append(f"<li><b>PBV:</b> {pbt}</li>")
            if roe:
                roep=roe*100
                if roep>20: rt=f"ROE {roep:.1f}% sangat baik (profitabilitas tinggi)."
                elif roep>10: rt=f"ROE {roep:.1f}% cukup baik."
                else: rt=f"ROE {roep:.1f}% rendah."
            else: rt="ROE tidak tersedia."
            interpretation_items.append(f"<li><b>ROE:</b> {rt}</li>")
            if de:
                if de>1: dt=f"D/E {de:.2f} tinggi (leverage tinggi, risiko lebih besar)."
                elif de>0.5: dt=f"D/E {de:.2f} moderat."
                else: dt=f"D/E {de:.2f} rendah (konservatif)."
            else: dt="D/E tidak tersedia."
            interpretation_items.append(f"<li><b>D/E:</b> {dt}</li>")
            st.markdown(f'<div style="background-color:#1e293b;border-radius:12px;padding:15px;margin-top:15px;color:#cbd5e1;font-size:14px;"><b style="color:#00ffcc;">📝 Interpretasi Metrik:</b><ul style="margin-top:8px;padding-left:20px;">{"".join(interpretation_items)}</ul></div>',unsafe_allow_html=True)
        else: st.warning("⚠️ Data fundamental finansial tidak tersedia.")
        st.divider()
        st.subheader("🎯 Target Pivot & Support/Resistance"); p1,p2,p3,p4,p5=st.columns(5)
        p1.metric("R2",f"Rp {r2:,.0f}".replace(",",".")); p2.metric("R1",f"Rp {r1:,.0f}".replace(",",".")); p3.metric("Pivot",f"Rp {pp:,.0f}".replace(",","."))
        p4.metric("S1",f"Rp {s1:,.0f}".replace(",",".")); p5.metric("S2",f"Rp {s2:,.0f}".replace(",","."))
        st.write(f"Kondisi Breakout 20 Hari: **{breakout}**"); st.divider()
        st.subheader("🔮 Sinyal Kuantitatif & Hasil Backtest (6 Bulan)"); t1,t2,t3,t4,t5=st.columns(5)
        t1.metric("Sinyal",signal); t2.metric("Estimasi Besok",f"Rp {est_besok:,.0f}".replace(",","."))
        t3.metric("Entry Zone", entry_zone); t4.metric("Take Profit",f"Rp {tp_harga:,.0f} (+{tp_pct:.1f}%)".replace(",","."), "Target Resist R1")
        t5.metric("Stop Loss",f"Rp {sl_harga:,.0f} (-{sl_pct:.1f}%)".replace(",","."), "Proteksi Batas S2")
        st.markdown("**Hasil Backtest Stateful Tracking 126 Hari:**"); b1,b2,b3,b4,b5,b6=st.columns(6)
        b1.metric("Win Rate",f"{win_bt:.1%}" if trades_bt else "N/A"); b2.metric("Profit Factor",f"{pf_bt:.2f}" if trades_bt and pf_bt!=np.inf else "N/A")
        b3.metric("Avg Return/Trade",f"{avg_bt:.2%}" if trades_bt else "N/A"); b4.metric("Max DD Strat",f"{max_dd_bt:.2f}%" if trades_bt else "N/A")
        b5.metric("Sharpe",f"{sharpe_bt:.2f}" if trades_bt else "N/A"); b6.metric("Total Trades",trades_bt)
        st.divider()
        st.subheader("🛡️ Manajemen Risiko Portofolio (Kelly)"); rc1,rc2=st.columns(2)
        rc1.metric("Alokasi Maks (Kelly)",f"{kelly_adj*100:.1f}%"); rc2.metric("Beta IHSG",f"{beta_ihsg:.2f}x")
        st.markdown(f"**Interpretasi:** Berdasarkan Win Rate **{win_bt:.1%}**, maksimal alokasi **{kelly_adj*100:.1f}%** dari total ekuitas.")
        st.markdown(f"Max DD Historis: `{max_dd:.2f}%` | DD 30 Hari: `{max_dd_30:.2f}%`")
        st.divider()
        st.subheader("🎲 Simulasi Monte Carlo Ornstein-Uhlenbeck"); pr1,pr2,pr3=st.columns(3)
        pr1.metric("Prob. Naik Besok",f"{prob_bull:.1f}%"); pr2.metric("Prob. Sentuh R1 (30H)",f"{hit_tp:.1f}%"); pr3.metric("Prob. Sentuh S2 (30H)",f"{hit_sl:.1f}%")

    # ══════════════════════════════════════════════════════════
    # V12 ADAPTIVE ENGINE – EXPANDER & LOGIC (DENGAN INSIGHT)
    # ══════════════════════════════════════════════════════════
    with st.expander("🧬 V12 Adaptive Engine (Coppock, Self‑Learning)", expanded=True):
        st.info(
            "⚙️ **Bagian ini adalah otak adaptif dari QuantRisk Pro.** "
            "Engine secara otomatis mempelajari akurasi setiap faktor teknikal berdasarkan riwayat analisis kamu. "
            "Semakin sering suatu ticker dianalisis, semakin akurat bobot yang dihasilkan."
        )

        st.markdown("### 📈 Coppock Curve & Beta IHSG")
        if coppock_turning_up:
            coppock_insight = "🟢 **Turning Up** – Sinyal awal akumulasi. Momentum bullish jangka panjang mulai terbentuk, potensi tren naik."
        elif coppock_rising:
            coppock_insight = "🟢 **Rising** – Tren bullish jangka panjang masih sehat. Akumulasi masih berlangsung."
        else:
            coppock_insight = "🔴 **Falling** – Momentum bullish melemah. Waspadai potensi koreksi atau perubahan tren."

        if beta_ihsg > 1.2:
            beta_insight = f"⚠️ **Beta Tinggi ({beta_ihsg:.2f})** – Saham lebih volatile dari IHSG. Cocok untuk *trading agresif*, namun risikonya lebih besar saat pasar turun."
        elif beta_ihsg > 0.8:
            beta_insight = f"✅ **Beta Moderat ({beta_ihsg:.2f})** – Pergerakan selaras dengan IHSG. Cocok untuk *swing trading*."
        else:
            beta_insight = f"🛡️ **Beta Rendah ({beta_ihsg:.2f})** – Saham defensif, lebih stabil dari IHSG. Cocok untuk *investasi jangka panjang*."

        col_cop1, col_cop2 = st.columns(2)
        with col_cop1:
            st.metric("Coppock Curve", f"{coppock_val:.3f}",
                      "Turning Up ✅" if coppock_turning_up else ("Rising 📈" if coppock_rising else "Falling 📉"))
            st.caption(coppock_insight)
        with col_cop2:
            st.metric("Beta IHSG", f"{beta_ihsg:.2f}x",
                      help="Beta > 1 : lebih volatile dari IHSG, Beta < 1 : lebih stabil.")
            st.caption(beta_insight)

        st.markdown("### ⚖️ Bobot Adaptif per Faktor")
        st.caption(
            "Bobot di bawah dihitung otomatis berdasarkan **akurasi historis** masing‑masing faktor. "
            "Faktor yang sering benar mendapat bobot lebih tinggi. Bobot ini digunakan untuk sinyal akhir."
        )
        adaptive_w = get_adaptive_weights(ticker_raw, regime)
        w_df = pd.DataFrame.from_dict(adaptive_w, orient='index', columns=['Weight'])
        st.bar_chart(w_df)

        if adaptive_w:
            max_factor = max(adaptive_w, key=adaptive_w.get)
            min_factor = min(adaptive_w, key=adaptive_w.get)
            max_weight = adaptive_w[max_factor]
            min_weight = adaptive_w[min_factor]

            weight_insight = f"🔍 **Faktor paling dominan:** **{max_factor}** (bobot {max_weight:.1%}). "
            weight_insight += f"**{min_factor}** memiliki bobot terendah ({min_weight:.1%}).\n\n"

            interpretations = {
                "Momentum": "Sinyal momentum (harga 5 hari) paling berpengaruh – pasar sedang *trend-following*. Ikuti tren yang sedang berlangsung.",
                "AI_Senti": "Sentimen berita paling berpengaruh – pergerakan saham banyak dipicu oleh berita/isu terkini. Pantau terus sentimen.",
                "MeanRev": "*Reversal* ke rata-rata (Z-Score) paling berpengaruh – saham cenderung kembali ke level wajar setelah jenuh beli/jual.",
                "Beta_IHSG": "Beta IHSG paling berpengaruh – saham sangat terpengaruh oleh pergerakan pasar secara keseluruhan. Perhatikan arah IHSG.",
                "Coppock": "Coppock Curve paling berpengaruh – sinyal jangka panjang mendominasi, tren utama sedang kuat. Ikuti sinyal makro."
            }
            weight_insight += interpretations.get(max_factor, "")
            st.info(weight_insight)

        st.markdown("### 🧠 Status Memori Adaptif")
        st.caption(
            "**Accuracy** = seberapa sering sinyal faktor sesuai arah harga. **Error EMA** = rata‑rata kesalahan prediksi (makin kecil makin baik)."
        )
        mem = st.session_state.v12_memory.get(ticker_raw, {})
        if mem:
            acc_data = {k: mem.get('accuracy',{}).get(k,0.5) for k in FACTOR_KEYS}
            err_data = {k: mem.get('error_ema',{}).get(k,1.0) for k in FACTOR_KEYS}
            col_a,col_e = st.columns(2)
            with col_a:
                st.caption("✅ Accuracy (higher = better)")
                st.bar_chart(pd.Series(acc_data))
            with col_e:
                st.caption("⚠️ Error EMA (lower = better)")
                st.bar_chart(pd.Series(err_data))

            best_factor = max(acc_data, key=acc_data.get)
            worst_factor = min(acc_data, key=acc_data.get)
            mem_insight = f"🏆 **Faktor paling akurat:** **{best_factor}** (akurasi {acc_data[best_factor]:.1%}). "
            mem_insight += f"Faktor **{worst_factor}** perlu dievaluasi (akurasi {acc_data[worst_factor]:.1%})."
            st.caption(mem_insight)
        else:
            st.info("Belum ada data memori untuk ticker ini. Lakukan analisis beberapa kali agar engine mulai belajar.")

        st.markdown("### 🔁 Proses Self‑Learning")
        st.caption(
            "Setiap analisis, engine membandingkan prediksi sebelumnya dengan harga aktual. "
            "Jika benar → akurasi naik. Jika salah → error bertambah. Bobot otomatis menyesuaikan."
        )
        if os.path.isfile(V12_PRED_FILE):
            pred_df = pd.read_csv(V12_PRED_FILE)
            prev = pred_df[pred_df['ticker']==ticker_raw]
            if not prev.empty:
                last = prev.iloc[-1]
                last_close = last['close_price']
                last_signals = {k: last[f'sig_{k}'] for k in FACTOR_KEYS if f'sig_{k}' in last}
                actual_return = (harga_terakhir - last_close) / last_close if last_close>0 else 0.0
                volatility = returns.std()
                update_v12_memory(ticker_raw, last_signals, actual_return, volatility)
                st.success(f"✅ **Memory updated!** Actual return sejak prediksi terakhir: {actual_return*100:.2f}%")
                pred_df = pred_df[pred_df['ticker']!=ticker_raw]
                pred_df.to_csv(V12_PRED_FILE, index=False)
            else:
                st.info("ℹ️ Tidak ada prediksi sebelumnya. Engine akan mulai belajar pada analisis berikutnya.")
        else:
            st.info("ℹ️ File prediksi belum ada. Engine akan membuatnya setelah analisis pertama.")

        new_pred = {'ticker': ticker_raw, 'close_price': harga_terakhir}
        factor_signals = {
            "Momentum": (df['Mom5D'].iloc[-1] - mom_median_th) / max(0.1, df['Mom5D'].std()),
            "AI_Senti": avg_sentiment,
            "MeanRev": -df['ZScore'].iloc[-1] / 3.0,
            "Beta_IHSG": beta_ihsg * (ihsg_ret.iloc[-1] if 'ihsg_ret' in dir() else 0.0),
            "Coppock": coppock_val / 10.0
        }
        for k,v in factor_signals.items(): new_pred[f'sig_{k}'] = max(-1.0,min(1.0,v))
        pred_df = pd.DataFrame([new_pred])
        if os.path.isfile(V12_PRED_FILE):
            pred_df.to_csv(V12_PRED_FILE, mode='a', header=False, index=False)
        else:
            pred_df.to_csv(V12_PRED_FILE, index=False)
        st.caption("📌 Prediksi hari ini telah disimpan. Lakukan analisis lagi di lain waktu untuk melanjutkan pembelajaran.")

    # ==================== AI INSIGHT OTOMATIS ====================
    st.markdown("---")
    if st.session_state.get("gemini_api_key"):
        with st.spinner("🧠 AI sedang menganalisis hasil dan riwayat..."):
            data_ai = {
                "Saham": ticker_input, "Harga": f"{harga_terakhir:,.0f}", "Sinyal": signal,
                "Rezim": regime, "Sentimen": f"{avg_sentiment:.2f} ({sentimen_status})",
                "RRR": f"{rrr:.2f} (Kontekstual)", "Prob Naik": f"{prob_bull:.1f}%",
                "TP%": f"{tp_pct:.1f}", "SL%": f"{sl_pct:.1f}",
                "Estimasi": f"{est_besok:,.0f}", "Beta": f"{beta_ihsg:.2f}x",
                "WinRate": f"{win_bt:.1%}" if trades_bt else "N/A",
                "ProfitFactor": f"{pf_bt:.2f}" if trades_bt else "N/A",
                "MaxDD": f"{max_dd_bt:.2f}%" if trades_bt else "N/A",
                "Kelly": f"{kelly_adj*100:.1f}",
                "Fundamental_MC": f"{mc:,.0f}" if mc else "N/A",
                "Fundamental_PER": f"{per:.2f}" if per else "N/A",
                "Fundamental_PBV": f"{pbv:.2f}" if pbv else "N/A",
                "Fundamental_ROE": f"{roe*100:.1f}" if roe else "N/A",
                "Fundamental_DE": f"{de:.2f}" if de else "N/A"
            }
            riwayat_konteks = [r for r in st.session_state.riwayat if r['Saham']==ticker_input][:10]
            hasil_ai, error_ai = analisis_saham_dengan_ai(data_ai, riwayat_konteks, st.session_state.gemini_api_key)
            if not error_ai and hasil_ai:
                ringkasan["AI_Insight"] = hasil_ai
                hasil_ai_bersih = bersihkan_teks_ai(hasil_ai)
                html_ai = f'<div class="ai-insight-card"><h3>🤖 Insight AI</h3><p>{hasil_ai_bersih}</p></div>'
                st.markdown(html_ai,unsafe_allow_html=True)
            elif error_ai: st.warning(f"AI tidak dapat memberikan insight: {error_ai}")
    else: st.info("💡 Isi API Key Gemini di sidebar untuk mendapatkan insight AI otomatis.")

    simpan_riwayat(ringkasan)
    st.session_state.riwayat.insert(0, ringkasan)
    if len(st.session_state.riwayat)>50: st.session_state.riwayat.pop()

# --- ANALISIS RIWAYAT DENGAN AI (TOMBOL SIDEBAR) ---
if ai_riwayat_btn:
    if not st.session_state.gemini_api_key: st.error("Masukkan API Key terlebih dahulu!")
    elif not st.session_state.riwayat: st.warning("Belum ada riwayat.")
    else:
        with st.spinner("🧠 AI menganalisis riwayat..."):
            hasil, error = analisis_riwayat_global(st.session_state.riwayat, st.session_state.gemini_api_key)
            if error: st.error(error)
            elif hasil:
                hasil_bersih = bersihkan_teks_ai(hasil)
                st.markdown(f'<div class="ai-insight-card" style="border-left-color:#06b6d4;"><h3 style="color:#67e8f9;">📊 Insight AI dari Riwayat</h3><p>{hasil_bersih}</p></div>', unsafe_allow_html=True)
