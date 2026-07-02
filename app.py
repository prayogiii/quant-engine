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
from datetime import datetime
import pytz
import google.generativeai as genai

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
# FUNGSI AI GEMINI (PROMPT & PEMBERSIHAN DIPERBAIKI)
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
- Take Profit (R1): +{data_saham['TP%']}%
- Stop Loss (S2): -{data_saham['SL%']}%
- Estimasi Harga Besok: Rp {data_saham['Estimasi']}
- Beta terhadap IHSG: {data_saham.get('Beta', 'N/A')}
- Win Rate Backtest: {data_saham.get('WinRate', 'N/A')}
- Profit Factor Backtest: {data_saham.get('ProfitFactor', 'N/A')}
- Max Drawdown Backtest: {data_saham.get('MaxDD', 'N/A')}
- Alokasi Kelly Maks: {data_saham.get('Kelly', 'N/A')}%
- Fundamental: Market Cap: {data_saham.get('Fundamental_MC', 'N/A')}, PER: {data_saham.get('Fundamental_PER', 'N/A')}, PBV: {data_saham.get('Fundamental_PBV', 'N/A')}, ROE: {data_saham.get('Fundamental_ROE', 'N/A')}, D/E: {data_saham.get('Fundamental_DE', 'N/A')}

{riwayat_text}

Tulis analisis ringkas dalam BAHASA INDONESIA, 4 paragraf pendek, langsung ke inti. Jangan gunakan bullet poin atau header.
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

    # Prompt super ketat – instruksi di AWAL agar model tidak mengabaikan
    prompt = (
        "TULIS HANYA ANALISIS NARATIF DALAM BAHASA INDONESIA, 3-4 PARAGRAF. "
        "JANGAN menulis proses berpikir, instruksi, draft, atau bahasa Inggris. "
        "JANGAN gunakan bullet poin atau header. "
        "JANGAN memulai dengan kalimat seperti 'Berikut adalah analisis...' atau 'Karena Anda hanya...'. "
        "Langsung tulis analisis yang mencakup pola sinyal, saham terbaik, perbaikan strategi, dan insight tambahan.\n\n"
        "Data riwayat:\n"
    )
    for r in riwayat_data[:30]:
        prompt += (
            f"{r['Waktu']} | {r['Saham']} | Sinyal: {r['Sinyal']} | "
            f"RRR: {r['RRR']} | Sentimen: {r['Sentimen']} | Rezim: {r['Rezim']}\n"
        )
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), None
    except Exception as e:
        return None, f"Gagal menghasilkan insight: {str(e)}"

def bersihkan_teks_ai(teks):
    if not teks:
        return teks
    # 1. Hapus header markdown dan karakter aneh
    teks = re.sub(r'^#{1,3}\s*', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'\*\*', '', teks)
    teks = re.sub(r'\*', '', teks)
    teks = re.sub(r'\$', '', teks)
    # 2. Hapus kalimat pembuka yang umum (jika di awal teks)
    teks = re.sub(r'^Berikut adalah analisis ringkas.*?\n', '', teks, flags=re.IGNORECASE)
    teks = re.sub(r'^Karena Anda hanya memberikan.*?\n', '', teks, flags=re.IGNORECASE)
    teks = re.sub(r'^Input Data:.*?\n', '', teks, flags=re.IGNORECASE)
    # 3. Proses per baris untuk membersihkan sisa‑sisa
    baris = teks.split('\n')
    baris_bersih = []
    for b in baris:
        b_stripped = b.strip()
        if not b_stripped:
            continue
        # Buang baris yang hanya berisi bullet
        if b_stripped in ['-', '*', '•']:
            continue
        b_lower = b_stripped.lower()
        # Daftar kata kunci yang menandakan instruksi / meta / bahasa Inggris (diperluas)
        if any(kata in b_lower for kata in [
            'input data:', 'requirement:', 'critical observation:',
            'drafting section', 'the rrr problem:', 'the imbalance:',
            'the regime vs. tp:', 'the sentiment:', 'nuance:',
            'constraint:', 'action:', 'stock:', 'wait, what?',
            'a good trader looks for', 'here, the risk is much larger',
            'tp (take profit):', 'sl (stop loss):',
            'data:', 'goal:', 'tickers observed:', 'signals:', 'rrr (risk-reward ratio):',
            'regimes:', 'sentiment:', 'tp vs sl:', 'point 1:', 'point 2:', 'point 3:', 'point 4:',
            'observation:', 'candidate', 'criteria:', 'problem:', 'fix:', 'key pattern:',
            'no bullets?', 'no headers?', 'no ?', 'indonesian language?',
            'check constraints', 'constraint', 'draft', 'final polish',
            'self-correction', 'paragraph', 'input:', 'dates:',
            'patterns:', 'formatting:', 'tone:', 'directly cover:',
            'language:', 'date:', 'strong buy:', 'buy (tactical):', 'hold/wait:',
            'avoid:', 'stocks analysis:', 'sentiments/regimes:', 'note the shift.',
            'look at the', 'focus on', 'no instructions', 'no thinking process',
            'just the final analysis', 'gunakan bahasa', 'jangan gunakan',
            'fokus pada makna', 'kekuatan/kelemahan', 'risiko utama',
            'rekomendasi langkah', 'jika ada pola', 'format:', 'columns:',
            'timestamp', 'total entries:', 'timeframe:', 'berdasarkan data di atas',
            'tulis analisis ringkas', 'bahasa indonesia', 'langsung ke inti',
        ]):
            continue
        # Buang baris yang hanya berisi angka/tanggal
        if re.match(r'^[\d\s\-:,.]+$', b_lower):
            continue
        # Buang baris yang mayoritas huruf kecil (indikasi bahasa Inggris)
        alpha_count = sum(c.isalpha() for c in b_stripped)
        if alpha_count > 0:
            lower_alpha = sum(c.islower() for c in b_stripped if c.isalpha())
            if lower_alpha / alpha_count > 0.7 and alpha_count > 10:
                continue
        # Buang baris yang terlalu pendek dan mengandung alfabet (sisa bullet)
        if len(b_stripped) < 10 and alpha_count > 0:
            continue
        # Hapus bullet di awal baris
        b_stripped = re.sub(r'^[\s]*[-*•]\s*', '', b_stripped)
        baris_bersih.append(b_stripped)

    teks = '\n'.join(baris_bersih)
    # 4. Hapus paragraf duplikat
    paragraf = teks.split('\n')
    paragraf_unik = []
    for p in paragraf:
        if p not in paragraf_unik:
            paragraf_unik.append(p)
    teks = '\n'.join(paragraf_unik)
    teks = teks.replace('\n', '<br>')
    return teks

# ==========================================
# KONFIGURASI HALAMAN & STYLING (TETAP SAMA)
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

    st.subheader("📜 Riwayat Analisis")
    if st.session_state.riwayat:
        for r in st.session_state.riwayat[:10]:
            with st.expander(f"{r['Saham']} - {r['Sinyal']} ({r['Waktu']})"):
                st.markdown(f"**Harga:** Rp {r['Harga']}")
                st.markdown(f"**Estimasi Besok:** Rp {r['Estimasi']}")
                st.markdown(f"**Prob Naik:** {r['Prob Naik']}")
                st.markdown(f"**RRR:** {r['RRR']}")
                st.markdown(f"**Sentimen:** {r['Sentimen']}")
                st.markdown(f"**Rezim:** {r['Rezim']}")
                st.markdown(f"**TP%:** {r['TP%']}% | **SL%:** {r['SL%']}%")
                ai = r.get("AI_Insight", "").strip()
                if ai:
                    ai_bersih = bersihkan_teks_ai(ai)   # ← dibersihkan sebelum ditampilkan
                    st.markdown("💬 **AI Insight:**")
                    st.caption(ai_bersih[:200] + ("..." if len(ai_bersih) > 200 else ""))
        if len(st.session_state.riwayat) > 10:
            st.caption(f"Menampilkan 10 dari {len(st.session_state.riwayat)} riwayat.")
    else:
        st.caption("Belum ada riwayat.")

    st.markdown("---")
    st.subheader("🧠 AI (Gemini)")

    def get_api_key():
        try:
            return st.secrets["GEMINI_API_KEY"]
        except KeyError:
            pass
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            return env_key
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
    if api_key:
        st.session_state.gemini_api_key = api_key

    ai_riwayat_btn = st.button("📊 Analisis Riwayat dgn AI", use_container_width=True)

    if st.button("🗑️ Hapus Semua Riwayat"):
        if os.path.isfile(RIWAYAT_FILE):
            os.remove(RIWAYAT_FILE)
        st.session_state.riwayat = []
        st.success("Riwayat dihapus!")

    st.markdown("---")
    st.caption("Data dari Yahoo Finance. Bukan rekomendasi investasi.")

# ==================== FUNGSI DATA & INDIKATOR ====================
# ... (tidak diubah, sama persis dengan kode sebelumnya)

# ==================== PROSES ANALISIS ====================
if run_btn:
    # ... (semua perhitungan kuantitatif, sama persis)
    # ... (tampilan utama, sama persis)
    # ... (detail expander, sama persis)

    # ==================== AI INSIGHT OTOMATIS ====================
    st.markdown("---")
    if st.session_state.get("gemini_api_key"):
        with st.spinner("🧠 AI sedang menganalisis hasil dan riwayat..."):
            data_ai = {
                "Saham": ticker_input,
                "Harga": f"{harga_terakhir:,.0f}",
                "Sinyal": signal,
                "Rezim": regime,
                "Sentimen": f"{avg_sentiment:.2f} ({sentimen_status})",
                "RRR": f"{rrr:.2f}",
                "Prob Naik": f"{prob_bull:.1f}%",
                "TP%": f"{tp_pct:.1f}",
                "SL%": f"{sl_pct:.1f}",
                "Estimasi": f"{est_besok:,.0f}",
                "Beta": f"{beta_ihsg:.2f}x",
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
            riwayat_konteks = [r for r in st.session_state.riwayat if r['Saham'] == ticker_input][:10]
            hasil_ai, error_ai = analisis_saham_dengan_ai(
                data_ai, riwayat_konteks, st.session_state.gemini_api_key
            )
            if not error_ai and hasil_ai:
                ringkasan["AI_Insight"] = hasil_ai
                hasil_ai_bersih = bersihkan_teks_ai(hasil_ai)
                html_ai = f"""
                <div class="ai-insight-card">
                    <h3>🤖 Insight AI</h3>
                    <p>{hasil_ai_bersih}</p>
                </div>
                """
                st.markdown(html_ai, unsafe_allow_html=True)
            elif error_ai:
                st.warning(f"AI tidak dapat memberikan insight: {error_ai}")
    else:
        st.info("💡 Isi API Key Gemini di sidebar untuk mendapatkan insight AI otomatis.")

    simpan_riwayat(ringkasan)
    st.session_state.riwayat.insert(0, ringkasan)
    if len(st.session_state.riwayat) > 50:
        st.session_state.riwayat.pop()

# --- ANALISIS RIWAYAT DENGAN AI (TOMBOL SIDEBAR) ---
if ai_riwayat_btn:
    if not st.session_state.gemini_api_key:
        st.error("Masukkan API Key terlebih dahulu!")
    elif not st.session_state.riwayat:
        st.warning("Belum ada riwayat.")
    else:
        with st.spinner("🧠 AI menganalisis riwayat..."):
            hasil, error = analisis_riwayat_global(st.session_state.riwayat, st.session_state.gemini_api_key)
            if error:
                st.error(error)
            elif hasil:
                hasil_bersih = bersihkan_teks_ai(hasil)
                html_riwayat_ai = f"""
                <div class="ai-insight-card" style="border-left-color: #06b6d4;">
                    <h3 style="color: #67e8f9;">📊 Insight AI dari Riwayat</h3>
                    <p>{hasil_bersih}</p>
                </div>
                """
                st.markdown(html_riwayat_ai, unsafe_allow_html=True)
