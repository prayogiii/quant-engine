import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, t as student_t
from scipy.optimize import minimize
import warnings
import urllib.parse
import re
import os
import json
from datetime import datetime, timedelta
import pytz
import math
import google.generativeai as genai
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Google Sheets integration
import gspread
from google.oauth2.service_account import Credentials

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
def safe_float(value, default=0.0):
    """Konversi aman ke float, kembalikan default jika gagal."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
# ═══════════════════════════════════════════════════════════════
# V12 ADAPTIVE ENGINE – KONSTANTA & STATE
# ═══════════════════════════════════════════════════════════════
FACTOR_KEYS   = ["Momentum","AI_Senti","MeanRev","Beta_IHSG","Coppock","OFI"]
WEIGHT_MIN    = 0.08
WEIGHT_MAX    = 0.40
SOFTMAX_TEMP  = 2.5
AI_SIGNAL_CAP = 0.30
MC_PESSIMISM  = 0.82

# ====================== FRAKSI HARGA BEI ======================
def fraksi_bei(harga):
    """Membulatkan harga ke kelipatan fraksi sesuai aturan BEI."""
    if harga < 200:
        fraksi = 1
    elif harga < 500:
        fraksi = 2
    elif harga < 2000:
        fraksi = 5
    elif harga < 5000:
        fraksi = 10
    else:
        fraksi = 25
    return round(harga / fraksi) * fraksi

def fraksi_step(harga):
    """Mengembalikan nilai kelipatan 1 fraksi BEI."""
    if harga < 200: return 1
    elif harga < 500: return 2
    elif harga < 2000: return 5
    elif harga < 5000: return 10
    else: return 25

# ====================== GOOGLE SHEETS FUNCTIONS ======================
def get_gsheet():
    """Mengembalikan objek spreadsheet berdasarkan secrets."""
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    return client.open_by_key(st.secrets["google_sheets"]["sheet_id"])

def init_sheets():
    """Membuat sheet 'riwayat', 'v12_memory', dan 'v12_predictions' jika belum ada."""
    try:
        sheet = get_gsheet()
        existing = {ws.title: ws for ws in sheet.worksheets()}   # ⬅️ ubah di sini
        if "riwayat" not in existing:
            sheet.add_worksheet("riwayat", rows=3000, cols=35)
        if "v12_memory" not in existing:
            sheet.add_worksheet("v12_memory", rows=100, cols=3)
        if "v12_predictions" not in existing:
            sheet.add_worksheet("v12_predictions", rows=500, cols=9)
        else:
            ws = existing["v12_predictions"]
            if ws.col_count < 9:
                ws.add_cols(9 - ws.col_count)
        if "riwayat_actual" not in existing:
            sheet.add_worksheet("riwayat_actual", rows=100, cols=7)
    except Exception as e:
        st.error(f"❌ Gagal inisialisasi Google Sheets: {e}")

#V12 Memory (Google Sheets)
def load_v12_memory():
    mem = {}
    try:
        sheet = get_gsheet().worksheet("v12_memory")
        records = sheet.get_all_records()
        for row in records:
            t = row.get('ticker')
            if t and 'data' in row and row['data']:
                try:
                    mem[t] = json.loads(row['data'])
                except:
                    pass
    except Exception as e:
        st.error(f"Gagal memuat V12 memory: {e}")
    return mem

def save_v12_memory(mem):
    try:
        sheet = get_gsheet().worksheet("v12_memory")
        rows = [{'ticker': t, 'data': json.dumps(d)} for t, d in mem.items()]
        sheet.clear()
        if rows:
            all_values = [['ticker', 'data']] + [[r['ticker'], r['data']] for r in rows]
            sheet.update(all_values, value_input_option='RAW')
    except Exception as e:
        st.error(f"Gagal menyimpan V12 memory: {e}")

def load_v12_predictions(ticker, mode="swing"):
    try:
        sheet = get_gsheet().worksheet("v12_predictions")
        records = sheet.get_all_records()
        for row in records:
            if row.get('ticker') == ticker and row.get('mode') == mode:
                return row
        return None
    except Exception as e:
        st.error(f"Gagal memuat prediksi: {e}")
        return None

def save_v12_prediction(ticker, close_price, factor_signals, entry_low=None, entry_high=None, mode="swing"):
    try:
        sheet = get_gsheet()
        ws = sheet.worksheet("v12_predictions")
        new_row = {
            'ticker': ticker,
            'mode': mode,               # <-- tambah ini
            'close_price': close_price,
            'timestamp': datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S"),
            'entry_low': entry_low,
            'entry_high': entry_high
        }
        for k in FACTOR_KEYS:
            new_row[f'sig_{k}'] = factor_signals.get(k, 0.0)

        headers = list(new_row.keys())   # sekarang 10 kolom (termasuk mode)
        # ---------- Pastikan jumlah kolom cukup ----------
        if ws.col_count < len(headers):
            ws.add_cols(len(headers) - ws.col_count)

        # Tulis ulang header
        last_col = chr(64 + len(headers))
        ws.update(f'A1:{last_col}1', [headers], value_input_option='RAW')

        records = ws.get_all_records()
        row_index = None
        for i, row in enumerate(records):
            if row.get('ticker') == ticker and row.get('mode') == mode:
                row_index = i + 2
                break

        if row_index:
            values = [new_row[h] for h in headers]
            last_col = chr(64 + len(headers))
            ws.update(f'A{row_index}:{last_col}{row_index}', [values], value_input_option='RAW')
        else:
            values = [new_row[h] for h in headers]
            ws.append_row(values, value_input_option='RAW')
    except Exception as e:
        st.error(f"Gagal menyimpan prediksi: {e}")

def default_weight(factor, regime):
    defaults = {
        "STABLE BULLISH": {"Momentum":0.25,"AI_Senti":0.18,"MeanRev":0.12,"Beta_IHSG":0.15,"Coppock":0.30,"OFI": 0.12},
        "VOLATILE UPTREND": {"Momentum":0.28,"AI_Senti":0.14,"MeanRev":0.12,"Beta_IHSG":0.16,"Coppock":0.30,"OFI": 0.10},
        "HIGH-STRESS PANIC": {"Momentum":0.15,"AI_Senti":0.18,"MeanRev":0.22,"Beta_IHSG":0.15,"Coppock":0.30,"OFI": 0.18},
        "SIDEWAYS / CONSOLIDATION": {"Momentum":0.15,"AI_Senti":0.18,"MeanRev":0.27,"Beta_IHSG":0.12,"Coppock":0.28,"OFI": 0.14},
        "BEARISH ACCUMULATION": {"Momentum":0.20,"AI_Senti":0.18,"MeanRev":0.20,"Beta_IHSG":0.15,"Coppock":0.27,"OFI": 0.13}
    }
    return defaults.get(regime, {"Momentum":0.23,"AI_Senti":0.17,"MeanRev":0.15,"Beta_IHSG":0.15,"Coppock":0.30}).get(factor,0.15)

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
def hitung_bars_remaining(now_jkt, actual_interval, bars_per_day_map):
    h, m = now_jkt.hour, now_jkt.minute
    interval_minutes_map = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}
    interval_menit = interval_minutes_map.get(actual_interval, 5)

    if h < 12 or (h == 12 and m == 0):
        menit_sesi1_tersisa = 12*60 - (h*60 + m)
        sisa_menit = menit_sesi1_tersisa + 90
        bars = math.ceil(sisa_menit / interval_menit)
    elif h == 12 or (h == 13 and m < 30):
        bars = math.ceil(90 / interval_menit)
    elif (h == 13 and m >= 30) or h == 14 or (h == 15 and m == 0):
        sisa_menit = 15*60 - (h*60 + m)
        bars = math.ceil(sisa_menit / interval_menit)
    else:
        bars = bars_per_day_map.get(actual_interval, 54)
    return max(1, bars)

# ---------- Adaptive Weights ----------
def get_adaptive_weights(ticker, regime, v12_mem=None):
    if v12_mem is not None:
        mem = v12_mem.get(ticker, {})
    else:
        try:
            mem = st.session_state.v12_memory.get(ticker, {})
        except (AttributeError, Exception):
            mem = {}
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
def bersihkan_untuk_json(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, pd.Timestamp): return obj.isoformat()
    elif isinstance(obj, (np.bool_,)): return bool(obj)
    return obj

def simpan_riwayat(ringkasan):
    try:
        sheet = get_gsheet().worksheet("riwayat")
        items_to_add = ringkasan if isinstance(ringkasan, list) else [ringkasan]
        records = sheet.get_all_records()
        valid_records = [r for r in records if any(str(v).strip() for v in r.values())]
        data = list(valid_records)
        for item in reversed(items_to_add):
            ringkasan_bersih = {k: bersihkan_untuk_json(v) for k, v in item.items()}
            data.insert(0, ringkasan_bersih)
        data = data[:3000]
        if data:
            headers = list(data[0].keys())
            rows = [[row.get(h, "") for h in headers] for row in data]
            sheet.clear()
            sheet.update([headers] + rows, value_input_option='RAW')
        st.session_state.riwayat = data
    except Exception as e:
        st.error(f"❌ Gagal menyimpan riwayat: {e}")

def muat_riwayat_dari_sheets():
    try:
        sheet = get_gsheet().worksheet("riwayat")
        records = sheet.get_all_records()
        valid_records = [r for r in records if any(str(v).strip() for v in r.values())]
        return valid_records[:3000]
    except Exception as e:
        st.error(f"❌ Gagal memuat riwayat: {e}")
        return []

def muat_riwayat_actual():
    data = {}
    # Normalisasi nilai mode dari berbagai format ke label pendek (SW/DT)
    def norm_gaya(val):
        v = str(val).strip().lower()
        if v in ('sw', 'swing'): return 'SW'
        if v in ('dt', 'daytrade', 'day_trade', 'day trade'): return 'DT'
        return val  # kembalikan apa adanya jika tidak dikenali

    try:
        sheet = get_gsheet().worksheet("riwayat_actual")
        records = sheet.get_all_records()
        for row in records:
            waktu = str(row.get('Waktu', ''))
            saham = str(row.get('Saham', ''))
            raw_gaya = row.get('Mode', '') or row.get('Gaya', '')
            gaya = norm_gaya(raw_gaya) if raw_gaya else ''

            # Isi nilai aktual
            val = {
                'Actual_High': str(row.get('Actual_High', '') or '').strip(),
                'Actual_Low': str(row.get('Actual_Low', '') or '').strip(),
                'Actual_Close': str(row.get('Actual_Close', '') or '').strip(),
                'Outcome': str(row.get('Outcome', '') or '').strip(),
                'Entry_Miss': str(row.get('Entry_Miss', '') or '').strip(),
                'Mode': gaya if gaya else ''
            }

            if waktu and saham:
                if gaya:
                    data[(waktu, saham, gaya)] = val
                    mode_long = "swing" if gaya == "SW" else ("daytrade" if gaya == "DT" else gaya)
                    data[(waktu, saham, mode_long)] = val
                elif raw_gaya:
                    data[(waktu, saham, str(raw_gaya))] = val
                else:
                    # Legacy data tanpa spesifikasi mode
                    data[(waktu, saham)] = val
    except Exception as e:
        st.error(f"Gagal memuat actual: {e}")
    return data

def hapus_riwayat_item(waktu, saham, gaya=None):
    try:
        sheet = get_gsheet().worksheet("riwayat")
        records = sheet.get_all_records()
        valid_records = [r for r in records if any(str(v).strip() for v in r.values())]
        waktu_str = str(waktu).strip()
        saham_str = str(saham).strip()
        gaya_str = str(gaya).strip() if gaya else None

        if gaya_str:
            filtered = [
                r for r in valid_records
                if not (str(r.get('Waktu', '')).strip() == waktu_str and
                        str(r.get('Saham', '')).strip() == saham_str and
                        str(r.get('Gaya', '')).strip() == gaya_str)
            ]
        else:
            filtered = [
                r for r in valid_records
                if not (str(r.get('Waktu', '')).strip() == waktu_str and
                        str(r.get('Saham', '')).strip() == saham_str)
            ]
        filtered = filtered[:3000]
        sheet.clear()
        if filtered:
            headers = list(filtered[0].keys())
            rows = [[row.get(h, "") for h in headers] for row in filtered]
            sheet.update([headers] + rows, value_input_option='RAW')
        st.session_state.riwayat = filtered
    except Exception as e:
        st.error(f"❌ Gagal menghapus riwayat: {e}")
        
def simpan_riwayat_actual(waktu, saham, actual_data, mode="swing"):
    def norm_gaya(val):
        v = str(val).strip().lower()
        if v in ('sw', 'swing'): return 'SW'
        if v in ('dt', 'daytrade', 'day_trade', 'day trade'): return 'DT'
        return val

    try:
        sheet = get_gsheet().worksheet("riwayat_actual")
        records = sheet.get_all_records()
        headers = ['Waktu', 'Saham', 'Mode', 'Actual_High', 'Actual_Low', 'Actual_Close', 'Outcome', 'Entry_Miss']

        # Jika belum ada data sama sekali, tulis header dulu
        if not records:
            sheet.insert_row(headers, 1)
        else:
            existing_headers = list(records[0].keys())
            if 'Mode' not in existing_headers:
                # Update baris 1 saja — JANGAN insert_row agar tidak duplikat header
                sheet.update('A1:H1', [headers], value_input_option='RAW')
                # Reload records karena header berubah
                records = sheet.get_all_records()

        row_index = None
        target_mode_norm = norm_gaya(mode)
        for i, row in enumerate(records):
            r_mode = row.get('Mode') or row.get('Gaya') or ''
            if str(row.get('Waktu')) == str(waktu) and str(row.get('Saham')) == str(saham) and norm_gaya(r_mode) == target_mode_norm:
                row_index = i + 2
                break
        new_row = [waktu, saham, mode,
                   actual_data.get('Actual_High', ''),
                   actual_data.get('Actual_Low', ''),
                   actual_data.get('Actual_Close', ''),
                   actual_data.get('Outcome', ''),
                   actual_data.get('Entry_Miss', '')]
        if row_index:
            sheet.update(f'A{row_index}:H{row_index}', [new_row], value_input_option='RAW')
        else:
            sheet.append_row(new_row, value_input_option='RAW')
        st.session_state.riwayat_actual = muat_riwayat_actual()
        integrate_actual_to_v12(waktu, saham, actual_data, mode=mode)
    except Exception as e:
        st.error(f"Gagal menyimpan actual: {e}")

def hitung_hari_bursa(start_date, end_date):
    """Menghitung jumlah hari bursa (Senin-Jumat) antara 2 tanggal"""
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    if start_date >= end_date:
        return 0
    cur = start_date + timedelta(days=1)
    b_days = 0
    while cur <= end_date:
        if cur.weekday() < 5:
            b_days += 1
        cur += timedelta(days=1)
    return b_days

def fetch_actual_data_yfinance(saham, waktu_str):
    """
    Mengambil data High, Low, Close historis dari yfinance sejak tanggal sinyal s/d hari ini.
    """
    try:
        ticker_input = saham if saham.endswith(".JK") else f"{saham}.JK"
        dt_part = waktu_str.split()[0]
        dt_sinyal = datetime.strptime(dt_part, "%Y-%m-%d").date()
        start_str = (dt_sinyal - timedelta(days=1)).strftime("%Y-%m-%d")
        df_hist = yf.download(ticker_input, start=start_str, progress=False)
        if df_hist is None or df_hist.empty:
            return None
        if isinstance(df_hist.columns, pd.MultiIndex):
            try:
                df_hist = df_hist.xs(ticker_input, axis=1, level=1)
            except:
                df_hist.columns = [c[0] for c in df_hist.columns]

        df_filtered = df_hist[df_hist.index.date >= dt_sinyal]
        if df_filtered.empty:
            df_filtered = df_hist

        max_hi = float(df_filtered['High'].max())
        min_lo = float(df_filtered['Low'].min())
        last_cl = float(df_filtered['Close'].iloc[-1])

        return {
            'Actual_High': f"{max_hi:,.0f}".replace(",", ""),
            'Actual_Low': f"{min_lo:,.0f}".replace(",", ""),
            'Actual_Close': f"{last_cl:,.0f}".replace(",", "")
        }
    except Exception as e:
        return None

def dapatkan_sinyal_perlu_dicatat(riwayat_data, riwayat_actual):
    urgent_items = []
    active_swing_items = []
    now_jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
    today_date = now_jkt.date()

    for r in riwayat_data:
        waktu_str = r.get('Waktu', '')
        saham = r.get('Saham', '')
        gaya = r.get('Gaya', 'SW')
        mode_actual = "swing" if gaya == "SW" else "daytrade"

        actual_data = (
            riwayat_actual.get((waktu_str, saham, gaya)) or
            riwayat_actual.get((waktu_str, saham, mode_actual)) or
            riwayat_actual.get((waktu_str, saham))
        )

        has_actual = False
        if actual_data:
            if (actual_data.get('Actual_High') or 
                actual_data.get('Actual_Low') or 
                actual_data.get('Actual_Close') or 
                actual_data.get('Outcome') or 
                actual_data.get('Entry_Miss') == 'Yes'):
                has_actual = True

        if has_actual:
            continue

        try:
            dt_sinyal = datetime.strptime(waktu_str.split()[0], "%Y-%m-%d").date()
        except:
            dt_sinyal = today_date

        b_days = hitung_hari_bursa(dt_sinyal, today_date)

        item = {
            'record': r,
            'waktu': waktu_str,
            'saham': saham,
            'gaya': gaya,
            'mode_actual': mode_actual,
            'b_days': b_days,
            'dt_sinyal': dt_sinyal
        }

        if gaya == "DT" or mode_actual == "daytrade":
            if dt_sinyal < today_date:
                item['alasan'] = f"Daytrade Sesi Sebelumnya ({waktu_str})"
                urgent_items.append(item)
        else:
            if b_days >= 7:
                item['alasan'] = f"Mencapai Batas Maksimal 7 Hari Bursa ({b_days} hari kerja)"
                urgent_items.append(item)
            elif b_days >= 1:
                item['alasan'] = f"Swing Berjalan (Hari bursa ke-{b_days})"
                active_swing_items.append(item)

    return urgent_items, active_swing_items

def render_notifikasi_evaluasi_riwayat():
    riwayat_data = st.session_state.get('riwayat', [])
    riwayat_actual = st.session_state.get('riwayat_actual', {})

    if not riwayat_data:
        return

    urgent_items, active_swing_items = dapatkan_sinyal_perlu_dicatat(riwayat_data, riwayat_actual)

    if not urgent_items and not active_swing_items:
        return

    n_urgent = len(urgent_items)
    n_active = len(active_swing_items)

    st.markdown("""
        <style>
        .notif-box {
            background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
            border-left: 5px solid #a855f7;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 18px;
        }
        </style>
    """, unsafe_allow_html=True)

    title_text = "🔔 Pengingat Evaluasi Outcome Trading"
    details = []
    if n_urgent > 0:
        details.append(f"⚠️ {n_urgent} sinyal perlu dicatat (Daytrade atau Swing ≥7 hari bursa)")
    if n_active > 0:
        details.append(f"⏳ {n_active} Swing aktif (1-6 hari bursa)")

    st.markdown(f"""
    <div class="notif-box">
        <div style="font-size:15px; font-weight:bold; color:#f472b6;">
            {title_text}
        </div>
        <div style="font-size:13px; color:#e2e8f0; margin-top:4px;">
            {' | '.join(details)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📝 Form Evaluasi Sinyal (Quick Outcome Journal)", expanded=(n_urgent > 0)):
        tab_urgent, tab_active = st.tabs([
            f"🚨 Perlu Catat Immediate ({n_urgent})",
            f"⏳ Swing Aktif ({n_active})"
        ])

        with tab_urgent:
            if not urgent_items:
                st.success("🎉 Semua sinyal jatuh tempo sudah dicatat!")
            else:
                for idx, item in enumerate(urgent_items):
                    r = item['record']
                    waktu_key = item['waktu']
                    saham_key = item['saham']
                    gaya_key = item['gaya']
                    mode_actual = item['mode_actual']
                    alasan = item['alasan']

                    st.markdown(f"**📌 {saham_key} ({gaya_key}) - {waktu_key}** | `{alasan}`")
                    st.caption(f"Sinyal: {r.get('Sinyal','?')} | Entry: {r.get('Entry_Zone','?')} | TP: {r.get('TP_Range','?')} | SL: Rp {r.get('SL_Harga','?')}")

                    fetch_key = f"fetch_urg_{idx}_{waktu_key}_{saham_key}_{gaya_key}"
                    form_key = f"form_urg_{idx}_{waktu_key}_{saham_key}_{gaya_key}"

                    col_auto, _ = st.columns([2, 1])
                    with col_auto:
                        if st.button(f"⚡ Fetch Otomatis Data Harga ({saham_key})", key=fetch_key):
                            fetched = fetch_actual_data_yfinance(saham_key, waktu_key)
                            if fetched:
                                st.session_state[f"hi_{fetch_key}"] = fetched['Actual_High']
                                st.session_state[f"lo_{fetch_key}"] = fetched['Actual_Low']
                                st.session_state[f"cl_{fetch_key}"] = fetched['Actual_Close']
                                st.success(f"Data harga {saham_key} berhasil ditarik!")
                            else:
                                st.error(f"Gagal mengambil data {saham_key} dari yfinance")

                    with st.form(key=form_key):
                        def_hi = st.session_state.get(f"hi_{fetch_key}", "")
                        def_lo = st.session_state.get(f"lo_{fetch_key}", "")
                        def_cl = st.session_state.get(f"cl_{fetch_key}", "")

                        c1, c2, c3 = st.columns(3)
                        actual_high = c1.text_input("Actual High", value=def_hi, placeholder="contoh: 5350")
                        actual_low = c2.text_input("Actual Low", value=def_lo, placeholder="contoh: 5050")
                        actual_close = c3.text_input("Actual Close", value=def_cl, placeholder="contoh: 5200")

                        c4, c5 = st.columns(2)
                        entry_miss = c4.checkbox("🚫 Entry Tidak Tersentuh", value=False)
                        if entry_miss:
                            outcome = "Not Touched"
                        else:
                            outcome = c5.selectbox("Outcome", ["", "Win", "Loss", "Not Touched"], format_func=lambda x: "Pilih Outcome" if x=="" else x)

                        submitted = st.form_submit_button("💾 Simpan Outcome")
                        if submitted:
                            if not entry_miss and outcome == "":
                                st.error("Pilih Outcome terlebih dahulu.")
                            else:
                                data = {
                                    'Actual_High': actual_high.strip(),
                                    'Actual_Low': actual_low.strip(),
                                    'Actual_Close': actual_close.strip(),
                                    'Outcome': outcome,
                                    'Entry_Miss': 'Yes' if entry_miss else 'No',
                                    'Mode': mode_actual
                                }
                                simpan_riwayat_actual(waktu_key, saham_key, data, mode=mode_actual)
                                st.success(f"✅ Outcome {saham_key} berhasil disimpan!")
                                st.rerun()
                    st.divider()

        with tab_active:
            if not active_swing_items:
                st.info("Tidak ada posisi Swing aktif (1-6 hari bursa) yang sedang berjalan.")
            else:
                for idx, item in enumerate(active_swing_items):
                    r = item['record']
                    waktu_key = item['waktu']
                    saham_key = item['saham']
                    gaya_key = item['gaya']
                    mode_actual = item['mode_actual']
                    alasan = item['alasan']

                    st.markdown(f"**⏳ {saham_key} ({gaya_key}) - {waktu_key}** | `{alasan}`")
                    st.caption(f"Sinyal: {r.get('Sinyal','?')} | Entry: {r.get('Entry_Zone','?')} | TP: {r.get('TP_Range','?')} | SL: Rp {r.get('SL_Harga','?')}")

                    fetch_key = f"fetch_act_{idx}_{waktu_key}_{saham_key}_{gaya_key}"
                    form_key = f"form_act_{idx}_{waktu_key}_{saham_key}_{gaya_key}"

                    col_auto, _ = st.columns([2, 1])
                    with col_auto:
                        if st.button(f"⚡ Fetch Otomatis Data Harga ({saham_key})", key=fetch_key):
                            fetched = fetch_actual_data_yfinance(saham_key, waktu_key)
                            if fetched:
                                st.session_state[f"hi_{fetch_key}"] = fetched['Actual_High']
                                st.session_state[f"lo_{fetch_key}"] = fetched['Actual_Low']
                                st.session_state[f"cl_{fetch_key}"] = fetched['Actual_Close']
                                st.success(f"Data harga {saham_key} berhasil ditarik!")
                            else:
                                st.error(f"Gagal mengambil data {saham_key} dari yfinance")

                    with st.form(key=form_key):
                        def_hi = st.session_state.get(f"hi_{fetch_key}", "")
                        def_lo = st.session_state.get(f"lo_{fetch_key}", "")
                        def_cl = st.session_state.get(f"cl_{fetch_key}", "")

                        c1, c2, c3 = st.columns(3)
                        actual_high = c1.text_input("Actual High", value=def_hi, placeholder="contoh: 5350")
                        actual_low = c2.text_input("Actual Low", value=def_lo, placeholder="contoh: 5050")
                        actual_close = c3.text_input("Actual Close", value=def_cl, placeholder="contoh: 5200")

                        c4, c5 = st.columns(2)
                        entry_miss = c4.checkbox("🚫 Entry Tidak Tersentuh", value=False)
                        if entry_miss:
                            outcome = "Not Touched"
                        else:
                            outcome = c5.selectbox("Outcome", ["", "Win", "Loss", "Not Touched"], format_func=lambda x: "Pilih Outcome" if x=="" else x)

                        submitted = st.form_submit_button("💾 Simpan Outcome (Early Exit)")
                        if submitted:
                            if not entry_miss and outcome == "":
                                st.error("Pilih Outcome terlebih dahulu.")
                            else:
                                data = {
                                    'Actual_High': actual_high.strip(),
                                    'Actual_Low': actual_low.strip(),
                                    'Actual_Close': actual_close.strip(),
                                    'Outcome': outcome,
                                    'Entry_Miss': 'Yes' if entry_miss else 'No',
                                    'Mode': mode_actual
                                }
                                simpan_riwayat_actual(waktu_key, saham_key, data, mode=mode_actual)
                                st.success(f"✅ Outcome {saham_key} berhasil disimpan!")
                                st.rerun()
                    st.divider()

def integrate_actual_to_v12(waktu, saham, actual_data, mode="swing"):
    try:
        ticker = saham
        last_pred = load_v12_predictions(ticker, mode=mode)
        if not last_pred:
            return

        factor_signals = {}
        for k in FACTOR_KEYS:
            key = f'sig_{k}'
            if key in last_pred:
                factor_signals[k] = float(last_pred[key])
            else:
                factor_signals[k] = 0.0

        # --- 1) Update arah prediksi berdasarkan Actual Close ---
        actual_close_str = actual_data.get('Actual_Close', '')
        if actual_close_str:
            try:
                actual_close = float(str(actual_close_str).replace(",", ""))
                last_close = safe_float(last_pred.get('close_price'), 0.0)
                if last_close > 0:
                    actual_return = (actual_close - last_close) / last_close
                    actual_return = max(-1.0, min(1.0, actual_return))
                    update_v12_memory(ticker, factor_signals, actual_return, volatility=0.02)
            except:
                pass  # gagal parse → arah tidak diupdate

        # --- 2) Belajar dari Entry Miss / Not Touched ---
        # Hanya berjalan jika prediksi sebelumnya menyimpan entry_low & entry_high
        entry_low = last_pred.get('entry_low')
        entry_high = last_pred.get('entry_high')
        if entry_low is not None and entry_high is not None:
            try:
                entry_low_f = safe_float(entry_low, None)
                entry_high_f = safe_float(entry_high, None)
            except:
                entry_low_f = None
                entry_high_f = None

            if entry_low_f is not None and entry_high_f is not None and entry_low_f < entry_high_f:
                gap = None

                # --- Path A: User mengisi Actual Low → hitung gap dari data nyata ---
                actual_low_str = actual_data.get('Actual_Low', '')
                if actual_low_str:
                    try:
                        actual_low_f = float(str(actual_low_str).replace(",", ""))
                        # Jika actual low > entry_high, harga tidak pernah menyentuh zona entry
                        if actual_low_f > entry_high_f:
                            gap = actual_low_f - entry_high_f
                    except:
                        pass

                # --- Path B: User centang "Entry Tidak Tersentuh" (Entry_Miss=Yes)
                #     tanpa mengisi Actual Low → estimasi gap dari selisih close price
                #     prediksi terakhir vs entry_high (fallback konservatif) ---
                if gap is None and actual_data.get('Entry_Miss', '') == 'Yes':
                    last_close = safe_float(last_pred.get('close_price'), 0.0)
                    if last_close > entry_high_f:
                        # Harga penutupan sudah di atas entry_high → gap = selisihnya
                        gap = last_close - entry_high_f
                    else:
                        # Tidak bisa estimasi gap dengan pasti, gunakan nilai kecil
                        # agar engine tahu ada miss tapi tidak over-koreksi
                        gap = entry_high_f * 0.01  # 1% dari entry_high sebagai proxy

                if gap is not None and gap > 0:
                    mem = st.session_state.v12_memory.get(ticker, {})
                    if 'entry_error_ema' not in mem:
                        mem['entry_error_ema'] = 0.0

                    alpha = 0.2
                    mem['entry_error_ema'] = (
                        mem['entry_error_ema'] * (1 - alpha) + gap * alpha
                    )

                    st.session_state.v12_memory[ticker] = mem
                    save_v12_memory(st.session_state.v12_memory)
    except Exception as e:
        st.error(f"Gagal integrasi V12: {e}")
# ====================== API IDX ======================
@st.cache_data(ttl=86400)   # cache 1 hari
def fetch_idx_stock_list_exclude_monitoring():
    excluded_boards = {
        "pemantauankhusus", "pemantauan_khusus", "pemantauankhusus",
        "monitoring", "special_monitoring", "specialmonitoring"
    }
    endpoints = [
        "https://www.idx.co.id/umbraco/Surface/ListedCompany/GetStockList?start=0&length=9999",
        "https://www.idx.co.id/umbraco/Surface/ListedCompany/GetStockList?language=id-id&start=0&length=9999",
        "https://www.idx.co.id/umbraco/Surface/ListedCompany/GetStockList?start=0&length=9999&exchangeBoard=&industry=&subIndustry=&search="
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.idx.co.id/",
        "X-Requested-With": "XMLHttpRequest"
    }

    all_codes = []
    seen = set()
    for url in endpoints:
        try:
            resp = requests.get(url, headers=headers, timeout=25)
            if resp.status_code != 200:
                continue
            raw = resp.json()

            # Cari key yang berisi list saham
            items = None
            if isinstance(raw, list):
                items = raw
            else:
                for key in ('data', 'Data', 'result', 'Result', 'results'):
                    if key in raw:
                        items = raw[key]
                        break
                if items is None:
                    # Coba ekstrak rekursif
                    def extract(obj):
                        out = []
                        if isinstance(obj, dict):
                            if 'code' in obj or 'Code' in obj or 'KodeSaham' in obj:
                                out.append(obj)
                            for v in obj.values():
                                out.extend(extract(v))
                        elif isinstance(obj, list):
                            for v in obj:
                                out.extend(extract(v))
                        return out
                    items = extract(raw)

            if not items:
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                code = item.get('Code') or item.get('code') or item.get('KodeSaham')
                if not code:
                    continue
                code = str(code).strip().upper()
                if len(code) > 6 or not code.isalnum():
                    continue

                board = item.get('BoardId') or item.get('Board') or item.get('boardId') or ""
                board_lower = str(board).lower().replace(" ", "").replace("-", "").replace("_", "")
                if board_lower in excluded_boards:
                    continue

                if code not in seen:
                    seen.add(code)
                    all_codes.append(code)

            if len(all_codes) >= 100:
                break
        except Exception as e:
            # st.write(f"Error endpoint {url}: {e}")   # debug
            continue

    return sorted(all_codes) if all_codes else None
def fetch_all_idx_stocks():
    """Ambil semua saham BEI non-Pemantauan Khusus."""
    return fetch_idx_stock_list_exclude_monitoring()
# ==========================================
# FUNGSI AI GEMINI
# ==========================================
def dapatkan_model_gemini(api_key):
    if not api_key: return None, "API key belum diisi."
    try:
        genai.configure(api_key=api_key)
        available = [m.name.split('/')[-1] for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not available: return None, "Tidak ada model Gemini."
        for model_id in available:
            try:
                model = genai.GenerativeModel(model_id)
                model.generate_content("test", generation_config={"max_output_tokens": 1})
                return model, None
            except Exception: continue
        return None, "Model gagal digunakan."
    except Exception as e:
        return None, f"Error: {str(e)}"
def analisis_saham_dengan_ai(data_saham, riwayat, api_key):
    model, error = dapatkan_model_gemini(api_key)
    if error: return None, error
    riwayat_text = ""
    if riwayat:
        riwayat_text = "Riwayat analisis sebelumnya (termasuk hasil aktual jika tersedia):\n"
        for r in riwayat:  # sudah difilter per emiten oleh pemanggil
            base = f"- {r['Waktu']} | {r['Saham']} | Sinyal: {r['Sinyal']} | RRR: {r['RRR']} | Rezim: {r['Rezim']}"
            # Tambahkan data aktual
            if r.get('Actual_High') or r.get('Actual_Outcome'):
                base += " | Hasil Aktual: "
                if r.get('Actual_High'):
                    base += f"High={r['Actual_High']}, "
                if r.get('Actual_Low'):
                    base += f"Low={r['Actual_Low']}, "
                if r.get('Actual_Close'):
                    base += f"Close={r['Actual_Close']}, "
                if r.get('Actual_Outcome'):
                    base += f"Outcome={r['Actual_Outcome']}"
                if r.get('Entry_Miss'):
                    base += " (Entry tidak tersentuh)"
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
- Probabilitas Naik: {data_saham['Prob Naik']}
- Take Profit: +{data_saham['TP%']}%
- Stop Loss: -{data_saham['SL%']}%
- Estimasi: Rp {data_saham['Estimasi']}
- Beta terhadap IHSG: {data_saham.get('Beta', 'N/A')}
- Win Rate Backtest: {data_saham.get('WinRate', 'N/A')}
- Profit Factor Backtest: {data_saham.get('ProfitFactor', 'N/A')}
- Max Drawdown Backtest: {data_saham.get('MaxDD', 'N/A')}
- Alokasi Kelly Maks: {data_saham.get('Kelly', 'N/A')}%
- Fundamental: Market Cap: {data_saham.get('Fundamental_MC', 'N/A')}, PER: {data_saham.get('Fundamental_PER', 'N/A')}, PBV: {data_saham.get('Fundamental_PBV', 'N/A')}, ROE: {data_saham.get('Fundamental_ROE', 'N/A')}, D/E: {data_saham.get('Fundamental_DE', 'N/A')}
- Status Posisi: {data_saham.get('Status_Posisi', 'Tidak diketahui')}
- Harga Beli: {data_saham.get('Harga_Beli', 'Tidak diisi')}
- Floating P/L: {data_saham.get('Floating_PL', 'N/A')}
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

def analisis_riwayat_global(riwayat_data, riwayat_actual, api_key):
    model, error = dapatkan_model_gemini(api_key)
    if error: return None, error
    if not riwayat_data: return None, "Belum ada riwayat."
    prompt = "Berikut adalah riwayat analisis saham yang telah dilakukan (termasuk hasil aktual jika tersedia):\n\n"
    for r in riwayat_data[:30]:
        gaya_label = "📆 SW" if r.get('Gaya') == "SW" else "⏱️ DT"
        base = f"- {r['Waktu']}|{gaya_label}|{r['Saham']}|Sinyal:{r['Sinyal']}|Harga:{r['Harga']}|RRR:{r['RRR']}|Sentimen:{r['Sentimen']}|Rezim:{r['Rezim']}|TP%:{r['TP%']}%|SL%:{r['SL%']}%"

        # Tambahkan data aktual jika tersedia
        gaya = r.get('Gaya', 'SW')
        # Coba key 3 elemen (dengan gaya), fallback ke key 2 elemen (data lama)
        key_actual = (r.get('Waktu'), r.get('Saham'), gaya)
        actual = riwayat_actual.get(key_actual) or riwayat_actual.get((r.get('Waktu'), r.get('Saham')), {})
        if actual:
            base += " | Hasil Aktual: "
            details = []
            if actual.get('Actual_High'):
                details.append(f"High={actual['Actual_High']}")
            if actual.get('Actual_Low'):
                details.append(f"Low={actual['Actual_Low']}")
            if actual.get('Actual_Close'):
                details.append(f"Close={actual['Actual_Close']}")
            if actual.get('Outcome'):
                details.append(f"Outcome={actual['Outcome']}")
            if actual.get('Entry_Miss') == 'Yes':
                details.append("Entry Tidak Tersentuh")
            base += ", ".join(details)

        prompt += base + "\n"

    prompt += (
        "\nBerdasarkan data di atas, berikan analisis ringkas (Bahasa Indonesia):\n"
        "- Pola sinyal yang sering muncul\n"
        "- Saham dengan peluang terbaik menurut data (termasuk hasil aktualnya)\n"
        "- Rekomendasi perbaikan strategi\n"
        "- Insight tambahan yang berguna untuk trader\n"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), None
    except Exception as e:
        return None, f"Gagal menghasilkan insight: {str(e)}"

def bersihkan_teks_ai(teks):
    if not teks: return teks
    teks = re.sub(r'^#{1,3}\s*', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'\*\*', '', teks)
    teks = re.sub(r'\*', '', teks)
    teks = teks.replace('\n', '<br>')
    return teks

# ==========================================
# KONFIGURASI HALAMAN & STYLING
# ==========================================
st.set_page_config(page_title="Quant Risk Engine Pro v2", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

if "sheets_initialized" not in st.session_state:
    init_sheets()
    st.session_state.sheets_initialized = True

if 'v12_memory' not in st.session_state:
    st.session_state.v12_memory = load_v12_memory()

if "riwayat" not in st.session_state:
    st.session_state.riwayat = muat_riwayat_dari_sheets()
if "riwayat_actual" not in st.session_state:
    st.session_state.riwayat_actual = muat_riwayat_actual()

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
        border-radius: 16px; padding: 20px; margin: 15px 0;
        border-left: 5px solid #8b5cf6; color: #cbd5e1; font-size: 15px; line-height: 1.6;
    }
    .ai-insight-card h3 { color: #a78bfa; margin-top: 0; font-size: 20px; }
    .ai-insight-card p { margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 📊 QuantRisk Pro")
    
    st.caption("⚙️ Sistem akan menganalisis **Swing (harian)** dan **Daytrade (intraday)** secara otomatis.")
    
    st.markdown("Masukkan kode saham IHSG untuk analisis lengkap.")
    ticker_raw = st.text_input("🔍 Kode Saham", value="BBRI", placeholder="Contoh: BBRI, TLKM, BMRI").upper().strip()
    if ticker_raw and not ticker_raw.endswith(".JK"):
        ticker_input = f"{ticker_raw}.JK"
    else:
        ticker_input = ticker_raw

    # --- Input Harga Manual ---
    harga_manual = st.text_input("💵 Harga Pasar Saat Ini (opsional)", placeholder="Kosongkan jika pakai harga data")
    if harga_manual:
        try:
            harga_terakhir_manual = float(harga_manual.replace(",",""))
        except:
            st.error("Format harga salah")
            harga_terakhir_manual = None
    else:
        harga_terakhir_manual = None
    # Letakkan sebelum tombol ANALISIS, misal setelah harga_manual
    sudah_beli = st.checkbox("🟢 Saya sudah punya posisi di saham ini", value=False)
    # ---- Tambahan input harga beli ----
    harga_beli_float = None
    if sudah_beli:
        harga_beli_str = st.text_input("💰 Harga Beli Rata‑rata (opsional)", placeholder="Kosongkan jika tidak tahu")
        if harga_beli_str:
            try:
                harga_beli_float = float(harga_beli_str.replace(",", ""))
            except:
                st.error("Format harga beli salah")

    # ---- Pengaturan Fee Broker ----
    with st.expander("⚙️ Fee Broker (Beli & Jual)", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fee_beli_pct = st.number_input("Fee Beli (%)", min_value=0.0, max_value=2.0, value=0.15, step=0.05, key="fee_beli_pct")
        with col_f2:
            fee_jual_pct = st.number_input("Fee Jual (%)", min_value=0.0, max_value=2.0, value=0.25, step=0.05, key="fee_jual_pct")

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("🚀 ANALISIS", use_container_width=True)
    with col2:
        if st.button("🗑️ Reset Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache dibersihkan!")
    #==================== SCANNER SAHAM IDX ====================
    st.markdown("---")
    st.subheader("🔍 Scanner Saham IDX")
    mode_scan = st.selectbox(
        "Pilih Mode Scan:",
        ["Cepat (LQ45)", "Papan Utama", "Komprehensif (Utama + Pengembangan)", "Full IDX", "Auto-Fetch (API BEI)"],
        index=0, key="mode_scan"
    )
    likuiditas_min = st.number_input(
        "Filter Likuiditas Minimum (Rp/hari, rata2 20 hari)",
        min_value=0, value=300_000_000, step=100_000_000, key="likuiditas_min"
    )
    ai_rerank = st.checkbox("Sertakan Scanner AI Re-Rank (Top 15 kandidat teknikal)", value=False, key="ai_rerank")
    if ai_rerank:
        st.caption("+15-30 detik. Hemat kuota Gemini gratis: HANYA 1 panggilan API dibatch utk semua kandidat + cache harian per-saham.")
    
    scan_btn = st.button("🔍 SCAN SAHAM", use_container_width=True)

    # ---------- HELPER RENDER CARD PER MODE (SIDE-BY-SIDE) ----------
    def render_mode_card(r, mode_title, mode_icon, container, idx_key):
        with container:
            st.markdown(f"**{mode_icon} {mode_title}**")
            if not r:
                st.caption("*(Data tidak tersedia)*")
                return

            sig_icon = "🔥" if "STRONG BUY" in r.get('Sinyal','') else ("⚡" if "BUY" in r.get('Sinyal','') else ("⏸️" if "HOLD" in r.get('Sinyal','') else "🚨"))
            st.markdown(f"{sig_icon} **{r.get('Sinyal','?')}**")
            st.caption(f"Score: **{r.get('Score','?')}** | RRR: **{r.get('RRR','?')}**")

            harga_beli_r = r.get('Harga_Beli', '')
            if harga_beli_r:
                st.caption(f"💰 Beli: Rp {harga_beli_r} | Float: {r.get('Floating_PL', '')}")

            entry_zone = r.get('Entry_Zone', '')
            tp_range   = r.get('TP_Range', '')
            sl_harga   = r.get('SL_Harga', '')
            if entry_zone or tp_range or sl_harga:
                info_teknis = []
                if entry_zone: info_teknis.append(f"🎯 Entry: {entry_zone}")
                if tp_range: info_teknis.append(f"TP: {tp_range}")
                if sl_harga: info_teknis.append(f"SL: Rp {sl_harga}")
                st.caption(" | ".join(info_teknis))

            waktu_key = r.get('Waktu','')
            saham_key = r.get('Saham','')
            gaya_key = r.get('Gaya', 'SW')
            mode_actual = "swing" if gaya_key == "SW" else "daytrade"
            actual_data = (
                st.session_state.riwayat_actual.get((waktu_key, saham_key, gaya_key)) or
                st.session_state.riwayat_actual.get((waktu_key, saham_key, mode_actual)) or
                st.session_state.riwayat_actual.get((waktu_key, saham_key))
            )

            btn_key = f"btn_{idx_key}_{waktu_key}_{saham_key}_{gaya_key}"
            del_key = f"del_{idx_key}_{waktu_key}_{saham_key}_{gaya_key}"
            show_key = f"show_{idx_key}_{waktu_key}_{saham_key}_{gaya_key}"
            form_key = f"form_{idx_key}_{waktu_key}_{saham_key}_{gaya_key}"

            has_actual = False
            if actual_data:
                if (actual_data.get('Actual_High') or 
                    actual_data.get('Actual_Low') or 
                    actual_data.get('Actual_Close') or 
                    actual_data.get('Outcome') or 
                    actual_data.get('Entry_Miss') == 'Yes'):
                    has_actual = True

            if has_actual:
                st.caption(f"📌 High: {actual_data.get('Actual_High','')} | Low: {actual_data.get('Actual_Low','')}")
                if actual_data.get('Entry_Miss') == 'Yes':
                    st.caption("⚠️ Entry Tidak Tersentuh")
                if actual_data.get('Outcome'):
                    warna = {'Win':'🟢','Loss':'🔴','Not Touched':'⚪'}.get(actual_data['Outcome'],'')
                    st.caption(f"🏁 Outcome: {warna} {actual_data['Outcome']}")
            else:
                col_b1, col_b2 = st.columns([1, 1])
                with col_b1:
                    if st.button("📝 Catat", key=btn_key):
                        st.session_state[show_key] = True
                with col_b2:
                    if st.button("🗑️ Hapus", key=del_key):
                        hapus_riwayat_item(waktu_key, saham_key, gaya=gaya_key)
                        st.rerun()

                if st.session_state.get(show_key, False):
                    with st.form(key=form_key):
                        actual_high = st.text_input("Actual High", placeholder="6250")
                        actual_low = st.text_input("Actual Low", placeholder="6100")
                        actual_close = st.text_input("Actual Close", placeholder="6200")
                        entry_miss = st.checkbox("🚫 Entry Tidak Tersentuh", value=False)
                        if entry_miss:
                            outcome = "Not Touched"
                            st.info("Outcome otomatis Not Touched")
                        else:
                            outcome = st.selectbox("Outcome", ["", "Win", "Loss", "Not Touched"], format_func=lambda x: "Pilih Outcome" if x == "" else x)
                        submitted = st.form_submit_button("Simpan")
                        if submitted:
                            if not entry_miss and outcome == "":
                                st.error("Pilih Outcome terlebih dahulu.")
                            else:
                                data = {
                                    'Actual_High': actual_high.strip(),
                                    'Actual_Low': actual_low.strip(),
                                    'Actual_Close': actual_close.strip(),
                                    'Outcome': outcome,
                                    'Entry_Miss': 'Yes' if entry_miss else ''
                                }
                                simpan_riwayat_actual(waktu_key, saham_key, data, mode=mode_actual)
                                st.success("Data actual tersimpan!")
                                st.session_state[show_key] = False
                                st.rerun()

    # ---------- RIWAYAT ANALISIS (dengan Search & Paginasi) ----------
    st.subheader("📜 Riwayat Analisis")
    
    if "riwayat_page" not in st.session_state:
        st.session_state.riwayat_page = 0
    if "prev_search" not in st.session_state:
        st.session_state.prev_search = ""
    
    render_notifikasi_evaluasi_riwayat()
    
    search_query = st.text_input("🔎 Cari Saham", key="search_riwayat", placeholder="Ketik kode saham...")
    
    if search_query != st.session_state.prev_search:
        st.session_state.riwayat_page = 0
        st.session_state.prev_search = search_query
    
    riwayat_data = st.session_state.riwayat if st.session_state.riwayat else []
    if search_query:
        riwayat_data = [r for r in riwayat_data if search_query.lower() in r.get('Saham', '').lower()]
    
    # ---------- Toggle tampilan per hari ----------
    group_by_day = st.checkbox("📅 Kelompokkan per Hari", value=True)
    
    if group_by_day:
        # Kelompokkan berdasarkan tanggal (10 karakter pertama dari Waktu)
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in riwayat_data:
            tgl = r.get('Waktu', '')[:10]  # ambil YYYY-MM-DD
            if tgl:
                grouped[tgl].append(r)
        sorted_days = sorted(grouped.keys(), reverse=True)
        items_per_page = 5  # jumlah hari per halaman
        total_items = len(sorted_days)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
        # Pagination untuk hari
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀ Sebelumnya", disabled=(st.session_state.riwayat_page == 0), key="prev_day"):
                    st.session_state.riwayat_page = max(0, st.session_state.riwayat_page - 1)
            with col2:
                st.markdown(f"<div style='text-align:center; color:#8892b0;'>Hal. {st.session_state.riwayat_page+1} / {total_pages}</div>", unsafe_allow_html=True)
            with col3:
                if st.button("Selanjutnya ▶", disabled=(st.session_state.riwayat_page >= total_pages - 1), key="next_day"):
                    st.session_state.riwayat_page = min(total_pages - 1, st.session_state.riwayat_page + 1)
    
        start_idx = st.session_state.riwayat_page * items_per_page
        end_idx = start_idx + items_per_page
        display_days = sorted_days[start_idx:end_idx]
    
        if display_days:
            for day in display_days:
                entries = grouped[day]
                # Kelompokkan entries per (Waktu, Saham)
                session_map = defaultdict(dict)
                for r in entries:
                    s_key = (r.get('Waktu', ''), r.get('Saham', ''))
                    gaya = r.get('Gaya', 'SW')
                    session_map[s_key][gaya] = r

                expander_title = f"📅 {day} – {len(session_map)} sesi analisis"
                with st.expander(expander_title):
                    for s_idx, ((waktu, saham), modes) in enumerate(session_map.items()):
                        r_sw = modes.get('SW')
                        r_dt = modes.get('DT')
                        harga_val = r_sw.get('Harga') if r_sw else (r_dt.get('Harga') if r_dt else '?')
                        st.markdown(f"🔥 **{saham}** @ Rp {harga_val} <span style='font-size:11px;color:#8892b0;'>(🕒 {waktu})</span>", unsafe_allow_html=True)
                        col_sw, col_dt = st.columns(2)
                        render_mode_card(r_sw, "Swing", "📆", col_sw, f"g_{day}_{s_idx}")
                        render_mode_card(r_dt, "Daytrade", "⏱️", col_dt, f"g_{day}_{s_idx}")
                        st.divider()
            st.caption(f"📋 Menampilkan {start_idx+1}-{min(end_idx, total_items)} dari {total_items} hari" +
                      (f" (hasil pencarian '{search_query}')" if search_query else ""))
        else:
            if search_query:
                st.caption(f"❌ Tidak ada riwayat yang cocok dengan '{search_query}'.")
            else:
                st.caption("Belum ada riwayat.")
    
    else:
        # ---------- Tampilan flat (penuh & lengkap per item) ----------
        items_per_page = 10
        total_items = len(riwayat_data)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀ Sebelumnya", disabled=(st.session_state.riwayat_page == 0), key="prev_flat"):
                    st.session_state.riwayat_page = max(0, st.session_state.riwayat_page - 1)
            with col2:
                st.markdown(f"<div style='text-align:center; color:#8892b0;'>Hal. {st.session_state.riwayat_page+1} / {total_pages}</div>", unsafe_allow_html=True)
            with col3:
                if st.button("Selanjutnya ▶", disabled=(st.session_state.riwayat_page >= total_pages - 1), key="next_flat"):
                    st.session_state.riwayat_page = min(total_pages - 1, st.session_state.riwayat_page + 1)
    
        start_idx = st.session_state.riwayat_page * items_per_page
        end_idx = start_idx + items_per_page
        display_riwayat = riwayat_data[start_idx:end_idx]
    
        if display_riwayat:
            for idx, r in enumerate(display_riwayat):
                sig_icon = "🔥" if "STRONG BUY" in r.get('Sinyal','') else ("⚡" if "BUY" in r.get('Sinyal','') else ("⏸️" if "HOLD" in r.get('Sinyal','') else "🚨"))
                conf_str = r.get('Confidence', '0%')
                try: conf_val = float(conf_str.replace('%',''))
                except: conf_val = 0
                conf_text = "Tinggi ▲" if conf_val >= 70 else ("Sedang ►" if conf_val >= 50 else "Rendah ▼")
                gaya = r.get('Gaya','?')
                gaya_label = "⏱️DT" if gaya == "DT" else ("📆SW" if gaya == "SW" else "")
                expander_title = f"{r.get('Saham','?')} @ Rp {r.get('Harga','?')} {sig_icon} {r.get('Sinyal','?')} ({gaya_label}) Score: {r.get('Score','?')}"
                with st.expander(expander_title):
                    st.markdown(f"**{sig_icon} {r.get('Sinyal','?')}** ({gaya_label})")
                    st.caption(f"Score: {r.get('Score','?')} | Confidence: {r.get('Confidence','?')} ({conf_text}) | Risk-Adj: {r.get('RRR','?')}")
                    waktu_analisis = r.get('Waktu', '?')
                    if waktu_analisis and waktu_analisis != '?':
                        st.caption(f"🕒 Waktu Analisis: {waktu_analisis}")
                    harga_beli_r = r.get('Harga_Beli', '')
                    if harga_beli_r:
                        st.caption(f"💰 Harga Beli: Rp {harga_beli_r} | Floating: {r.get('Floating_PL', '')}")
                    st.divider()
                
                    # Coppock full width
                    st.metric("Coppock", r.get('Coppock','?'))
                
                    # Estimasi (caption)
                    est_netral = r.get('Estimasi_Netral', '?')
                    est_sinyal = r.get('Estimasi_Sinyal', '?')
                    ret_netral = r.get('Est_Return', '?')
                    ret_sinyal = r.get('Est_Return_Sinyal', '?')
                    st.caption(f"📊 Netral: {est_netral} ({ret_netral})  |  🎯 Sinyal: {est_sinyal} ({ret_sinyal})")
                
                    # TP & SL dalam dua kolom
                    c1, c2 = st.columns(2)
                    tplabel = "Est. TP Sesi Berikutnya" if r.get('Gaya') == 'DT' else "Est. TP Besok"
                    sllabel = "Est. SL Sesi Berikutnya" if r.get('Gaya') == 'DT' else "Est. SL Besok"
                    tp_val = r.get('TP_Harga') or r.get('TP_Range', '?')
                    tp_display = str(tp_val) if str(tp_val).startswith('Rp') else f"Rp {tp_val}"
                    sl_val = r.get('SL_Harga', '?')
                    sl_display = str(sl_val) if str(sl_val).startswith('Rp') else f"Rp {sl_val}"
                    with c1:
                        st.markdown(f"""<div style="margin-top: 0px;"><label data-testid="stMetricLabel" style="color:rgb(255, 255, 255); font-size:14px; margin:0 0 4px 0; display:block;">{tplabel}</label><div data-testid="stMetricValue" style="color:rgb(0, 255, 204); font-size:24px; font-weight:700; line-height:1.2;">{tp_display}</div></div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div style="margin-top: 0px;"><label data-testid="stMetricLabel" style="color:rgb(255, 255, 255); font-size:14px; margin:0 0 4px 0; display:block;">{sllabel}</label><div data-testid="stMetricValue" style="color:rgb(239, 68, 68); font-size:24px; font-weight:700; line-height:1.2;">{sl_display}</div></div>""", unsafe_allow_html=True)
                
                    # Likuiditas
                    st.metric("Likuiditas", r.get('Likuiditas','?'), delta="/hari")
                
                    # Entry Zone (bila ada)
                    entry_zone_val = r.get('Entry_Zone', '?')
                    if entry_zone_val and entry_zone_val != '?':
                        st.markdown(f"""<div style="margin-top: 8px;">
                            <label data-testid="stMetricLabel" style="color:rgb(255, 255, 255); font-size:14px; margin:0 0 4px 0; display:block;">🎯 Entry Zone</label>
                            <div data-testid="stMetricValue" style="color:rgb(0, 255, 204); font-size:24px; font-weight:700; line-height:1.2;">{entry_zone_val}</div>
                        </div>""", unsafe_allow_html=True)
                
                    # Indikator tambahan
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
                    status_pos = r.get('Status_Posisi', '')
                    if status_pos == 'Sudah Beli':
                        st.caption("🟢 Saat analisis: **Sudah memiliki posisi**")
    
                    # ---- Fitur Catat Actual ----
                    waktu_key = r.get('Waktu','')
                    saham_key = r.get('Saham','')
                    gaya_key = r.get('Gaya', 'SW')
                    mode_actual = "swing" if gaya_key == "SW" else "daytrade"
                    actual_data = (
                        st.session_state.riwayat_actual.get((waktu_key, saham_key, gaya_key)) or
                        st.session_state.riwayat_actual.get((waktu_key, saham_key, mode_actual)) or
                        st.session_state.riwayat_actual.get((waktu_key, saham_key))
                    )

                    has_actual = False
                    if actual_data:
                        if (actual_data.get('Actual_High') or 
                            actual_data.get('Actual_Low') or 
                            actual_data.get('Actual_Close') or 
                            actual_data.get('Outcome') or 
                            actual_data.get('Entry_Miss') == 'Yes'):
                            has_actual = True

                    if has_actual:
                        st.caption(f"📌 Actual High: {actual_data.get('Actual_High','')} | Low: {actual_data.get('Actual_Low','')}")
                        if actual_data.get('Entry_Miss') == 'Yes':
                            st.caption("⚠️ Entry Tidak Tersentuh")
                        if actual_data.get('Outcome'):
                            warna_outcome = {
                                'Win': '🟢',
                                'Loss': '🔴',
                                'Not Touched': '⚪'
                            }.get(actual_data['Outcome'], '')
                            st.caption(f"🏁 Outcome: {warna_outcome} {actual_data['Outcome']}")
                    else:
                        btn_key = f"btn_actual_{idx}_{waktu_key}_{saham_key}_{gaya_key}"
                        form_key = f"form_actual_{idx}_{waktu_key}_{saham_key}_{gaya_key}"
                        show_key = f"show_form_{idx}_{waktu_key}_{saham_key}_{gaya_key}"
                        hapus_key = f"hapus_{idx}_{waktu_key}_{saham_key}_{gaya_key}"

                        col_btn1, col_btn2 = st.columns([1, 1])
                        with col_btn1:
                            if st.button("📝 Catat Hasil", key=btn_key):
                                st.session_state[show_key] = True
                        with col_btn2:
                            if st.button("🗑️ Hapus", key=hapus_key):
                                hapus_riwayat_item(waktu_key, saham_key, gaya=gaya_key)
                                st.rerun()

                        if st.session_state.get(show_key, False):
                            with st.form(key=form_key):
                                actual_high = st.text_input("Actual High", placeholder="contoh: 6250")
                                actual_low = st.text_input("Actual Low (opsional)", placeholder="contoh: 6100")
                                actual_close = st.text_input("Actual Close (opsional)", placeholder="contoh: 6200")

                                entry_miss = st.checkbox(
                                    "🚫 Entry Tidak Tersentuh",
                                    value=False,
                                    help="Centang jika harga tidak pernah menyentuh zona entry (meskipun TP/ SL tersentuh)."
                                )

                                if entry_miss:
                                    outcome = "Not Touched"
                                    st.info("ℹ️ Entry tidak tersentuh → outcome otomatis **Not Touched**.")
                                else:
                                    outcome = st.selectbox(
                                        "Outcome",
                                        options=["", "Win", "Loss", "Not Touched"],
                                        format_func=lambda x: "Pilih Outcome" if x == "" else x
                                    )

                                submitted = st.form_submit_button("Simpan", key=f"submit_{form_key}")
                                if submitted:
                                    if not entry_miss and outcome == "":
                                        st.error("Pilih Outcome terlebih dahulu.")
                                    else:
                                        data = {
                                            'Actual_High': actual_high.strip(),
                                            'Actual_Low': actual_low.strip(),
                                            'Actual_Close': actual_close.strip(),
                                            'Outcome': outcome,
                                            'Entry_Miss': 'Yes' if entry_miss else ''
                                        }
                                        simpan_riwayat_actual(waktu_key, saham_key, data, mode=mode_actual)
                                        st.success("Data actual tersimpan!")
                                        st.session_state[show_key] = False
                                        st.rerun()
                    if ai:
                        st.caption(f"💡 {ai[:150]}")
    
            st.caption(f"📋 Menampilkan {start_idx+1}-{min(end_idx, total_items)} dari {total_items} riwayat" +
                      (f" (hasil pencarian '{search_query}')" if search_query else ""))
        else:
            if search_query:
                st.caption(f"❌ Tidak ada riwayat mecocok dengan '{search_query}'.")
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
    api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.gemini_api_key, placeholder="AIza...", help="Kunci API Gemini. Disimpan di secrets atau env.")
    if api_key: st.session_state.gemini_api_key = api_key
    ai_riwayat_btn = st.button("📊 Analisis Riwayat dgn AI", use_container_width=True)
    if st.button("🗑️ Hapus Semua Riwayat"):
        try:
            sheet = get_gsheet().worksheet("riwayat")
            sheet.clear()
            st.session_state.riwayat = []
            st.success("Riwayat dihapus!")
        except Exception as e:
            st.error(f"Gagal menghapus riwayat: {e}")

    # ---------- KALENDER BURSA ----------
    st.markdown("---")
    now_jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
    today_str = now_jkt.strftime("%Y-%m-%d")
    today_day = now_jkt.strftime("%A")
    current_hour, current_minute = now_jkt.hour, now_jkt.minute
    current_year = now_jkt.strftime("%Y")
    st.subheader(f"📅 Kalender Bursa {current_year}")
    libur_bursa = {
        "2025-01-01": "Tahun Baru Masehi", "2025-01-29": "Tahun Baru Imlek", "2025-03-14": "Hari Suci Nyepi",
        "2025-04-18": "Wafat Yesus Kristus", "2025-05-01": "Hari Buruh", "2025-05-29": "Kenaikan Yesus Kristus",
        "2025-05-30": "Hari Raya Waisak", "2025-06-06": "Idul Adha", "2025-06-27": "Tahun Baru Islam",
        "2025-08-17": "Hari Kemerdekaan", "2025-09-05": "Maulid Nabi", "2025-12-25": "Hari Raya Natal",
        "2026-01-01": "Tahun Baru Masehi", "2026-02-17": "Tahun Baru Imlek", "2026-03-03": "Hari Suci Nyepi",
        "2026-04-03": "Wafat Yesus Kristus", "2026-05-01": "Hari Buruh", "2026-05-14": "Kenaikan Yesus Kristus",
        "2026-05-15": "Hari Raya Waisak", "2026-05-25": "Idul Adha", "2026-06-15": "Tahun Baru Islam",
        "2026-08-17": "Hari Kemerdekaan", "2026-08-24": "Maulid Nabi", "2026-12-25": "Hari Raya Natal",
    }
    def dalam_jam_perdagangan(hour, minute):
        sesi1 = (hour == 9 and minute >= 0) or (10 <= hour < 12) or (hour == 12 and minute == 0)
        sesi2 = (hour == 13 and minute >= 30) or (hour == 14) or (hour == 15 and minute == 0)
        return sesi1 or sesi2
    if today_str in libur_bursa: st.warning(f"Hari ini bursa **TUTUP**: {libur_bursa[today_str]}")
    elif today_day in ["Saturday", "Sunday"]: st.warning("Hari ini **AKHIR PEKAN**, bursa tutup.")
    elif dalam_jam_perdagangan(current_hour, current_minute): st.success("Bursa **TERBUKA** (Sesi 1: 09:00-12:00, Sesi 2: 13:30-15:00 WIB)")
    else: st.info("Bursa **TUTUP** (di luar jam perdagangan).")
    st.caption("Libur dalam 2 minggu ke depan:")
    future_libur = []
    for date_str, desc in libur_bursa.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        delta = (dt.date() - now_jkt.date()).days
        if 0 < delta <= 14: future_libur.append(f"- {dt.strftime('%d %b')}: {desc}")
    if future_libur:
        for item in future_libur: st.caption(item)
    else: st.caption("Tidak ada libur dalam 2 minggu.")
    st.markdown("---")
    st.caption("Data dari Yahoo Finance. Bukan rekomendasi investasi.")
    # ==================== FUNGSI DATA & INDIKATOR ====================
@st.cache_data(ttl=60)   # sudah Anda ubah
def load_stock_data(ticker, period="2y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, prepost=True, actions=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=60)
def load_ihsg_data(period="2y", interval="1d"):
    df = yf.download("^JKSE", period=period, interval=interval, prepost=True, actions=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df
@st.cache_data(ttl=3600)
def get_daftar_saham(mode):
    """Mengembalikan list kode saham (tanpa .JK) berdasarkan mode scan."""
    # Daftar statis fallback (contoh, kamu bisa lengkapi sendiri)
    lq45 = ["AADI", "ADMR", "ADRO", "AKRA", "AMMN", "AMRT", "ANTM", "ASII", "BBCA", "BBNI",
            "BBRI", "BBTN", "BMRI", "BRPT", "BUMI", "CPIN", "CUAN", "DEWA", "EMTK", "ESSA",
            "EXCL", "GOTO", "HRTA", "ICBP", "INCO", "INDF", "INDY", "INKP", "ISAT", "ITMG",
            "JPFA", "KLBF", "MAPI", "MBMA", "MDKA", "MEDC", "NCKL", "PGAS", "PGEO", "PTBA",
            "SCMA", "TLKM", "UNTR", "UNVR", "WIFI"]
    
    papan_utama = lq45 + ["AALI", "ABMM", "ACES", "ADHI", "AISA", "ALDO", "AMAG", "APLN", "ARNA", "ARTO",
                          "ASGR", "ASRI", "ASSA", "AUTO", "BACA", "BALI", "BAYU", "BBHI", "BBMD", "BBYB",
                          "BCAP", "BDMN", "BEST", "BFIN", "BGTG", "BINA", "BIRD", "BISI", "BJBR", "BJTM",
                          "BKSL", "BMTR", "BNGA", "BNII", "BNLI", "BRMS", "BSDE", "BSIM", "BSSR", "BTPN",
                          "BUDI", "BVIC", "BWPT", "BYAN", "CASS", "CFIN", "CITA", "CMNP", "CTRA", "DILD",
                          "DKFT", "DLTA", "DMAS", "DNET", "DSNG", "DSSA", "ELSA", "ENRG", "EPMT", "ERAA",
                          "FISH", "GEMS", "GGRM", "GJTL", "GZCO", "HERO", "HEXA", "HMSP", "HRUM", "IMAS",
                          "IMPC", "INPC", "INTP", "ISSP", "JIHD", "JKON", "JRPT", "JSMR", "JSPT", "JTPE",
                          "KBLI", "KIJA", "KKGI", "KPIG", "LPCK", "LPKR", "LPPF", "LSIP", "LTLS", "MAIN",
                          "MAYA", "MBSS", "MCOR", "MEGA", "MERK", "MIDI", "MIKA", "MLBI", "MLPL", "MNCN",
                          "MPMX", "MTDL", "MTLA", "MYOR", "NISP", "NOBU", "PADI", "PALM", "PANS", "PNBN",
                          "PNIN", "PNLF", "PTPP", "PTRO", "PWON", "RAJA", "RALS", "SAME", "SGRO", "SIDO",
                          "SILO", "SIMP", "SMAR", "SMBR", "SMDR", "SMGR", "SMRA", "SMSM", "SRTG", "SSIA",
                          "SSMS", "TBIG", "TBLA", "TINS", "TKIM", "TMAS", "TOBA", "TOTL", "TOTO", "TOWR",
                          "TPIA", "TPMA", "TRIM", "TSPC", "ULTJ", "UNIC", "VICO", "WIIM", "WINS", "WTON",
                          "SHIP", "POWR", "PRDA", "BRIS", "PORT", "CARS", "CLEO", "WOOD", "MARK", "PSSI",
                          "MORA", "PBID", "IPCM", "BTPS", "SPTO", "HEAL", "TUGU", "MSIN", "MAPA", "IPCC",
                          "FILM", "PANI", "GOOD", "SKRN", "BOLA", "KOTA", "HDIT", "KEEN", "TEBE", "KEJU",
                          "PSGO", "UCID", "GLVA", "AMAR", "DMND", "SAMF", "SGER", "BBSI", "VICI", "TAPG",
                          "MASB", "BMHS", "MCOL", "MTEL", "CMRY", "STAA", "TLDN", "MTMH", "TRGU", "HATM",
                          "JARR", "ELPI", "PRAY", "CBUT", "MKTR", "OMED", "SUNI", "BDKR", "SMIL", "MAHA",
                          "ERAL", "BREN", "MSTI", "ALII", "GOLF", "DAAZ", "MDIY", "DGWG", "CBDK", "BLOG",
                          "YUPI", "MDLA", "RAAM", "JECX", "BACH", "RMKE", "AVIA", "DRMA", "AGRO"]
                        
    
    pengembangan = papan_utama + ["ABDA", "AKPI", "AKSI", "AMFG", "AMIN", "ANJT", "APEX", "APIC", "APII", "APLI",
                                  "ARGO", "ARII", "ARTA", "ASBI", "ASDM", "ASJT", "ASRM", "ATIC", "BABP", "BAJA",
                                  "BAPA", "BBKP", "BBLD", "BBRM", "BCIC", "BCIP", "BIPI", "BIPP", "BKDP", "BKSW",
                                  "BMAS", "BMSR", "BNBA", "BNBR", "BOLT", "BPFI", "BPII", "BRAM", "BRNA", "BTON",
                                  "BUKK", "BULL", "BUVA", "CEKA", "CENT", "CINT", "CLPI", "CPRO", "CSAP", "CTBN",
                                  "CTTH", "DART", "DEFI", "DGIK", "DNAR", "DOID", "DPNS", "DSFI", "DVLA", "DYAN",
                                  "ECII", "EKAD", "EMDE", "ERTX", "ESTI", "FAST", "FMII", "FORU", "FPNI", "GDST",
                                  "GDYR", "GEMA", "GIAA", "GMTD", "GOLD", "GPRA", "GSMF", "GTBO", "GWSA", "HDFA",
                                  "IATA", "ICON", "IGAR", "IKBI", "IMJS", "INAI", "INCI", "INDR", "INDS", "INDX",
                                  "INPP", "INTD", "IPOL", "ITMA", "JAWA", "JECC", "KAEF", "KBLM", "KBLV", "KDSI",
                                  "KICI", "KOBX", "KONI", "KOPI", "KRAS", "LAPD", "LEAD", "LINK", "LION", "LMPI",
                                  "LPGI", "LPIN", "LPLI", "LPPS", "LRNA", "MBAP", "MBTO", "MDIA", "MDLN", "META",
                                  "MGNA", "MICE", "MITI", "MKPI", "MLPT", "MMLP", "MRAT", "MREI", "MSKY", "MYOH",
                                  "NELY", "NIKL", "NIRO", "NRCA", "OKAS", "OMRE", "PANR", "PDES", "PEGE", "PGLI",
                                  "PICO", "PJAA", "PKPK", "PNBS", "PSAB", "PSDN", "PSKT", "PTIS", "PTSN", "PTSP",
                                  "PUDP", "PYFA", "RANC", "RBMS", "RDTX", "RELI", "RICY", "RIGS", "RODA", "ROTI",
                                  "RUIS", "SAFE", "SCCO", "SDMU", "SDPC", "SDRA", "SHID", "SIPD", "SKBM", "SKLT",
                                  "SMDM", "SMMA", "SMMT", "SOCI", "SPMA", "SQMI", "SRAJ", "SRSN", "SSTM", "STAR",
                                  "STTP", "SULI", "TALF", "TBMS", "TCID", "TGKA", "TIFA", "TIRA", "TMPO", "TRIS",
                                  "TRST", "TRUS", "UNIT", "VINS", "VOKS", "VRNA", "WAPO", "WEHA", "WOMF", "YPAS",
                                  "YULE", "CASA", "DAYA", "DPUM", "IDPR", "JGLE", "KINO", "OASA", "PBSA", "BOGA",
                                  "MINA", "CSIS", "FIRE", "KMTR", "HOKI", "MPOW", "MDKI", "BELL", "KIOS", "GMFI",
                                  "MTWI", "MCAS", "PPRE", "WEGE", "DWGL", "JMAS", "CAMP", "LCKM", "HELI", "GHON",
                                  "DFAM", "NICK", "PRIM", "TRUK", "PZZA", "TNCA", "TCPI", "RISE", "BPTR", "NFCX",
                                  "MGRO", "LAND", "MOLI", "CITY", "SAPX", "SURE", "MPRO", "YELO", "CAKK", "SATU",
                                  "POLA", "DIVA", "LUCK", "SOTS", "ZONE", "PEHA", "BEEF", "POLI", "CLAY", "NATO",
                                  "JAYA", "COCO", "JAST", "FITT", "CCSI", "SFAN", "POLU", "KJEN", "ITIC", "PAMG",
                                  "BLUE", "EAST", "LIFE", "FUJI", "INOV", "SMKL", "TFAS", "GGRP", "OPMS", "NZIA",
                                  "SLIS", "IRRA", "DMMX", "WOWS", "ESIP", "REAL", "IFII", "PMJS", "CSRA", "INDO",
                                  "AMOR", "TRIN", "PTPW", "TAMA", "IKAN", "RONY", "CSMI", "BBSS", "BHAT", "EPAC",
                                  "UANG", "PGUN", "TRJA", "SCNP", "KMDS", "PURI", "SOHO", "HOMI", "ROCK", "ENZO",
                                  "ATAP", "BANK", "WMUU", "EDGE", "UNIQ", "SNLK", "ZYRX", "NPGF", "ADCP", "HOPE",
                                  "TRUE", "LABA", "ARCI", "NICL", "UVCR", "HAIS", "OILS", "GPSO", "RSGK", "SBMA",
                                  "CMNT", "GTSI", "KUAS", "BOBA", "DEPO", "BINO", "TAYS", "SEMA", "ASLC", "NETV",
                                  "ENAK", "NTBK", "BIKE", "WIRG", "SICO", "GOTO", "ASHA", "SWID", "ARKO", "CHEM",
                                  "DEWI", "AXIO", "KRYA", "GULA", "TOOL", "BUAH", "CRAB", "MEDS", "COAL", "BELI",
                                  "BSBK", "PDPP", "KDTN", "ZATA", "MMIX", "PADA", "VTNY", "ELIT", "BEER", "CBPE",
                                  "CBRE", "WINE", "PEVE", "LAJU", "FWCT", "IRSX", "VAST", "HALO", "FUTR", "PTMP",
                                  "TRON", "NSSS", "GTRA", "JATI", "TYRE", "MPXL", "KLAS", "MAXI", "VKTR", "CRSN",
                                  "INET", "RMKO", "CNMA", "FOLK", "GRIA", "PPRI", "CYBR", "MUTU", "HUMI", "RSCH",
                                  "BABY", "IOTF", "KOCI", "PTPS", "STRK", "KOKA", "RGAS", "IKPM", "AYAM", "SURI",
                                  "ASLI", "GRPH", "SMGA", "UNTD", "TOSK", "MPIX", "MKAP", "LIVE", "HYGN", "BAIK",
                                  "VISI", "AREA", "MHKI", "ATLA", "DATA", "SOLA", "BATR", "PART", "ISEA", "BLES",
                                  "GUNA", "LABS", "DOSS", "NEST", "VERN", "BOAT", "NAIK", "KSIX", "RATU", "YOII",
                                  "HGII", "BRRC", "OBAT", "MINE", "ASPR", "PSAT", "COIN", "CDIA", "MERI", "KAQI",
                                  "FORE", "DKHH", "AYLS", "DADA", "ASPI", "ESTA", "BESS", "AMAN", "CARE", "PIPA",
                                  "NCKL", "AWAN", "DOOH", "CGAS", "NICE", "MSJA", "SMLE", "ACRO", "WIFI", "FAPA",
                                  "DCII", "KETR", "DGNS", "UFOE", "CHEK", "PMUI", "EMAS", "PJHB", "RLCO", "SUPA",
                                  "WBSA", "JELI", "EMMI", "PRDL", "RANS", "OBMD", "NASI", "BSML", "ADMF", "ADMG",
                                  "AGII", "AGRS", "AHAP", "AIMS", "PNSE"]
    akselerasi_ekonomi = ["CASH", "SOFA", "PPGL", "PLAN", "LFLO", "LUCY", "MGLV", "IPAC", "FLMC", "RUNS",
                          "IDEA", "WGSH", "SMKM", "NANO", "IBOS", "OLIV", "RCCC", "AMMS", "EURO", "KLIN",
                          "NINE", "ISAP", "SOUL", "BMBL", "NAYZ", "PACK", "CHIP", "KING", "HAJJ", "RELF",
                          "GRPM", "WIDI", "HBAT", "LMAX", "MSIE", "AEGS", "LOPI", "UDNG", "MEJA", "SPRE",
                          "MANG", "BUKA"]
    pemantauan_khusus = ["ABBA", "ACST", "ADES", "AKKU", "ALKA", "ALMI", "ALTO", "ARTI", "ASMI",
                         "BATA", "BEKS", "BHIT", "BIKA", "BIMA", "BLTA", "BLTZ", "BSWD", "BTEK", "BTEL",
                         "CANI", "CMPP", "CNKO", "COWL", "DUTI", "ELTY", "ETWA", "FASW", "GAMA", "GLOB",
                         "GOLL", "HADE", "HITS", "HOME", "HOTL", "IBFN", "IBST", "IIKP", "IKAI", "INAF",
                         "INRU", "INTA", "KARW", "KBRI", "KIAS", "KOIN", "KREN", "LCGP", "LMAS", "LMSH",
                         "MAGP", "MDRN", "MFMI", "MIRA", "MLIA", "MPPA", "MTFN", "MTSM", "MYTX", "OCAP",
                         "PBRX", "PLAS", "PLIN", "RIMO", "SCPI", "SIMA", "SKYB", "SMCB", "SMRU", "SONA",
                         "SRIL", "SUGI", "SUPR", "TARA", "TAXI", "TELE", "TFCO", "TIRT", "TRAM", "TRIL",
                         "TRIO", "UNSP", "VIVA", "WICO", "WIKA", "WSKT", "ZBRA", "MARI", "MKNT", "MTRA",
                         "INCF", "WSBP", "TAMU", "TGRA", "TOPS", "ARMY", "MAPB", "MABA", "NASA", "ZINC",
                         "PCAR", "BOSS", "JSKY", "INPS", "TDPM", "SWAT", "POLL", "NUSA", "ANDI", "DIGI",
                         "HKMU", "DUCK", "SOSS", "DEAL", "URBN", "FOOD", "MTPS", "CPRI", "HRME", "POSA",
                         "KAYU", "IPTV", "ENVY", "ARKA", "BAPI", "PURE", "SINI", "IFSH", "PGJO", "PURA",
                         "SBAT", "KBAG", "CBMF", "TECH", "TOYS", "PNGO", "PTDU", "PMMP", "BEBS", "FIMP",
                         "BAUT", "WINR", "RAFI", "KKES", "HILL", "SAGE", "TGUK", "RGAS", "PTMR", "MENN",
                         "WMPP", "IPPE", "POLY", "POOL", "PPRO"]
    # Set untuk filter cepat
    pemantauan_khusus_set = set(pemantauan_khusus)
    # Gabungan dasar semua emiten statis (non-khusus)
    base_non_khusus = list(dict.fromkeys(pengembangan + akselerasi_ekonomi))
    # Full IDX = non-Pemantauan Khusus
    full_idx_static_non_khusus = [
        c for c in base_non_khusus
        if c not in pemantauan_khusus_set
    ]
    # Auto-Fetch = SEMUA emiten (termasuk Pemantauan Khusus)
    full_idx_static_all = list(dict.fromkeys(
        base_non_khusus + pemantauan_khusus
    ))
    if mode == "Cepat (LQ45)":
        return lq45
    elif mode == "Papan Utama":
        return papan_utama
    elif mode == "Komprehensif (Utama + Pengembangan)":
        return pengembangan
    elif mode == "Full IDX":
        api_codes = fetch_all_idx_stocks()
        if api_codes:
            combined = list(dict.fromkeys(api_codes + full_idx_static_non_khusus))
            combined = [c for c in combined if c not in pemantauan_khusus_set]
            return combined
        return full_idx_static_non_khusus
    elif mode == "Auto-Fetch (API BEI)":
        api_codes = fetch_all_idx_stocks()
        if api_codes:
            combined = list(dict.fromkeys(api_codes + full_idx_static_all))
            # jangan filter apa pun, semua emiten diikutsertakan
            return combined[:1000]
        return full_idx_static_all
@st.cache_data(ttl=30)
def get_realtime_price(ticker):
    """Ambil harga real-time terpisah dari bar historis, untuk override kalau bar terakhir stale."""
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        return {
            "last_price": fi.get("last_price"),
            "market_state": fi.get("market_state") if hasattr(fi, "get") else None,
        }
    except Exception:
        return None

def cek_kesegaran_data(df_ihsg_preview, now_jkt, max_lag_minutes=20):
    """Return (is_stale, lag_minutes)."""
    if df_ihsg_preview.empty:
        return True, None
    last_bar_time = df_ihsg_preview.index[-1]
    if last_bar_time.tzinfo is None:
        last_bar_time = pytz.timezone("Asia/Jakarta").localize(last_bar_time)
    else:
        last_bar_time = last_bar_time.astimezone(pytz.timezone("Asia/Jakarta"))
    lag = (now_jkt - last_bar_time).total_seconds() / 60
    return lag > max_lag_minutes, lag
    
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
        news = [{'title': e.get('title','').strip(),
                 'summary': re.sub('<[^<]+?>','',e.get('summary','')),
                 'source':'Google News'} for e in feed.entries[:num]]
        return news, None
    except Exception as e:
        return [], str(e)
@st.cache_data(ttl=1800, show_spinner=False)
def get_headlines_for_ticker(ticker):
    """Ambil maks 3 judul berita terbaru dari Google News RSS."""
    try:
        news, _ = get_google_news_rss(f"{ticker} saham", num=3)
        return [n['title'] for n in news] if news else ["(tidak ada berita terbaru)"]
    except:
        return ["(gagal mengambil berita)"]
def get_ipot_news(query, num=5):
    """Ambil berita dari Ipotnews berdasarkan kata kunci."""
    try:
        import requests
        from bs4 import BeautifulSoup
        url = f"https://www.ipotnews.com/search?q={urllib.parse.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        news = []
        for item in soup.select('.title a')[:num]:   # selector bisa disesuaikan
            title = item.get_text(strip=True)
            link = item.get('href', '')
            news.append({'title': title, 'summary': '', 'source': 'Ipotnews'})
        return news, None
    except Exception as e:
        return [], str(e)

def get_yahoo_search_news(query_str, num=5):
    try:
        items = yf.Search(query_str).news or []
        news = []
        for item in items[:num]:
            inner = item.get('content') or item
            title = inner.get('title') or inner.get('shortTitle') or inner.get('headline') or ''
            summary = inner.get('summary') or inner.get('longSummary') or inner.get('description') or ''
            if title:
                news.append({'title':title,'summary':summary,'source':'Yahoo Search'})
        return news, None
    except:
        return [], "Yahoo Search gagal"

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
def robust_std(series):
    """Robust standard deviation berbasis MAD (Median Absolute Deviation)"""
    arr = np.array(series)
    if len(arr) < 4:
        return np.std(arr, ddof=0) if len(arr) > 1 else 0.001
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    return mad * 1.4826
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
def score_stock_tech(df_stock, ticker, ihsg_data):
    """
    Menghitung skor teknikal + metrik lengkap untuk scanner V12.
    Mengembalikan dictionary hasil atau None jika data tidak cukup.
    """
    try:
        closes = df_stock['Close'].values
        highs = df_stock['High'].values
        lows = df_stock['Low'].values
        volumes = df_stock['Volume'].values
        last_price = float(closes[-1])
        if last_price <= 0:
            return None

        ihsg_closes = ihsg_data['Close'].values
        if len(closes) < 65 or len(ihsg_closes) < 65:
            return None

        # Gunakan 60 bar terakhir untuk perhitungan
        n = min(60, len(closes) - 1, len(ihsg_closes) - 1)
        s_adj = closes[-n-1:]
        i_adj = ihsg_closes[-n-1:]
        sT = len(s_adj) - 1
        iT = len(i_adj) - 1
        if sT < 20 or iT < 20:
            return None

        s_ret = np.diff(s_adj) / s_adj[:-1]
        i_ret = np.diff(i_adj) / i_adj[:-1]

        # --- Beta & IHSG return 5 hari ---
        i_ret5 = (i_adj[iT] - i_adj[iT-5]) / i_adj[iT-5] if iT >= 5 else 0.0
        common_len = min(len(s_ret), len(i_ret))
        cov = np.cov(s_ret[:common_len], i_ret[:common_len])[0,1]
        var_i = np.var(i_ret[:common_len])
        beta = (cov / var_i).clip(-3, 3) if var_i > 1e-8 else 1.0
        beta_norm = np.clip(beta * i_ret5 / 0.05, -1, 1)

        # --- Momentum combo (3/5/10) ---
        mom3 = (s_adj[sT] - s_adj[sT-3]) / s_adj[sT-3] if sT >= 3 else 0.0
        mom5 = (s_adj[sT] - s_adj[sT-5]) / s_adj[sT-5] if sT >= 5 else 0.0
        mom10 = (s_adj[sT] - s_adj[sT-10]) / s_adj[sT-10] if sT >= 10 else 0.0
        mom_combo = mom3*0.50 + mom5*0.30 + mom10*0.20
        mom_norm = np.clip(mom_combo / 0.05, -1, 1)

        # --- Coppock Curve ---
        copp_std, copp_prev = coppock_curve(s_adj, 14, 11, 10)
        copp_fast, copp_fast_prev = coppock_curve(s_adj, 6, 4, 5)
        copp_rising = copp_std > copp_prev
        copp_fast_rising = copp_fast > copp_fast_prev
        is_turning_up = copp_rising and copp_prev <= 0.0
        is_turning_down = not copp_rising and copp_prev >= 0.0
        fast_align = 1.08 if (copp_fast_rising == copp_rising) else 0.92

        if is_turning_up:
            copp_dir_base = 1.0
        elif copp_std > 0 and copp_rising:
            copp_dir_base = 0.70
        elif copp_std <= 0 and copp_rising:
            copp_dir_base = 0.40
        elif is_turning_down:
            copp_dir_base = -1.0
        elif copp_std > 0 and not copp_rising:
            copp_dir_base = -0.30
        else:
            copp_dir_base = -0.70
        copp_norm = np.clip(copp_dir_base * fast_align, -1, 1)

        # Label Coppock
        if is_turning_up:
            copp_label = "🔼 Turning Up"
        elif copp_std > 0 and copp_rising:
            copp_label = "↑ Rising+"
        elif copp_std <= 0 and copp_rising:
            copp_label = "↑ Recovering"
        elif is_turning_down:
            copp_label = "🔽 Turning Down"
        else:
            copp_label = "↓ Bearish"

        # --- Mean Reversion (Z-Score) ---
        sigma20 = robust_std(s_ret[-20:])
        sma20 = np.mean(s_adj[-20:])
        z_score_val = (last_price - sma20) / (sigma20 * sma20 + 1e-9)
        mr_norm = np.clip(-z_score_val / 0.05, -1, 1)

        # --- RSI ---
        rsi_ch = s_ret[-14:]
        gains = np.mean(rsi_ch[rsi_ch > 0]) if np.any(rsi_ch > 0) else 0.0
        losses = -np.mean(rsi_ch[rsi_ch < 0]) if np.any(rsi_ch < 0) else 1e-6
        rsi_val = 100.0 - (100.0 / (1.0 + gains / (losses + 1e-9)))
        if rsi_val < 25: rsi_norm = 0.90
        elif rsi_val < 35: rsi_norm = 0.55
        elif rsi_val < 45: rsi_norm = 0.20
        elif rsi_val < 55: rsi_norm = -0.10
        elif rsi_val < 65: rsi_norm = -0.35
        elif rsi_val < 75: rsi_norm = -0.55
        else: rsi_norm = -0.80

        # --- Volume Surge ---
        vol_ma20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-20:].mean()
        vol5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
        vol_surge = np.clip((vol5 / (vol_ma20 + 1) - 1.0), -1, 1)

        # --- Breakout bonus ---
        res20 = np.max(highs[-21:-1]) if len(highs) >= 21 else np.max(highs)
        breakout_bonus = 0.10 if (last_price > res20 * 0.995 and vol_surge > 0.3) else 0.0

        # --- Tech Score ---
        tech_score = (mom_norm*0.30 + copp_norm*0.28 + beta_norm*0.17 +
                      mr_norm*0.10 + vol_surge*0.08 + rsi_norm*0.07 +
                      breakout_bonus)
        tech_score = np.clip(tech_score, -1.0, 1.0)

        # --- Sinyal ---
        if tech_score > 0.42: signal = "STRONG BUY ▲▲"
        elif tech_score > 0.18: signal = "BUY ▲"
        elif tech_score > 0.05: signal = "WEAK BUY ▲"
        elif tech_score < -0.42: signal = "STRONG SELL ▼▼"
        elif tech_score < -0.18: signal = "SELL ▼"
        else: signal = "NEUTRAL →"

        # --- Regime ---
        ema20_ihsg = pd.Series(i_adj).ewm(span=20, adjust=False).mean().iloc[-1]
        sma20_ihsg = np.mean(i_adj[-20:])
        risk_on = ema20_ihsg > sma20_ihsg and i_ret5 > 0
        fast_vc = np.std(s_ret[-3:]) / (np.std(s_ret[-20:]) + 1e-9)
        if not risk_on and fast_vc >= 1.2: regime = "PANIC"
        elif risk_on and fast_vc >= 1.0: regime = "VOL UP"
        elif risk_on: regime = "BULLISH"
        else: regime = "BEARISH"

        # --- Estimasi return & TP/SL ---
        alpha = np.mean(s_ret) - beta * np.mean(i_ret)
        mu_est = np.clip(beta * i_ret5 + alpha + mom_combo * 0.15, -0.04, 0.04)

        # --- Metrik tambahan ---
        # Bollinger %B
        std20 = np.std(s_adj[-20:])
        upper_bb = sma20 + 2*std20
        lower_bb = sma20 - 2*std20
        bb_pct = np.clip((last_price - lower_bb) / (upper_bb - lower_bb + 1e-9), 0, 1)

        # Trend Consistency
        ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1] if len(closes) >= 50 else ema20
        trend_searah = 0
        if len(closes) >= 6:
            for i in range(1, 6):
                if ((closes[-i] > closes[-i-1]) and (ema20 > ema50)) or \
                   ((closes[-i] < closes[-i-1]) and (ema20 < ema50)):
                    trend_searah += 1
            trend_consistency = trend_searah / 5 * 100
        else:
            trend_consistency = 50.0

        # Entry Zone sederhana (berbasis pivot & fraksi BEI)
        pivot = (highs[-1] + lows[-1] + closes[-1]) / 3.0
        s1 = 2 * pivot - highs[-1]
        entry_low = fraksi_bei(min(s1, last_price * (1 - sigma20)))
        entry_high = fraksi_bei(last_price)
        if entry_low >= entry_high:
            step = fraksi_step(last_price)
            entry_low = fraksi_bei(entry_high - step)

        # Take Profit Est (di atas harga pasar, dibulatkan ke fraksi BEI)
        tp_dist_pct = max(0.02, max(mu_est, 0.01) + 0.8 * sigma20)
        tp_est = fraksi_bei(last_price * (1 + tp_dist_pct))
        if tp_est <= last_price:
            step = fraksi_step(last_price)
            tp_est = fraksi_bei(last_price + 2 * step)

        # Stop Loss Est (selalu di bawah entry_low & last_price, dibulatkan ke fraksi BEI)
        sl_dist_pct = max(0.02, 1.5 * sigma20)
        sl_est = fraksi_bei(entry_low * (1 - sl_dist_pct))
        if sl_est >= entry_low:
            step = fraksi_step(entry_low)
            sl_est = fraksi_bei(entry_low - 2 * step)

        # Likuiditas
        avg_value = np.mean(volumes[-20:] * closes[-20:])
        if avg_value >= 1e9:
            likuiditas_str = f"Rp {avg_value/1e9:.2f} M/hari"
        elif avg_value >= 1e6:
            likuiditas_str = f"Rp {avg_value/1e6:.0f} Jt/hari"
        else:
            likuiditas_str = f"Rp {avg_value:,.0f}"

        # Risk/Reward
        risk = last_price - sl_est
        reward = tp_est - last_price
        rrr = reward / risk if risk > 0 else 0.0

        # Confidence
        confidence = min(0.99, 0.5 + abs(tech_score) * 0.5)

        return {
            "ticker": ticker,
            "techScore": tech_score,
            "signal": signal,
            "muEst": mu_est,
            "coppockLabel": copp_label,
            "lastPrice": last_price,
            "tpEst": tp_est,
            "slEst": sl_est,
            "regime": regime,
            "rsi": rsi_val,
            "beta": beta,
            "momScore": mom_combo,
            "isCoppockTurningUp": is_turning_up,
            "volSurge": vol_surge,
            "zScore": z_score_val,
            "bbPct": bb_pct,
            "trendConsistency": trend_consistency,
            "entryLow": entry_low,
            "entryHigh": entry_high,
            "likuiditas": likuiditas_str,
            "rrr": rrr,
            "confidence": confidence
        }

    except Exception as e:
        # st.write(f"Error scoring {ticker}: {e}")  # untuk debugging
        return None
def generate_regime_insight(regime, adx, ofi_raw, ihsg_cond):
    base = REGIME_INFO.get(regime, "Rezim tidak terdefinisi.")
    notes = []

    # Analisis OFI
    if ofi_raw > 0.5:
        notes.append("🔹 OFI positif kuat → akumulasi tinggi, konfirmasi bullish.")
    elif ofi_raw < -0.5:
        notes.append("🔹 OFI negatif signifikan → tekanan jual, waspadai potensi distribusi.")
    elif ofi_raw < 0:
        notes.append("🔹 OFI sedikit negatif → aliran dana netral cenderung keluar.")

    # Analisis ADX (selalu tampil)
    if adx > 40:
        notes.append("🔹 ADX > 40 → tren sangat kuat, tapi waspadai kejenuhan.")
    elif adx < 20:
        notes.append("🔹 ADX rendah → pasar sedang konsolidasi, breakout mungkin terjadi.")
    else:
        notes.append(f"🔹 ADX {adx:.1f} → kekuatan tren moderat.")   # ← tambahan

    # Tambahan untuk RISK-ON / RISK-OFF
    if "RISK-ON" in ihsg_cond:
        notes.append("🔹 Sentimen pasar luas mendukung (RISK-ON).")
    elif "RISK-OFF" in ihsg_cond:
        notes.append("🔹 Sentimen pasar luas sedang defensif (RISK-OFF).")
    else:
        notes.append(f"🔹 Sentimen pasar luas: {ihsg_cond}")

    if notes:
        return base + " " + " ".join(notes)
    return base
# ==================== FUNGSI ANALISIS UTAMA ====================
def analyze_stock(ticker_input, harga_manual, sudah_beli, harga_beli_float, is_daytrade, v12_mem=None, fee_beli_pct=0.15, fee_jual_pct=0.25):
    """
    Menjalankan analisis lengkap untuk satu mode (swing/daytrade).
    Mengembalikan dictionary hasil atau None jika data tidak cukup.
    """
    # ------------------------------------------------------------------
    # 1. AMBIL DATA
    # ------------------------------------------------------------------
    bars_per_day_map = {"5m": 54, "15m": 18, "30m": 9, "60m": 5}

    if is_daytrade:
        actual_interval = "5m"
        df = load_stock_data(ticker_input, period="5d", interval=actual_interval)
        if df.empty or len(df) < 20:
            actual_interval = "15m"
            df = load_stock_data(ticker_input, period="5d", interval=actual_interval)
        if df.empty or len(df) < 20:
            actual_interval = "30m"
            df = load_stock_data(ticker_input, period="5d", interval=actual_interval)
        if df.empty or len(df) < 20:
            actual_interval = "60m"
            df = load_stock_data(ticker_input, period="5d", interval=actual_interval)
        df_ihsg = load_ihsg_data(period="5d", interval="5m")
        df_daily = load_stock_data(ticker_input, period="1mo", interval="1d")
    else:
        actual_interval = "1d"
        df = load_stock_data(ticker_input, period="2y", interval="1d")
        df_ihsg = load_ihsg_data(period="2y", interval="1d")
        df_daily = df

    if df.empty:
        return None

    # ------------------------------------------------------------------
    # 2. PERHITUNGAN DASAR
    # ------------------------------------------------------------------
    harga_terakhir_asli = float(df['Close'].iloc[-1])
    harga_terakhir = harga_terakhir_manual if harga_terakhir_manual else harga_terakhir_asli

    floating_pl_pct = None
    if sudah_beli and harga_beli_float and harga_beli_float > 0:
        floating_pl_pct = (harga_terakhir - harga_beli_float) / harga_beli_float * 100

    returns = df['Close'].pct_change().dropna()
    if len(returns) < 20:
        return None

    # ------------------------------------------------------------------
    # 3. INDIKATOR TEKNIKAL
    # ------------------------------------------------------------------
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ADX'] = compute_adx_series(df)

    if is_daytrade:
        df['Mom5D'] = df['Close'].pct_change(10) * 100   # 10 bar intraday
    else:
        df['Mom5D'] = df['Close'].pct_change(5) * 100    # 5 hari

    df['ZScore'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean() if 'Volume' in df.columns else 0

    # OFI
    df['Delta'] = np.where(df['Close'] > df['Open'], df['Volume'], -df['Volume'])
    df['Cumulative_OFI'] = df['Delta'].cumsum()
    df['OFI_raw'] = df['Delta'] / df['Volume'].rolling(20).mean().fillna(1)

    # VWAP hanya untuk daytrade
    if is_daytrade:
        df['CumVol'] = df['Volume'].cumsum()
        df['CumPV'] = (df['Close'] * df['Volume']).cumsum()
        df['VWAP'] = df['CumPV'] / df['CumVol']
        vwap_now = df['VWAP'].iloc[-1]
        vwap_bias = "Di Atas VWAP (Bullish)" if harga_terakhir > vwap_now else "Di Bawah VWAP (Bearish)"
    else:
        vwap_now = None
        vwap_bias = "N/A"

    # ------------------------------------------------------------------
    # 4. FUNDAMENTAL
    # ------------------------------------------------------------------
    try:
        ticker_info = yf.Ticker(ticker_input).info
    except:
        ticker_info = {}
    mc = ticker_info.get('marketCap')
    per = ticker_info.get('trailingPE') or ticker_info.get('forwardPE')
    pbv = ticker_info.get('priceToBook')
    roe = ticker_info.get('returnOnEquity')
    de = ticker_info.get('debtToEquity')

    # ------------------------------------------------------------------
    # 5. BERITA & SENTIMEN
    # ------------------------------------------------------------------
    news_pool = []
    translator_en = GoogleTranslator(source='auto', target='en') if TRANSLATOR_AVAILABLE else None
    translator_id = GoogleTranslator(source='auto', target='id') if TRANSLATOR_AVAILABLE else None

    rss, _ = get_google_news_rss(f"{ticker_raw} saham")
    if rss:
        news_pool.extend(rss)
    ysearch, _ = get_yahoo_search_news(f"{ticker_raw} saham")
    if ysearch:
        news_pool.extend(ysearch)
    ipot, _ = get_ipot_news(f"{ticker_raw}")
    if ipot:
        news_pool.extend(ipot)

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
    translated = []
    for n in unique_news:
        if TRANSLATOR_AVAILABLE and translator_id:
            try:
                translated.append(translator_id.translate(n['title']))
            except:
                translated.append("")
        else:
            translated.append("")
    sentimen_status = "Positif 🟢" if avg_sentiment >= 0.05 else ("Negatif 🔴" if avg_sentiment <= -0.05 else "Netral ⚪")

    # ------------------------------------------------------------------
    # 6. THRESHOLD & DISTRIBUSI
    # ------------------------------------------------------------------
    if is_daytrade:
        n_recent = min(200, len(df))
        df_thresh = df.iloc[-n_recent:]
    else:
        split_idx = max(126, len(df) - 126)
        df_thresh = df.iloc[:split_idx]

    returns_thresh = df_thresh['Close'].pct_change().dropna()
    adx_threshold = np.percentile(df_thresh['ADX'].dropna(), 75) if not df_thresh['ADX'].dropna().empty else 20
    z_oversold_th = -1.5
    mom_median_th = np.percentile(df_thresh['Mom5D'].dropna(), 50) if not df_thresh['Mom5D'].dropna().empty else 0.0

    def t_loglike(p, d):
        if p[0] <= 2 or p[2] <= 0:
            return np.inf
        return -np.sum(student_t.logpdf(d, p[0], p[1], p[2]))

    res_opt = minimize(
        t_loglike,
        [5, returns_thresh.mean(), returns_thresh.std()],
        bounds=[(2.1, 100), (-0.1, 0.1), (1e-6, None)],
        args=(returns_thresh,),
        method='L-BFGS-B'
    )
    df_est, t_loc, t_scale = res_opt.x if res_opt.success else (5, returns_thresh.mean(), returns_thresh.std())

    # ------------------------------------------------------------------
    # 7. REGIME
    # ------------------------------------------------------------------
    def get_regime_row(row):
        h, e20, e50, a, z, m = row['Close'], row['EMA20'], row['EMA50'], row['ADX'], row['ZScore'], row['Mom5D']
        if a > adx_threshold:
            if h > e20 and e20 > e50:
                return ("Strong Bullish 🚀", "RISK-ON 🔥") if (m > mom_median_th or z > z_oversold_th) else ("Bullish 📈", "RISK-ON 🔥")
            elif h < e20 and e20 < e50:
                return ("Panic Sell 🚨", "RISK-OFF 🛑") if (m < mom_median_th or z < z_oversold_th) else ("Bearish 🔻", "RISK-OFF 🛑")
            elif h > e20 and e20 < e50:
                return ("Early Recovery 🔄", "TRANSISI ⚠️")
            elif h < e20 and e20 > e50:
                return ("Distribution 📉", "TRANSISI ⚠️")
            else:
                return ("Konsolidasi Tren ↔️", "NEUTRAL ⚖️")
        else:
            if h > e20 and e20 > e50:
                return ("Bullish Accumulation 🏗️", "NEUTRAL ⚖️")
            elif h < e20 and e20 < e50:
                return ("Bearish Accumulation 🧊", "NEUTRAL ⚖️")
            elif h > e20 and e20 < e50:
                return ("Sideways Bias Naik ↗️", "NEUTRAL ⚖️")
            elif h < e20 and e20 > e50:
                return ("Sideways Bias Turun ↘️", "NEUTRAL ⚖️")
            else:
                return ("Sideways Normal ↔️", "NEUTRAL ⚖️")

    regime, ihsg_cond = get_regime_row(df.iloc[-1])
    adx = df['ADX'].iloc[-1]

    # ------------------------------------------------------------------
    # 8. BETA
    # ------------------------------------------------------------------
    beta_ihsg = 1.0
    ihsg_ret = pd.Series(dtype=float)
    try:
        if not df_ihsg.empty:
            ihsg_ret = df_ihsg['Close'].pct_change().dropna()
            common = returns.index.intersection(ihsg_ret.index)
            if len(common) > 20:
                beta_ihsg = np.cov(returns.loc[common], ihsg_ret.loc[common])[0, 1] / np.var(ihsg_ret.loc[common])
    except:
        pass

    # ------------------------------------------------------------------
    # 9. ATR & RSI
    # ------------------------------------------------------------------
    df['TR'] = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr14_val = df['TR'].rolling(14).mean().iloc[-1]
    atr_pct = (atr14_val / harga_terakhir_asli) * 100

    now_jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
    if is_daytrade:
        bars_remaining = hitung_bars_remaining(now_jkt, actual_interval, bars_per_day_map)
    else:
        bars_remaining = None

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean().iloc[-1]
    avg_loss = loss.rolling(14).mean().iloc[-1]
    if avg_loss is None or avg_loss == 0:
        rsi14 = 100.0
    else:
        rsi14 = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))

    # ------------------------------------------------------------------
    # 10. PIVOT
    # ------------------------------------------------------------------
    if is_daytrade:
        today_jkt = datetime.now(pytz.timezone("Asia/Jakarta")).date()
        if not df_daily.empty:
            df_daily_filtered = df_daily[df_daily.index.date < today_jkt]
            if not df_daily_filtered.empty:
                prev_day = df_daily_filtered.iloc[-1]
                hi_daily = float(prev_day['High'])
                lo_daily = float(prev_day['Low'])
                cl_daily = float(prev_day['Close'])
            else:
                prev_day = None
                for i in range(1, min(len(df_daily), 6)):
                    row = df_daily.iloc[-i]
                    h_val = float(row['High'])
                    l_val = float(row['Low'])
                    c_val = float(row['Close'])
                    if h_val != l_val and h_val > 0 and l_val > 0:
                        prev_day = row
                        hi_daily, lo_daily, cl_daily = h_val, l_val, c_val
                        break
                if prev_day is None:
                    last = df_daily.iloc[-1]
                    hi_daily = float(last['High'])
                    lo_daily = float(last['Low'])
                    cl_daily = float(last['Close'])
        else:
            hi_daily = float(df['High'].iloc[-1])
            lo_daily = float(df['Low'].iloc[-1])
            cl_daily = float(df['Close'].iloc[-1])

        if hi_daily != lo_daily:
            pp = (hi_daily + lo_daily + cl_daily) / 3
            r1 = 2 * pp - lo_daily
            s1 = 2 * pp - hi_daily
            r2 = pp + (hi_daily - lo_daily)
            s2 = pp - (hi_daily - lo_daily)
        else:
            pp = r1 = s1 = r2 = s2 = cl_daily
    else:
        hi = lo = cl = None
        for i in range(1, min(6, len(df))):
            row = df.iloc[-i]
            h_val = float(row['High']); l_val = float(row['Low']); c_val = float(row['Close'])
            if h_val != l_val and h_val > 0 and l_val > 0:
                hi, lo, cl = h_val, l_val, c_val
                break
        if hi is None:
            hi = float(df['High'].iloc[-1]); lo = float(df['Low'].iloc[-1]); cl = float(df['Close'].iloc[-1])
        if hi == lo:
            pp = r1 = s1 = r2 = s2 = cl
        else:
            pp = (hi + lo + cl) / 3
            r1 = 2 * pp - lo; s1 = 2 * pp - hi
            r2 = pp + (hi - lo); s2 = pp - (hi - lo)

    # ------------------------------------------------------------------
    # 11. V12 ADAPTIVE SIGNAL
    # ------------------------------------------------------------------
    adaptive_w = get_adaptive_weights(ticker_raw, regime, v12_mem=v12_mem)
    coppock_val, coppock_prev = coppock_curve(df['Close'].values)
    factor_signals = {
        "Momentum": (df['Mom5D'].iloc[-1] - mom_median_th) / max(0.1, df['Mom5D'].std()),
        "AI_Senti": avg_sentiment,
        "MeanRev": -df['ZScore'].iloc[-1] / 3.0,
        "Beta_IHSG": beta_ihsg * (ihsg_ret.iloc[-1] if not ihsg_ret.empty else 0.0),
        "Coppock": coppock_val / 10.0,
        "OFI": df['OFI_raw'].iloc[-1] / 3.0
    }
    norm_signals = {k: max(-1.0, min(1.0, v)) for k, v in factor_signals.items()}
    total_score = sum(norm_signals[k] * adaptive_w.get(k, 0.15) for k in FACTOR_KEYS)

    if total_score > 0.3:
        signal = "🔥 STRONG BUY"
    elif total_score > 0.1:
        signal = "⚡ BUY (TACTICAL)"
    elif total_score > -0.1:
        signal = "⏸️ HOLD / WAIT"
    else:
        signal = "🚨 AVOID"

    # ------------------------------------------------------------------
    # 12. ENTRY ZONE
    # ------------------------------------------------------------------
    if s1 >= harga_terakhir * 0.98:
        entry_low = s1
    else:
        entry_low = harga_terakhir * (1 - atr_pct / 100)

    if "STRONG BUY" in signal:
        entry_high = harga_terakhir
    else:
        entry_high = harga_terakhir * (1 - 0.3 * atr_pct / 100)

    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low

    min_entry_width = 0.5 * atr14_val
    if (entry_high - entry_low) < min_entry_width:
        entry_low = max(0, entry_high - min_entry_width)
        entry_high = entry_low + min_entry_width
        entry_high = min(entry_high, harga_terakhir)

    # Baca entry_error dari v12_mem (parameter thread-safe), fallback ke session_state
    if v12_mem is not None:
        mem_for_entry = v12_mem.get(ticker_raw, {})
    else:
        mem_for_entry = st.session_state.v12_memory.get(ticker_raw, {})
    entry_error = mem_for_entry.get('entry_error_ema', 0.0)
    if entry_error > 0:
        entry_low += entry_error * 0.2
        entry_high += entry_error * 0.2

    entry_high = min(entry_high, harga_terakhir)
    entry_low = min(entry_low, entry_high)

    entry_low_f = fraksi_bei(entry_low)
    entry_high_f = fraksi_bei(entry_high)
    entry_zone_f = f"Rp {entry_low_f:,.0f} - Rp {entry_high_f:,.0f}"

    # ------------------------------------------------------------------
    # 13. SL & TP
    # ------------------------------------------------------------------
    sl_mult = 1.0
    if adx > 30 and 30 < rsi14 < 70:
        sl_mult = 0.75
    elif adx < 20:
        sl_mult = 1.25
    if rsi14 > 70 or rsi14 < 30:
        sl_mult = 1.5

    tp_mult_low = 1.5
    tp_mult_high = 2.5
    if adx > 30 and 30 < rsi14 < 70:
        tp_mult_low, tp_mult_high = 2.0, 3.0
    elif adx < 20:
        tp_mult_low, tp_mult_high = 1.2, 1.8

    if is_daytrade:
        base_sl_dist = harga_terakhir * 0.04 * sl_mult
        min_ticks_dist = 2 * fraksi_step(entry_low)
        sl_dist = max(min_ticks_dist, base_sl_dist)
        sl_harga = entry_low - sl_dist
    else:
        sl_harga = entry_low - sl_mult * atr14_val

    sl_harga = fraksi_bei(sl_harga)
    step = fraksi_step(entry_low)
    if sl_harga >= entry_low:
        sl_harga = fraksi_bei(entry_low - 2 * step)
    if sl_harga <= 0:
        sl_harga = fraksi_bei(harga_terakhir * 0.95)
    sl_pct = (harga_terakhir - sl_harga) / harga_terakhir * 100

    if is_daytrade:
        # --- PERHITUNGAN TP DAYTRADE BERBASIS ATR INTRADAY & FAKTOR FEE BROKER ---
        # 1. Target ATR Intraday (5m)
        tp_low_raw = entry_low + (tp_mult_low * atr14_val)
        tp_high_raw = entry_low + (tp_mult_high * atr14_val)

        # 2. Safety Floor untuk Memastikan Cover Fee Broker (Beli + Jual) + Target Net Profit Margin (+0.6% net)
        total_fee_pct = (fee_beli_pct + fee_jual_pct) / 100.0
        min_net_margin = 0.0060
        fee_floor = entry_low * (1.0 + total_fee_pct + min_net_margin)

        # Gunakan nilai terbesar antara ATR Intraday & Fee Floor
        tp_low_raw = max(tp_low_raw, fee_floor)
        tp_high_raw = max(tp_high_raw, tp_low_raw + (2 * fraksi_step(tp_low_raw)))

        # 3. Pastikan minimal 2 tick di atas entry_low dan harga_terakhir agar komisi tercover penuh
        min_tp_low_ticks = max(entry_low + (2 * step), harga_terakhir + (2 * fraksi_step(harga_terakhir)))
        if tp_low_raw < min_tp_low_ticks:
            tp_low_raw = min_tp_low_ticks

        tp_low = fraksi_bei(tp_low_raw)
        tp_high = fraksi_bei(tp_high_raw)
        if tp_low <= entry_low:
            tp_low = fraksi_bei(entry_low + 2 * step)
        if tp_high <= tp_low:
            tp_high = fraksi_bei(tp_low + 2 * fraksi_step(tp_low))
    else:
        # --- PERHITUNGAN TP SWING (RESISTANCE HARIAN / PIVOT R1 R2) ---
        if r1 > harga_terakhir:
            tp_low = r1
        else:
            tp_low = harga_terakhir + tp_mult_low * atr14_val
        if r2 > harga_terakhir:
            tp_high = r2
        else:
            tp_high = harga_terakhir + tp_mult_high * atr14_val
        if tp_low > tp_high:
            tp_low, tp_high = tp_high, tp_low

    tp_pct_low = (tp_low - harga_terakhir) / harga_terakhir * 100
    tp_pct_high = (tp_high - harga_terakhir) / harga_terakhir * 100

    risk = harga_terakhir - sl_harga
    reward = tp_low - harga_terakhir
    rrr = reward / risk if risk > 0 else 0
    if rrr >= 2.0:
        rrr_status = "Sangat Baik (≥ 2.0) 🟢"
    elif rrr >= 1.5:
        rrr_status = "Baik (1.5 - 2.0) 🟢"
    elif rrr >= 1.0:
        rrr_status = "Cukup (1.0 - 1.5) 🟡"
    else:
        rrr_status = "Buruk (< 1.0) 🔴"

    # Breakout
    if is_daytrade:
        bars_per_day = bars_per_day_map.get(actual_interval, 54)
        if len(df) >= bars_per_day:
            res20 = float(df['High'].iloc[-bars_per_day:-1].max())
            breakout_label = f"Breakout Sesi Sebelumnya ({bars_per_day} bar)"
        else:
            res20 = float(df['High'].max())
            breakout_label = "Breakout N-Bar"
    else:
        if len(df) >= 21:
            res20 = float(df['High'].iloc[-21:-1].max())
        else:
            res20 = float(df['High'].max())
        breakout_label = "Breakout 20 Hari"
    breakout = f"YES (🔥)" if harga_terakhir > res20 else "NO"

    # ------------------------------------------------------------------
    # 14. BACKTEST
    # ------------------------------------------------------------------
    def generate_signals_vectorized(dataframe, mom_th):
        score = pd.Series(0, index=dataframe.index)
        is_uptrend = (dataframe['Close'] > dataframe['EMA20']) & (dataframe['EMA20'] > dataframe['EMA50'])
        score += is_uptrend.astype(int) * 2
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

    if is_daytrade:
        backtest_window = min(200, len(df))
    else:
        backtest_window = 126
    df_back = df.iloc[-backtest_window:].copy()
    trades, daily_returns = [], []
    in_position, entry_price = False, 0.0
    for i in range(len(df_back)):
        curr_sig = df_back['Signal'].iloc[i]
        curr_close = float(df_back['Close'].iloc[i])
        prev_close = float(df_back['Close'].iloc[i-1]) if i > 0 else curr_close
        if in_position:
            daily_returns.append((curr_close - prev_close) / prev_close if prev_close else 0)
            if "AVOID" in curr_sig or i == len(df_back) - 1:
                trades.append((curr_close - entry_price) / entry_price)
                in_position = False
        else:
            daily_returns.append(0.0)
            if "BUY" in curr_sig:
                in_position, entry_price = True, curr_close

    if trades:
        win_bt = sum(1 for r in trades if r > 0) / len(trades)
        loss_trades = [r for r in trades if r < 0]
        profit_trades = [r for r in trades if r > 0]
        pf_bt = abs(sum(profit_trades) / sum(loss_trades)) if loss_trades else np.inf
        avg_bt = np.mean(trades)
        equity = np.cumprod([1 + r for r in trades])
        max_dd_bt = float(np.min(equity / np.maximum.accumulate(equity) - 1) * 100) if len(equity) else 0
        daily_ret = np.array(daily_returns)
        if is_daytrade:
            bars_per_day = bars_per_day_map.get(actual_interval, 54)
            annual_factor = np.sqrt(bars_per_day * 252)
        else:
            annual_factor = np.sqrt(252)
        sharpe_bt = (daily_ret.mean() / daily_ret.std()) * annual_factor if daily_ret.std() else 0
        trades_bt = len(trades)
    else:
        win_bt = pf_bt = avg_bt = max_dd_bt = sharpe_bt = trades_bt = 0

    # ------------------------------------------------------------------
    # 15. KELLY & DRAWDOWN
    # ------------------------------------------------------------------
    roll_max_th = df_thresh['Close'].cummax()
    drawdown_th = (df_thresh['Close'] - roll_max_th) / roll_max_th
    max_dd = float(drawdown_th.min() * 100)
    max_dd_30 = float(drawdown_th.tail(30).min() * 100) if len(drawdown_th) >= 30 else max_dd

    if trades_bt >= 2:
        win_r = win_bt
        avg_g = np.mean(profit_trades) if profit_trades else 0.01
        avg_l = abs(np.mean(loss_trades)) if loss_trades else 0.01
    else:
        win_r = len(returns_thresh[returns_thresh > 0]) / len(returns_thresh)
        avg_g = returns_thresh[returns_thresh > 0].mean() if win_r > 0 else 0.01
        avg_l = abs(returns_thresh[returns_thresh < 0].mean()) if len(returns_thresh[returns_thresh < 0]) else 0.01

    wl = avg_g / avg_l if avg_l else 1
    kelly_raw = win_r - (1 - win_r) / wl
    ret_skew = float(skew(returns_thresh))
    ret_kurt = float(kurtosis(returns_thresh, fisher=True))
    kurt_penalty = 0.5 if ret_kurt > 3 else 1.0
    kelly_adj = min(0.25, max(0.0, kelly_raw * 0.3 * (0.5 if ret_skew < -0.5 else 1) * kurt_penalty))

    # ------------------------------------------------------------------
    # 16. MONTE CARLO
    # ------------------------------------------------------------------
    if is_daytrade:
        n_sim = 2000
        n_steps = max(1, bars_remaining)
    else:
        n_sim = 2000
        n_steps = 30

    latest_vol = np.sqrt(df['Close'].pct_change().ewm(alpha=0.06).var().iloc[-1])
    scale_corrected = latest_vol / np.sqrt(df_est / (df_est - 2)) if df_est > 2 else latest_vol
    theta_ou = estimate_theta_ou(df['Close'])
    locked_log_mean20 = np.log(df['Close']).tail(20).mean()

    paths = np.zeros((n_steps, n_sim))
    current_log = np.ones(n_sim) * np.log(harga_terakhir)
    for step in range(n_steps):
        inov = student_t.rvs(df_est, loc=0, scale=scale_corrected, size=n_sim)
        current_log = current_log + theta_ou * (locked_log_mean20 - current_log) + inov
        paths[step] = np.exp(current_log)

    final_prices = paths[-1, :]
    est_besok = float(np.median(final_prices))
    if "STRONG BUY" in signal:
        est_besok_sinyal = float(np.percentile(final_prices, 75))
    elif "BUY" in signal:
        est_besok_sinyal = float(np.percentile(final_prices, 65))
    elif "HOLD" in signal:
        est_besok_sinyal = float(np.percentile(final_prices, 50))
    else:
        est_besok_sinyal = float(np.percentile(final_prices, 35))

    low_est, up_est = float(np.percentile(final_prices, 25)), float(np.percentile(final_prices, 75))
    prob_bull = (final_prices > harga_terakhir).mean() * 100
    hit_tp = (np.any(paths >= r1, axis=0).sum() / n_sim) * 100
    hit_sl = (np.any(paths <= s2, axis=0).sum() / n_sim) * 100

    if is_daytrade:
        estimasi_label = "Estimasi Sesi Berikutnya"
        prob_label = "Prob Naik Sesi Berikutnya"
    else:
        estimasi_label = "Estimasi Besok"
        prob_label = "Prob Naik Besok"

    # ------------------------------------------------------------------
    # 17. METRIK TAMBAHAN
    # ------------------------------------------------------------------
    if "STRONG BUY" in signal:
        signal_score = 0.7 + (prob_bull / 200)
    elif "BUY" in signal:
        signal_score = 0.4 + (prob_bull / 200)
    elif "HOLD" in signal:
        signal_score = 0.2 + (prob_bull / 300)
    else:
        signal_score = max(0, (prob_bull - 30) / 100)
    signal_score = min(1.0, max(0.0, signal_score))
    confidence = min(0.99, 0.5 + (signal_score * 0.5) + (win_bt - 0.5) * 0.1)
    if confidence is None or np.isnan(confidence):
        confidence = 0.5

    trend_consistency = np.mean([
        1 if (df['Close'].iloc[-i] > df['Close'].iloc[-i-1]) == (df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]) else 0
        for i in range(1, 11)
    ]) * 100
    if np.isnan(trend_consistency):
        trend_consistency = 50.0

    avg_vol_5 = df['Volume'].iloc[-5:].mean()
    avg_vol_20 = df['Volume'].iloc[-20:].mean()
    if avg_vol_20 > 0:
        vol_surge_pct = ((avg_vol_5 / avg_vol_20) - 1) * 100
    else:
        vol_surge_pct = 0.0

    avg_value = (df['Volume'].iloc[-5:] * df['Close'].iloc[-5:]).mean()
    if np.isnan(avg_value):
        avg_value = 0.0
    if avg_value >= 1e9:
        likuiditas_str = f"Rp {avg_value/1e9:.2f} M"
    elif avg_value >= 1e6:
        likuiditas_str = f"Rp {avg_value/1e6:.0f} Jt"
    elif avg_value >= 1e3:
        likuiditas_str = f"Rp {avg_value/1e3:.0f} rb"
    else:
        likuiditas_str = f"Rp {avg_value:,.0f}"

    if rsi14 > 70:
        rsi_status = "Overbought"
    elif rsi14 < 30:
        rsi_status = "Oversold"
    else:
        rsi_status = "Normal"

    zscore_val = df['ZScore'].iloc[-1]
    if pd.isna(zscore_val):
        zscore_val = 0.0
    if zscore_val > 2:
        zs_status = "Overbought"
    elif zscore_val < -2:
        zs_status = "Oversold"
    else:
        zs_status = "Normal"

    if vol_surge_pct > 50:
        vs_status = "Tinggi"
    elif vol_surge_pct < -30:
        vs_status = "Rendah"
    else:
        vs_status = "Normal"

    coppock_rising = coppock_val > coppock_prev
    coppock_turning_up = coppock_rising and coppock_prev <= 0
    if coppock_turning_up:
        coppock_status = "Turning Up"
    elif coppock_rising:
        coppock_status = "Rising"
    else:
        coppock_status = "Falling"

    est_besok_f = fraksi_bei(est_besok)
    est_besok_sinyal_f = fraksi_bei(est_besok_sinyal)
    low_est_f = fraksi_bei(low_est)
    up_est_f = fraksi_bei(up_est)
    tp_low_f = fraksi_bei(tp_low)
    tp_high_f = fraksi_bei(tp_high)
    sl_harga_f = fraksi_bei(sl_harga)

    # ------------------------------------------------------------------
    # 18. RINGKASAN UNTUK RIWAYAT
    # ------------------------------------------------------------------
    ringkasan = {
        "Waktu": datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M"),
        "Saham": ticker_raw,
        "Harga": f"{harga_terakhir:,.0f}",
        "Sinyal": signal,
        "Estimasi": f"{est_besok:,.0f}",
        "Estimasi_Netral": f"Rp {est_besok_f:,.0f}",
        "Estimasi_Sinyal": f"Rp {est_besok_sinyal_f:,.0f}",
        "Prob Naik": f"{prob_bull:.1f}%",
        "RRR": f"{rrr:.2f}",
        "Sentimen": f"{avg_sentiment:.2f} ({sentimen_status})",
        "Rezim": regime,
        "TP%": f"{tp_pct_low:.1f}% - {tp_pct_high:.1f}%",
        "SL%": f"{sl_pct:.1f}%",
        "AI_Insight": "",
        "Score": f"{signal_score:.3f}",
        "Confidence": f"{confidence:.0%}",
        "Coppock": coppock_status,
        "Est_Return": f"{((est_besok - harga_terakhir) / harga_terakhir * 100):+.2f}%",
        "Est_Return_Sinyal": f"{((est_besok_sinyal - harga_terakhir) / harga_terakhir * 100):+.2f}%",
        "TP_Harga": f"{tp_low_f:,.0f} - {tp_high_f:,.0f}",
        "TP_Range": f"Rp {tp_low_f:,.0f} - Rp {tp_high_f:,.0f}",
        "SL_Harga": f"{sl_harga_f:,.0f}",
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
        "Entry_Zone": entry_zone_f,
        "Gaya": "DT" if is_daytrade else "SW",
        "Status_Posisi": "Sudah Beli" if sudah_beli else "Belum",
        "Harga_Beli": f"{harga_beli_float:,.0f}" if harga_beli_float else "",
        "Floating_PL": f"{floating_pl_pct:+.2f}%" if floating_pl_pct is not None else ""
    }

    # ------------------------------------------------------------------
    # 19. KUMPULKAN RESULT
    # ------------------------------------------------------------------
    result = {
        "df": df,
        "df_back": df_back,
        "harga_terakhir": harga_terakhir,
        "signal": signal,
        "entry_zone_f": entry_zone_f,
        "sl_harga_f": sl_harga_f,
        "tp_low_f": tp_low_f,
        "tp_high_f": tp_high_f,
        "rrr": rrr,
        "rrr_status": rrr_status,
        "prob_bull": prob_bull,
        "signal_score": signal_score,
        "confidence": confidence,
        "est_besok_f": est_besok_f,
        "est_besok_sinyal_f": est_besok_sinyal_f,
        "low_est_f": low_est_f,
        "up_est_f": up_est_f,
        "tp_pct_low": tp_pct_low,
        "tp_pct_high": tp_pct_high,
        "sl_pct": sl_pct,
        "adx": adx,
        "rsi14": rsi14,
        "atr_pct": atr_pct,
        "avg_sentiment": avg_sentiment,
        "sentimen_status": sentimen_status,
        "headlines": headlines,
        "sources": sources,
        "translated": translated,
        "regime": regime,
        "ihsg_cond": ihsg_cond,
        "coppock_val": coppock_val,
        "coppock_prev": coppock_prev,
        "coppock_turning_up": coppock_turning_up,
        "beta_ihsg": beta_ihsg,
        "win_bt": win_bt,
        "pf_bt": pf_bt,
        "avg_bt": avg_bt,
        "max_dd_bt": max_dd_bt,
        "sharpe_bt": sharpe_bt,
        "trades_bt": trades_bt,
        "kelly_adj": kelly_adj,
        "max_dd": max_dd,
        "max_dd_30": max_dd_30,
        "breakout": breakout,
        "breakout_label": breakout_label,
        "vwap_now": vwap_now,
        "vwap_bias": vwap_bias,
        "r1": r1, "r2": r2, "s1": s1, "s2": s2, "pp": pp,
        "mc": mc, "per": per, "pbv": pbv, "roe": roe, "de": de,
        "norm_signals": norm_signals,   # untuk simpan prediksi
        "ringkasan": ringkasan,
        "is_daytrade": is_daytrade,
        "mode": "daytrade" if is_daytrade else "swing",
        "actual_interval": actual_interval,
        "harga_terakhir_asli": harga_terakhir_asli,
        "floating_pl_pct": floating_pl_pct,
        "harga_beli_float": harga_beli_float,
        "sudah_beli": sudah_beli,
        "ticker_raw": ticker_raw
    }
        # Tambahan untuk UI
    result["ticker_info"] = ticker_info
    result["adx_threshold"] = adx_threshold
    result["hit_tp"] = hit_tp
    result["hit_sl"] = hit_sl
    result["estimasi_label"] = estimasi_label
    result["prob_label"] = prob_label
    result["backtest_window"] = backtest_window
    result["ofi_now"] = df['OFI_raw'].iloc[-1]
    result["adaptive_w"] = adaptive_w
    result["returns"] = returns
    result["mom_median_th"] = mom_median_th
    result["coppock_rising"] = coppock_rising
    result["coppock_turning_up"] = coppock_turning_up
    result["coppock_status"] = coppock_status
    result["avg_sentiment"] = avg_sentiment
    result["norm_signals"] = norm_signals
    result["entry_low_f"] = entry_low_f
    result["entry_high_f"] = entry_high_f
    result["ticker_raw"] = ticker_raw
    result["harga_terakhir"] = harga_terakhir
    result["floating_pl_pct"] = floating_pl_pct
    result["sudah_beli"] = sudah_beli
    result["harga_beli_float"] = harga_beli_float
    result["df_est"] = df_est
    return result
def display_analysis_result(res):
    # ===== AMBIL SEMUA VARIABEL DARI RES =====
    df = res['df']
    df_back = res['df_back']
    harga_terakhir = res['harga_terakhir']
    signal = res['signal']
    entry_zone_f = res['entry_zone_f']
    sl_harga_f = res['sl_harga_f']
    tp_low_f = res['tp_low_f']
    tp_high_f = res['tp_high_f']
    rrr = res['rrr']
    rrr_status = res['rrr_status']
    prob_bull = res['prob_bull']
    est_besok_f = res['est_besok_f']
    est_besok_sinyal_f = res['est_besok_sinyal_f']
    low_est_f = res['low_est_f']
    up_est_f = res['up_est_f']
    tp_pct_low = res['tp_pct_low']
    tp_pct_high = res['tp_pct_high']
    sl_pct = res['sl_pct']
    adx = res['adx']
    rsi14 = res['rsi14']
    atr_pct = res['atr_pct']
    avg_sentiment = res['avg_sentiment']
    sentimen_status = res['sentimen_status']
    headlines = res['headlines']
    sources = res['sources']
    translated = res['translated']
    regime = res['regime']
    ihsg_cond = res['ihsg_cond']
    coppock_val = res['coppock_val']
    coppock_prev = res['coppock_prev']
    coppock_turning_up = res['coppock_turning_up']
    coppock_rising = res['coppock_rising']
    coppock_status = res['coppock_status']
    beta_ihsg = res['beta_ihsg']
    win_bt = res['win_bt']
    pf_bt = res['pf_bt']
    avg_bt = res['avg_bt']
    max_dd_bt = res['max_dd_bt']
    sharpe_bt = res['sharpe_bt']
    trades_bt = res['trades_bt']
    kelly_adj = res['kelly_adj']
    max_dd = res['max_dd']
    max_dd_30 = res['max_dd_30']
    breakout = res['breakout']
    breakout_label = res['breakout_label']
    vwap_now = res['vwap_now']
    vwap_bias = res['vwap_bias']
    r1 = res['r1']
    r2 = res['r2']
    s1 = res['s1']
    s2 = res['s2']
    pp = res['pp']
    mc = res['mc']
    per = res['per']
    pbv = res['pbv']
    roe = res['roe']
    de = res['de']
    ticker_info = res['ticker_info']
    adx_threshold = res['adx_threshold']
    hit_tp = res['hit_tp']
    hit_sl = res['hit_sl']
    estimasi_label = res['estimasi_label']
    prob_label = res['prob_label']
    backtest_window = res['backtest_window']
    ofi_now = res['ofi_now']
    is_daytrade = res['is_daytrade']
    floating_pl_pct = res['floating_pl_pct']
    sudah_beli = res['sudah_beli']
    ticker_raw = res['ticker_raw']

    # === Tambahan untuk V12 Adaptive & AI Insight ===
    adaptive_w = res['adaptive_w']
    returns = res['returns']
    mom_median_th = res['mom_median_th']
    harga_beli_float = res['harga_beli_float']

    # ===== TAMPILAN UTAMA =====
    st.title("📊 Quant & Risk Engine Pro")
    st.write("Algoritma kuantitatif + Berita + Backtest + AI + Grafik Interaktif + Fundamental")
    st.success(f"✅ Analisis Berhasil: {ticker_raw} | Closing Price: Rp {harga_terakhir:,.0f}".replace(",", "."))

    now_jkt = datetime.now(pytz.timezone("Asia/Jakarta"))
    st.caption(f"⏱️ **Waktu Analisis:** {now_jkt.strftime('%d %B %Y, %H:%M:%S WIB')}")
    waktu_str = now_jkt.strftime('%d %B %Y, %H:%M WIB')

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Sinyal Eksekusi",
        signal,
        delta=f"per {now_jkt.strftime('%d/%m %H:%M')} WIB",
        delta_color="off"
    )
    with col2:
        st.metric(
            label=f"{estimasi_label} (Netral)",
            value=f"Rp {est_besok_f:,.0f}",
            delta=f"Range: Rp {low_est_f:,.0f} - {up_est_f:,.0f}"
        )
        st.metric(
            label=f"{estimasi_label} (Sinyal {signal.split()[0]})",
            value=f"Rp {est_besok_sinyal_f:,.0f}",
            delta=f"{((est_besok_sinyal_f - harga_terakhir) / harga_terakhir * 100):+.2f}%"
        )
    col3.metric(prob_label, f"{prob_bull:.1f}%")

    # ===== GRAFIK =====
    if PLOTLY_AVAILABLE:
        st.header("📈 Chart Harga & Sinyal")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#00ffcc')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', line=dict(color='#f59e0b', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50', line=dict(color='#ef4444', dash='dot')))
        buy_signals = df_back[df_back['Signal'].str.contains("BUY")]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers',
                                 marker=dict(symbol='triangle-up', size=10, color='#10b981'), name='Buy Signal'))
        for lvl, lbl, clr in [(r1, 'R1', 'orange'), (s1, 'S1', 'red'), (pp, 'PP', 'gray')]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=clr, annotation_text=lbl, annotation_position="right")
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=20, b=10), dragmode='pan')
        st.plotly_chart(fig, use_container_width=True)

    # ===== RINGKASAN EKSEKUTIF =====
    st.markdown("---")
    st.header("📋 Ringkasan Eksekutif & Rekomendasi")

    if rrr < 1.0 and ("BUY" in signal):
        ac, ai = "#ef4444", "⚠️"
        at = f"• <b>KONDISI:</b> Tren Valid, RRR {rrr:.2f} ({rrr_status})<br>• <b>REKOMENDASI:</b> BUY ON WEAKNESS<br>• <b>LANGKAH:</b> Entry di zona {entry_zone_f}, SL {sl_harga_f:,.0f}, TP bertahap {tp_low_f:,.0f} - {tp_high_f:,.0f}."
    elif "STRONG BUY" in signal:
        ac, ai = "#10b981", "🟢"
        at = f"• <b>KONDISI:</b> Tren Kuat & Akumulasi Volume<br>• <b>REKOMENDASI:</b> AGGRESSIVE BUY<br>• <b>LANGKAH:</b> Entry di zona {entry_zone_f}, SL {sl_harga_f:,.0f} (-{sl_pct:.1f}%), TP bertahap {tp_low_f:,.0f} - {tp_high_f:,.0f}."
    elif "BUY" in signal:
        ac, ai = "#f59e0b", "🟡"
        at = f"• <b>KONDISI:</b> Tren Valid, RRR {rrr:.2f} ({rrr_status})<br>• <b>REKOMENDASI:</b> BUY ON WEAKNESS<br>• <b>LANGKAH:</b> Entry di zona {entry_zone_f}, SL {sl_harga_f:,.0f}, TP bertahap {tp_low_f:,.0f} - {tp_high_f:,.0f}."
    elif "HOLD" in signal:
        ac, ai = "#3b82f6", "🔵"
        at = f"• <b>KONDISI:</b> Konsolidasi / Transisi<br>• <b>REKOMENDASI:</b> HOLD<br>• <b>LANGKAH:</b> Jangan tambah posisi, pantau SL."
    else:
        ac, ai = "#ef4444", "🔴"
        at = f"• <b>KONDISI:</b> Risiko Penurunan / Distribusi<br>• <b>REKOMENDASI:</b> AVOID / LIQUIDATE<br>• <b>LANGKAH:</b> Amankan modal."

    # Tambahan untuk status kepemilikan
    if sudah_beli:
        if "AVOID" in signal:
            extra = "⚠️ Karena kamu sudah memegang saham ini, pertimbangkan untuk <b>take profit sebagian</b> atau <b>keluar seluruhnya</b> untuk mengamankan modal."
        elif "HOLD" in signal:
            extra = "🔒 Kamu sudah punya posisi. Disarankan <b>tahan</b> dan pasang <b>trailing stop</b> di bawah support terdekat."
        elif "STRONG BUY" in signal or "BUY" in signal:
            extra = "✅ Posisi sudah ada. Tidak perlu menambah agresif. Jika ingin averaging, tunggu harga menyentuh <b>entry zone</b>."
        else:
            extra = ""

        if floating_pl_pct is not None:
            pl_str = f"💰 <b>Floating P/L:</b> {floating_pl_pct:+.2f}%"
            if floating_pl_pct > 5:
                extra += f" (Profit sudah >5%. Pertimbangkan <b>take profit sebagian</b> atau <b>trailing stop</b>.)"
            elif floating_pl_pct > 0:
                extra += f" (Masih profit. Pantau SL ketat.)"
            elif floating_pl_pct < -3:
                extra += f" (Rugi >3%. Jika menembus SL, keluar.)"
            else:
                extra += f" (Rugi kecil. Tahan dengan SL sesuai rekomendasi.)"
            extra = pl_str + " " + extra

        if extra:
            at += f"<br><br><b>📌 Status Posisi:</b> {extra}"

    col1, col2 = st.columns([1, 1])
    with col1:
        if "AVOID" not in signal:
            st.markdown(f'''
                <div class="summary-card">
                    <div class="summary-item">🕒 <b>Waktu Scan:</b> {waktu_str}</div>
                    <div class="section-title">📌 Profil Risiko (Kontekstual)</div>
                    <div class="summary-item">🛡️ <b>Stop Loss:</b> Rp {sl_harga_f:,.0f} (-{sl_pct:.1f}%)</div>
                    <div class="summary-item">🎯 <b>Take Profit Range:</b> Rp {tp_low_f:,.0f} - Rp {tp_high_f:,.0f}<br>
                        <span style="font-size:13px;color:#8892b0;">(+{tp_pct_low:.1f}% ~ +{tp_pct_high:.1f}%)</span></div>
                    <div class="summary-item">⚖️ <b>Risk:Reward (min):</b> 1 : {rrr:.2f} ({rrr_status})</div>
                    <div class="summary-item" style="color:#8892b0;">📊 ADX {adx:.1f} | RSI {rsi14:.1f} | ATR {atr_pct:.2f}%</div>
                    <div class="summary-item">🏷️ <b>Rezim:</b> {regime} | {ihsg_cond}</div>
                    <div class="summary-item">🛡️ <b>Alokasi Maks (Kelly):</b> {kelly_adj*100:.1f}% dari Total Ekuitas</div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="summary-card">
                    <div class="summary-item">🕒 <b>Waktu Scan:</b> {waktu_str}</div>
                    <div class="section-title">⛔ Sinyal AVOID</div>
                    <div class="summary-item">Tidak ada rekomendasi entry untuk saat ini.</div>
                    <div class="summary-item" style="color:#8892b0;">📊 ADX {adx:.1f} | RSI {rsi14:.1f} | ATR {atr_pct:.2f}%</div>
                    <div class="summary-item">🏷️ <b>Rezim:</b> {regime} | {ihsg_cond}</div>
                    <div class="summary-item">🛡️ <b>Alokasi Maks (Kelly):</b> {kelly_adj*100:.1f}% dari Total Ekuitas</div>
                </div>
            ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="action-card" style="border-left-color: {ac};"><div class="section-title">{ai} Panduan Eksekusi Trader</div><div class="summary-item" style="font-size:15px;margin-top:8px;line-height:1.6;">{at}</div><hr style="border-color:#334155;margin:15px 0;"><div style="color:#94a3b8;font-size:13px;">⚠️ <i>Disclaimer: Hasil pengujian berbasis permodelan matematika probabilitas kuantitatif historis. Keputusan akhir eksekusi modal tetap merupakan tanggung jawab penuh masing-masing investor.</i></div></div>', unsafe_allow_html=True)

    # ===== DETAIL EXPANDER =====
    with st.expander("🔍 Lihat Detail Analisis (Berita, Fundamental, Backtest, dll)"):
        st.subheader("📰 Sentimen Berita Terbobot")
        c1, c2 = st.columns([1, 2])
        c1.metric("Sentimen Skor", f"{avg_sentiment:.2f}", sentimen_status)
        with c2:
            st.markdown("**5 Berita Utama Pasar:**")
            for i, h in enumerate(headlines):
                src = sources[i] if i < len(sources) else ""
                t = translated[i] if i < len(translated) else ""
                st.markdown(f"{i+1}. **{h}** <span class='source'>({src})</span>", unsafe_allow_html=True)
                if t and t != h:
                    st.markdown(f"<span class='translated'>🇮🇩 {t}</span>", unsafe_allow_html=True)

        st.divider()
        st.subheader("🧬 Regime Pasar & Volatilitas")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Market Regime", regime)
        m2.metric("Kondisi Makro IHSG", ihsg_cond)
        m3.metric("ADX Adaptif", f"{adx:.1f} (Thresh: {adx_threshold:.1f})")
        m4.metric("OFI Ratio", f"{ofi_now:.2f}")
        if is_daytrade:
            vwap_col = st.columns(1)[0]
            vwap_col.metric("VWAP", f"{vwap_now:,.0f}", vwap_bias)
        st.markdown(f"**Insight Regime:** {generate_regime_insight(regime, adx, ofi_now, ihsg_cond)}")

        st.divider()
        st.subheader("📊 Metrik Fundamental Saham (IDX)")
        if ticker_info:
            def clean_val(v, f="{:.2f}"):
                return "N/A" if v is None else f.format(v)

            def singkat_angka(n):
                if n is None:
                    return "N/A"
                n = float(n)
                if n >= 1e12:
                    return f"{n/1e12:,.1f} T"
                elif n >= 1e9:
                    return f"{n/1e9:,.0f} M"
                else:
                    return f"{n:,.0f}"

            mc_short = singkat_angka(mc)
            table_html = (
                f"<table class='fundamental-table'>"
                f"<tr><td>Market Cap</td><td>{mc_short} IDR</td></tr>"
                f"<tr><td>PER</td><td>{clean_val(per, '{:.2f}x')}</td></tr>"
                f"<tr><td>PBV</td><td>{clean_val(pbv, '{:.2f}x')}</td></tr>"
                f"<tr><td>ROE</td><td>{clean_val(roe*100 if roe else None, '{:.1f}%')}</td></tr>"
                f"<tr><td>D/E</td><td>{clean_val(de, '{:.2f}%')}</td></tr>"
                f"</table>"
            )
            st.markdown(table_html, unsafe_allow_html=True)

            interpretation_items = []
            if mc:
                if mc >= 1e13:
                    mct = f"Market Cap Rp {mc:,.0f} tergolong sangat besar (Mega Cap)."
                elif mc >= 1e12:
                    mct = f"Market Cap Rp {mc:,.0f} tergolong besar (Blue Chip)."
                elif mc >= 1e10:
                    mct = f"Market Cap Rp {mc:,.0f} tergolong menengah (Mid Cap)."
                else:
                    mct = f"Market Cap Rp {mc:,.0f} tergolong kecil (Small Cap)."
            else:
                mct = "Market Cap tidak tersedia."
            interpretation_items.append(f"<li><b>Market Cap:</b> {mct}</li>")

            if per:
                if per < 10:
                    pt = f"PER {per:.2f}x tergolong rendah (potensi undervalue)."
                elif per < 20:
                    pt = f"PER {per:.2f}x moderat."
                else:
                    pt = f"PER {per:.2f}x tergolong tinggi (premium)."
            else:
                pt = "PER tidak tersedia."
            interpretation_items.append(f"<li><b>PER:</b> {pt}</li>")

            if pbv:
                if pbv < 1:
                    pbt = f"PBV {pbv:.2f}x di bawah 1 (di bawah nilai buku, bisa undervalue)."
                elif pbv < 3:
                    pbt = f"PBV {pbv:.2f}x moderat."
                else:
                    pbt = f"PBV {pbv:.2f}x tinggi (premium)."
            else:
                pbt = "PBV tidak tersedia."
            interpretation_items.append(f"<li><b>PBV:</b> {pbt}</li>")

            if roe:
                roep = roe * 100
                if roep > 20:
                    rt = f"ROE {roep:.1f}% sangat baik (profitabilitas tinggi)."
                elif roep > 10:
                    rt = f"ROE {roep:.1f}% cukup baik."
                else:
                    rt = f"ROE {roep:.1f}% rendah."
            else:
                rt = "ROE tidak tersedia."
            interpretation_items.append(f"<li><b>ROE:</b> {rt}</li>")

            if de:
                if de > 1:
                    dt = f"D/E {de:.2f} tinggi (leverage tinggi, risiko lebih besar)."
                elif de > 0.5:
                    dt = f"D/E {de:.2f} moderat."
                else:
                    dt = f"D/E {de:.2f} rendah (konservatif)."
            else:
                dt = "D/E tidak tersedia."
            interpretation_items.append(f"<li><b>D/E:</b> {dt}</li>")

            st.markdown(f'<div style="background-color:#1e293b;border-radius:12px;padding:15px;margin-top:15px;color:#cbd5e1;font-size:14px;"><b style="color:#00ffcc;">📝 Interpretasi Metrik:</b><ul style="margin-top:8px;padding-left:20px;">{"".join(interpretation_items)}</ul></div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data fundamental finansial tidak tersedia.")

        st.divider()
        st.subheader("🎯 Target Pivot & Support/Resistance")
        p1, p2, p3, p4, p5 = st.columns(5)
        r2_f = fraksi_bei(r2)
        r1_f = fraksi_bei(r1)
        pp_f = fraksi_bei(pp)
        s1_f = fraksi_bei(s1)
        s2_f = fraksi_bei(s2)

        p1.metric("R2", f"Rp {r2_f:,.0f}".replace(",", "."))
        p2.metric("R1", f"Rp {r1_f:,.0f}".replace(",", "."))
        p3.metric("Pivot", f"Rp {pp_f:,.0f}".replace(",", "."))
        p4.metric("S1", f"Rp {s1_f:,.0f}".replace(",", "."))
        p5.metric("S2", f"Rp {s2_f:,.0f}".replace(",", "."))
        st.write(f"Kondisi {breakout_label}: **{breakout}**")

        st.divider()
        st.subheader("🔮 Sinyal Kuantitatif & Hasil Backtest" + (" (Intraday)" if is_daytrade else " (6 Bulan)"))
        if "AVOID" not in signal:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Sinyal", signal)
            c2.metric(estimasi_label, f"Rp {est_besok_f:,.0f}".replace(",", "."))
            c3.metric("Entry Zone", entry_zone_f)
            c4.metric("TP Range", f"Rp {tp_low_f:,.0f} - Rp {tp_high_f:,.0f}",
                      f"+{tp_pct_low:.1f}% ~ +{tp_pct_high:.1f}%")
            c5.metric("Stop Loss", f"Rp {sl_harga_f:,.0f}", f"-{sl_pct:.1f}%")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Sinyal", signal)
            c2.metric(estimasi_label, f"Rp {est_besok_f:,.0f}".replace(",", "."))
            st.info("⛔ Tidak ada rekomendasi entry, TP, atau SL untuk sinyal AVOID.")

        st.markdown(f"**Hasil Backtest ({backtest_window} Bar):**")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Win Rate", f"{win_bt:.1%}" if trades_bt else "N/A")
        b2.metric("Profit Factor", f"{pf_bt:.2f}" if trades_bt and pf_bt != np.inf else "N/A")
        b3.metric("Avg Return/Trade", f"{avg_bt:.2%}" if trades_bt else "N/A")
        b4.metric("Max DD Strat", f"{max_dd_bt:.2f}%" if trades_bt else "N/A")
        b5.metric("Sharpe", f"{sharpe_bt:.2f}" if trades_bt else "N/A")
        b6.metric("Total Trades", trades_bt)

        st.divider()
        st.subheader("🛡️ Manajemen Risiko Portofolio (Kelly)")
        rc1, rc2 = st.columns(2)
        rc1.metric("Alokasi Maks (Kelly)", f"{kelly_adj*100:.1f}%")
        rc2.metric("Beta IHSG", f"{beta_ihsg:.2f}x")
        st.markdown(f"**Interpretasi:** Berdasarkan Win Rate **{win_bt:.1%}**, maksimal alokasi **{kelly_adj*100:.1f}%** dari total ekuitas.")
        st.markdown(f"Max DD Historis: `{max_dd:.2f}%` | DD 30 Hari: `{max_dd_30:.2f}%`")

        st.divider()
        st.subheader("🎲 Simulasi Monte Carlo Ornstein-Uhlenbeck")
        pr1, pr2, pr3 = st.columns(3)
        pr1.metric(prob_label, f"{prob_bull:.1f}%")
        pr2.metric("Prob. Sentuh R1 (30H)", f"{hit_tp:.1f}%")
        pr3.metric("Prob. Sentuh S2 (30H)", f"{hit_sl:.1f}%")

    # ══════════════════════════════════════════════════════════
    # V12 ADAPTIVE ENGINE – EXPANDER & LOGIC (DENGAN INSIGHT)
    # ══════════════════════════════════════════════════════════
    with st.expander("🧬 V12 Adaptive Engine (Coppock, Self‑Learning)", expanded=True):
        st.info(
            "⚙️ **Bagian ini adalah otak adaptif dari QuantRisk Pro.** "
            "Engine secara otomatis mempelajari akurasi setiap faktor teknikal berdasarkan riwayat analisis kamu. "
            "Semakin sering suatu ticker dianalisis, semakin akurat bobot yang dihasilkan."
        )

        if not is_daytrade:
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
                st.metric("Beta IHSG", f"{beta_ihsg:.2f}x", help="Beta > 1 : lebih volatile dari IHSG, Beta < 1 : lebih stabil.")
                st.caption(beta_insight)
        else:
            st.markdown("### 📈 Beta IHSG")
            if beta_ihsg > 1.2:
                beta_insight = f"⚠️ **Beta Tinggi ({beta_ihsg:.2f})** – Saham lebih volatile dari IHSG. Cocok untuk *trading agresif*, namun risikonya lebih besar saat pasar turun."
            elif beta_ihsg > 0.8:
                beta_insight = f"✅ **Beta Moderat ({beta_ihsg:.2f})** – Pergerakan selaras dengan IHSG. Cocok untuk *swing trading*."
            else:
                beta_insight = f"🛡️ **Beta Rendah ({beta_ihsg:.2f})** – Saham defensif, lebih stabil dari IHSG. Cocok untuk *investasi jangka panjang*."
            st.metric("Beta IHSG", f"{beta_ihsg:.2f}x", help="Beta > 1 : lebih volatile dari IHSG, Beta < 1 : lebih stabil.")
            st.caption(beta_insight)
            st.info("ℹ️ Coppock Curve tidak ditampilkan untuk Day Trade karena kurang relevan dengan timeframe intraday.")

        st.markdown("### ⚖️ Bobot Adaptif per Faktor")
        st.caption(
            "Bobot di bawah dihitung otomatis berdasarkan **akurasi historis** masing‑masing faktor. "
            "Faktor yang sering benar mendapat bobot lebih tinggi. Bobot ini digunakan untuk sinyal akhir."
        )

        if is_daytrade:
            display_adaptive_w = {k: v for k, v in adaptive_w.items() if k != "Coppock"}
            st.caption("ℹ️ Faktor **Coppock** tidak ditampilkan dalam bobot adaptif untuk Day Trade karena kurang relevan secara intraday. "
                       "Namun, data-nya tetap dihitung di background untuk menjaga konsistensi historis.")
        else:
            display_adaptive_w = adaptive_w

        w_df = pd.DataFrame.from_dict(display_adaptive_w, orient='index', columns=['Weight'])
        st.bar_chart(w_df)

        if display_adaptive_w:
            max_factor = max(display_adaptive_w, key=display_adaptive_w.get)
            min_factor = min(display_adaptive_w, key=display_adaptive_w.get)
            max_weight = display_adaptive_w[max_factor]
            min_weight = display_adaptive_w[min_factor]

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
            keys_to_show = [k for k in FACTOR_KEYS if not (is_daytrade and k == "Coppock")]
            acc_data = {k: mem.get('accuracy', {}).get(k, 0.5) for k in keys_to_show}
            err_data = {k: mem.get('error_ema', {}).get(k, 1.0) for k in keys_to_show}

            col_a, col_e = st.columns(2)
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

            entry_err = mem.get('entry_error_ema', 0.0)
            if entry_err > 0:
                st.caption(
                    f"🎯 **Rata‑rata error entry:** {entry_err:.2f} poin. "
                    "Entry sering tidak tersentuh, engine akan menggeser zona entry lebih dekat ke harga."
                )
            else:
                st.caption("🎯 **Error entry:** 0 — entry zone sudah cukup baik atau belum ada data Not Touched.")
        else:
            st.info("Belum ada data memori untuk ticker ini. Lakukan analisis beberapa kali agar engine mulai belajar.")

        st.markdown("### 🔁 Proses Self‑Learning")
        st.caption(
            "Setiap analisis, engine membandingkan prediksi sebelumnya dengan harga aktual. "
            "Jika benar → akurasi naik. Jika salah → error bertambah. Bobot otomatis menyesuaikan. "
            "Selain itu, engine juga mempelajari **level entry** dari kejadian Entry Tidak Tersentuh."
        )

        last_pred = load_v12_predictions(ticker_raw, mode='daytrade' if is_daytrade else 'swing')
        if last_pred:
            last_close = safe_float(last_pred.get('close_price'), 0.0)
            if last_close > 0:
                last_signals = {}
                for k in FACTOR_KEYS:
                    key = f'sig_{k}'
                    if key in last_pred:
                        last_signals[k] = safe_float(last_pred[key], 0.0)
                    else:
                        last_signals[k] = 0.0

                actual_return = (harga_terakhir - last_close) / last_close
                volatility = returns.std()
                update_v12_memory(ticker_raw, last_signals, actual_return, volatility)
                st.success(f"✅ Memory updated! Actual return sejak prediksi terakhir: {actual_return*100:.2f}%")
            else:
                st.info("ℹ️ Prediksi sebelumnya tidak memiliki close_price yang valid.")
        else:
            st.info("ℹ️ Tidak ada prediksi sebelumnya. Engine akan mulai belajar pada analisis berikutnya.")

    # ==================== AI INSIGHT OTOMATIS ====================
    st.markdown("---")
    if st.session_state.get("gemini_api_key"):
        with st.spinner("🧠 AI sedang menganalisis hasil dan riwayat..."):
            data_ai = {
                "Saham": ticker_raw,
                "Harga": f"{harga_terakhir:,.0f}",
                "Sinyal": signal,
                "Rezim": regime,
                "Sentimen": f"{avg_sentiment:.2f} ({sentimen_status})",
                "RRR": f"{rrr:.2f} (Kontekstual)",
                "Prob Naik": f"{prob_bull:.1f}%",
                "TP%": f"{tp_pct_low:.1f}% - {tp_pct_high:.1f}%",
                "SL%": f"{sl_pct:.1f}",
                "Estimasi": f"{est_besok_f:,.0f}",
                "Beta": f"{beta_ihsg:.2f}x",
                "WinRate": f"{win_bt:.1%}" if trades_bt else "N/A",
                "ProfitFactor": f"{pf_bt:.2f}" if trades_bt else "N/A",
                "MaxDD": f"{max_dd_bt:.2f}%" if trades_bt else "N/A",
                "Kelly": f"{kelly_adj*100:.1f}",
                "Fundamental_MC": f"{mc:,.0f}" if mc else "N/A",
                "Fundamental_PER": f"{per:.2f}" if per else "N/A",
                "Fundamental_PBV": f"{pbv:.2f}" if pbv else "N/A",
                "Fundamental_ROE": f"{roe*100:.1f}" if roe else "N/A",
                "Fundamental_DE": f"{de:.2f}" if de else "N/A",
                "Status_Posisi": "Sudah memiliki saham" if sudah_beli else "Belum memiliki saham",
                "Harga_Beli": f"Rp {harga_beli_float:,.0f}" if harga_beli_float else "Tidak diisi",
                "Floating_PL": f"{floating_pl_pct:+.2f}%" if floating_pl_pct is not None else "N/A"
            }
            riwayat_konteks = []
            for r in st.session_state.riwayat:
                if r['Saham'] == ticker_raw:
                    r_copy = dict(r)
                    # Ambil mode/gaya dari baris riwayat
                    mode_actual = r.get('Gaya', 'SW')   # "SW" atau "DT"

                    # Coba key 3 elemen (format baru)
                    key_actual_baru = (r.get('Waktu'), r.get('Saham'), mode_actual)
                    actual = st.session_state.riwayat_actual.get(key_actual_baru)

                    # Fallback ke key 2 elemen (format lama)
                    if actual is None:
                        key_actual_lama = (r.get('Waktu'), r.get('Saham'))
                        actual = st.session_state.riwayat_actual.get(key_actual_lama, {})

                    if actual:
                        r_copy['Actual_High']   = actual.get('Actual_High', '')
                        r_copy['Actual_Low']    = actual.get('Actual_Low', '')
                        r_copy['Actual_Close']  = actual.get('Actual_Close', '')
                        r_copy['Actual_Outcome']= actual.get('Outcome', '')
                        r_copy['Entry_Miss']    = actual.get('Entry_Miss', '')

                    riwayat_konteks.append(r_copy)
                    if len(riwayat_konteks) >= 20:
                        break

            hasil_ai, error_ai = analisis_saham_dengan_ai(data_ai, riwayat_konteks, st.session_state.gemini_api_key)
            if not error_ai and hasil_ai:
                hasil_ai_bersih = bersihkan_teks_ai(hasil_ai)
                html_ai = f'<div class="ai-insight-card"><h3>🤖 Insight AI</h3><p>{hasil_ai_bersih}</p></div>'
                st.markdown(html_ai, unsafe_allow_html=True)
            elif error_ai:
                st.warning(f"AI tidak dapat memberikan insight: {error_ai}")
    else:
        st.info("💡 Isi API Key Gemini di sidebar untuk mendapatkan insight AI otomatis.")
# ==================== PROSES ANALISIS ====================
if run_btn:
    if not ticker_input:
        st.warning("⚠️ Kode saham tidak boleh kosong!")
        st.stop()

    with st.spinner("🤖 Menganalisis mode Swing dan Daytrade secara paralel..."):
        from concurrent.futures import ThreadPoolExecutor
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        except ImportError:
            try:
                from streamlit.scriptrunner import add_script_run_ctx, get_script_run_ctx
            except ImportError:
                add_script_run_ctx = None
                get_script_run_ctx = None

        ctx = get_script_run_ctx() if get_script_run_ctx is not None else None
        v12_mem_snapshot = dict(st.session_state.v12_memory) if "v12_memory" in st.session_state else {}

        def run_analysis_task(func, *args, **kwargs):
            if add_script_run_ctx and ctx:
                add_script_run_ctx(ctx=ctx)
            return func(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_swing = executor.submit(
                run_analysis_task,
                analyze_stock,
                ticker_input, harga_manual, sudah_beli, harga_beli_float,
                False, v12_mem_snapshot, fee_beli_pct, fee_jual_pct  # swing
            )
            future_day = executor.submit(
                run_analysis_task,
                analyze_stock,
                ticker_input, harga_manual, sudah_beli, harga_beli_float,
                True, v12_mem_snapshot, fee_beli_pct, fee_jual_pct   # daytrade
            )
            res_swing = future_swing.result()
            res_day = future_day.result()

    if res_swing is None or res_day is None:
        st.error("❌ Gagal mengambil data untuk salah satu mode.")
        st.stop()

    # ----- REKOMENDASI MODE -----
    def skor_mode(res):
        return res['signal_score'] * 0.5 + min(res['rrr'], 5) * 0.2 + res['confidence'] * 0.3

    skor_swing = skor_mode(res_swing)
    skor_day = skor_mode(res_day)

    if skor_swing >= skor_day:
        mode_terbaik = "Swing Trade"
        alasan = "Sinyal swing lebih kuat dan RRR lebih baik."
        res_terbaik = res_swing
    else:
        mode_terbaik = "Day Trade"
        alasan = "Sinyal intraday lebih kuat dan probabilitas naik lebih tinggi."
        res_terbaik = res_day

    st.success(f"🏆 **Rekomendasi Mode: {mode_terbaik}** — {alasan}")

    # ----- TAMPILKAN HASIL KEDUA MODE DALAM TAB -----
    tab_swing, tab_day = st.tabs(["📆 Swing Trade", "⏱️ Day Trade"])

    with tab_swing:
        display_analysis_result(res_swing)

    with tab_day:
        display_analysis_result(res_day)

    # ----- SIMPAN PREDIKSI V12 UNTUK KEDUA MODE -----
    for res in [res_swing, res_day]:
        try:
            save_v12_prediction(
                ticker_raw,
                res['harga_terakhir'],
                res['norm_signals'],   # sudah berupa dict norm_signals
                entry_low=res['entry_low_f'],
                entry_high=res['entry_high_f'],
                mode=res['mode']
            )
        except Exception as e:
            st.warning(f"Gagal menyimpan prediksi {res['mode']}: {e}")

    # ----- SIMPAN RIWAYAT UNTUK KEDUA MODE (SWING & DAYTRADE) -----
    simpan_riwayat([res_swing['ringkasan'], res_day['ringkasan']])

    st.stop()
# ==================== SCANNER SAHAM IDX (V12 TECH SCORE) ====================
if scan_btn:
    st.title("🔍 Scanner Saham IDX (V12 Tech Score)")
    st.write(f"Mode: {mode_scan} | Likuiditas Min: Rp {likuiditas_min:,.0f}/hari")

    with st.spinner("📡 Mengambil daftar saham..."):
        daftar_saham = get_daftar_saham(mode_scan)
        st.info(f"📋 {len(daftar_saham)} saham akan dipindai.")

    # Ambil data IHSG sekali saja untuk semua saham (perlu 6 bulan agar Coppock akurat)
    ihsg_data = load_ihsg_data(period="6mo", interval="1d")
    if ihsg_data.empty:
        st.error("❌ Gagal mengambil data IHSG. Pastikan koneksi internet stabil.")
        st.stop()

    # --- Tombol batal scan (dengan session state) ---
    if "cancel_scan" not in st.session_state:
        st.session_state.cancel_scan = False

    cancel_col, _ = st.columns([1, 5])
    with cancel_col:
        if st.button("⏹️ Batalkan Scan"):
            st.session_state.cancel_scan = True

    progress_bar = st.progress(0)
    status_text = st.empty()
    hasil_scan = []

    # --- Fungsi worker per saham (menggunakan score_stock_tech) ---
    def process_ticker(ticker):
        try:
            ticker_jk = f"{ticker}.JK"
            df = load_stock_data(ticker_jk, period="6mo", interval="1d")
            if df.empty or len(df) < 65:
                return None
            # Filter likuiditas: volume rata2 20 hari * harga terakhir
            if 'Volume' in df.columns and len(df) >= 20:
                avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
                last_price = float(df['Close'].iloc[-1])
                if avg_vol * last_price < likuiditas_min:
                    return None
            # Panggil scoring teknikal (adaptasi Kotlin)
            return score_stock_tech(df, ticker, ihsg_data)
        except:
            return None

    total = len(daftar_saham)
    max_workers = 4  # batasi koneksi paralel agar tidak di-banned Yahoo

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(process_ticker, t): t for t in daftar_saham}
        completed = 0
        for future in as_completed(future_to_ticker):
            if st.session_state.cancel_scan:
                executor.shutdown(wait=False, cancel_futures=True)
                break
            res = future.result()
            if res is not None:
                hasil_scan.append(res)
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"Memindai {future_to_ticker[future]} ({completed}/{total})...")

    if st.session_state.cancel_scan:
        st.warning("Scan dibatalkan oleh pengguna.")
        st.stop()

    progress_bar.empty()
    status_text.empty()

    if not hasil_scan:
        st.warning("Tidak ada saham yang lolos filter likuiditas atau data tidak lengkap.")
        st.stop()

    # Urutkan berdasarkan techScore
    hasil_scan.sort(key=lambda x: x['techScore'], reverse=True)

        # --- Pisahkan Beli dan Jual ---
    buy_signals = [r for r in hasil_scan if r['techScore'] > 0.05]
    sell_signals = [r for r in hasil_scan if r['techScore'] < -0.05]
    # TOP BUY: ambil 10 terkuat (techScore tertinggi)
    top_buys = buy_signals[:10]
    # TOP SELL: ambil 10 terlemah (techScore paling negatif)
    # Urutkan sell_signals dari paling negatif ke kurang negatif
    top_sells = sorted(sell_signals, key=lambda x: x['techScore'])[:10]
    # Simpan hasil scan ke session_state agar tidak hilang saat re-run
    st.session_state.scan_results = {
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'top_buys': top_buys,
        'top_sells': top_sells,
        'total': total,
        'hasil_scan_count': len(hasil_scan),
        'daftar_saham_count': len(daftar_saham)
    }
    st.rerun()
# ==================== TAMPILAN HASIL SCAN (DARI SESSION STATE) ====================
if st.session_state.get('scan_results'):
    sr = st.session_state.scan_results
    buy_signals  = sr['buy_signals']
    sell_signals = sr['sell_signals']
    top_buys     = sr['top_buys']
    top_sells    = sr['top_sells']
    total        = sr['total']
    hasil_scan_count = sr['hasil_scan_count']
    daftar_saham_count = sr['daftar_saham_count']

    st.title("🔍 Scanner Saham IDX (V12 Tech Score)")
    st.write(f"Mode: {mode_scan} | Likuiditas Min: Rp {likuiditas_min:,.0f}/hari")

    st.markdown(f"✅ **Berhasil scan:** {hasil_scan_count}/{daftar_saham_count} saham")
    st.markdown(f"📈 Kandidat Beli: {len(buy_signals)} | 📉 Kandidat Jual: {len(sell_signals)}")

    # ==================== TAMPILAN UTAMA: TOP BUY & TOP SELL ====================
    col_buy, col_sell = st.columns([3, 1])
    with col_buy:
        # Jumlah top_buys bisa berubah (maksimal 10)
        st.subheader(f"🏆 TOP {len(top_buys)} RELATIF TERKUAT - Beli")
    with col_sell:
        # Jumlah top_sells juga adaptif (maksimal 10)
        st.subheader(f"🔻 TOP {len(top_sells)} RELATIF TERLEMAH - Jual")

    # ==================== AI RE-RANK (ditingkatkan) ===================
    if ai_rerank and st.session_state.get("gemini_api_key"):
        with st.spinner("🤖 AI memverifikasi 15 kandidat (1 panggilan batch)..."):
            candidates = buy_signals[:15]
            if not candidates:
                st.info("Tidak ada kandidat Beli untuk diverifikasi AI.")
            else:
                # Kumpulkan berita terbaru untuk setiap kandidat
                headlines_map = {}
                for r in candidates:
                    headlines_map[r['ticker']] = get_headlines_for_ticker(r['ticker'])

                # Prompt dengan berita aktual
                prompt = (
                    "Berikut hasil scan teknikal 15 saham. Verifikasi sinyal BUY dengan sentimen berita TERBARU yang saya berikan untuk setiap saham. "
                    "KELUARKAN HANYA JSON array, TANPA teks pembuka, analisis, atau catatan apapun. "
                    "Format: [{\"ticker\": \"BBRI\", \"confirm\": true, \"confidence_boost\": 0.0-0.15, \"reason\": \"singkat berdasarkan berita\"}]\n\n"
                )
                for r in candidates:
                    tick = r['ticker']
                    headlines = headlines_map.get(tick, ["(tidak ada berita)"])
                    prompt += (
                        f"{tick} | Signal: {r['signal']} | Tech Score: {r['techScore']:.3f} | "
                        f"Coppock: {r['coppockLabel']} | Est Return: {r['muEst']*100:.2f}% | "
                        f"Vol Surge: {r['volSurge']*100:.0f}% | RSI: {r['rsi']:.1f} | "
                        f"Z-Score: {r['zScore']:.2f} | Regime: {r['regime']} | "
                        f"Berita: {'; '.join(headlines)}\n"
                    )

                model, err = dapatkan_model_gemini(st.session_state.gemini_api_key)
                if model and not err:
                    try:
                        response = model.generate_content(prompt)
                        raw = response.text.strip()

                        start_idx = raw.rfind('[')
                        ai_data = []
                        if start_idx != -1:
                            json_str = raw[start_idx:].strip()
                            if json_str.startswith("```json"):
                                json_str = json_str[7:]
                            if json_str.endswith("```"):
                                json_str = json_str[:-3]
                            try:
                                ai_data = json.loads(json_str)
                            except json.JSONDecodeError:
                                st.error("Gagal parse JSON dari akhir respons. Menampilkan debug...")
                                with st.expander("🔎 Debug: Raw Response"):
                                    st.code(raw)
                        else:
                            st.error("Tidak ditemukan array JSON dalam respons AI.")
                            with st.expander("🔎 Debug: Raw Response"):
                                st.code(raw)
                            ai_data = []

                        ai_confirmed = 0
                        ai_upgraded = 0
                        for item in ai_data:
                            ticker = item.get("ticker", "").upper()
                            for r in candidates:
                                if r['ticker'] == ticker:
                                    r['ai_confirm'] = item.get('confirm', False)
                                    r['ai_reason'] = item.get('reason', '')
                                    boost = item.get('confidence_boost', 0.0)
                                    r['ai_boost'] = boost
                                    r['hybrid_score'] = r['techScore'] + boost
                                    if r['ai_confirm']:
                                        ai_confirmed += 1
                                        if boost > 0.01:
                                            ai_upgraded += 1
                                    break

                        msg = f"🤖 AI Re‑Rank selesai: **{ai_confirmed}** saham dikonfirmasi"
                        if ai_upgraded > 0:
                            msg += f", **{ai_upgraded}** naik peringkat karena AI"
                        st.success(msg)
                        # Urutkan ulang buy_signals berdasarkan hybrid_score
                        if candidates:
                            for r in candidates:
                                if 'hybrid_score' not in r:
                                    r['hybrid_score'] = r['techScore']
                            buy_signals.sort(key=lambda x: x.get('hybrid_score', x['techScore']), reverse=True)
                            top_buys = buy_signals[:10]
                            # Update session_state agar render menggunakan urutan baru
                            st.session_state.scan_results['buy_signals'] = buy_signals
                            st.session_state.scan_results['top_buys'] = top_buys
                        with st.expander("📋 Lihat Detail AI Re‑Rank"):
                            ai_table = []
                            for r in candidates:
                                if r.get('ai_confirm') is not None:
                                    ai_table.append({
                                        "Ticker": r['ticker'],
                                        "Tech Score": f"{r['techScore']:.3f}",
                                        "Hybrid Score": f"{r.get('hybrid_score', r['techScore']):.3f}",
                                        "AI Boost": f"{r.get('ai_boost', 0):.3f}",
                                        "AI Confirm": "✅" if r['ai_confirm'] else "❌",
                                        "Reason": r.get('ai_reason', '')
                                    })
                            if ai_table:
                                df_ai = pd.DataFrame(ai_table)
                                df_ai.index = range(1, len(df_ai) + 1)
                                st.dataframe(df_ai, use_container_width=True)
                    except Exception as e:
                        st.error(f"Gagal memproses respons AI: {e}")
                else:
                    st.error("Gagal mengakses Gemini untuk AI Re‑Rank.")
    elif ai_rerank:
        st.info("Isi API Key Gemini di sidebar untuk mengaktifkan AI Re‑Rank.")

    # ---------- Kartu BUY (mirip UI Kotlin) ----------
    for idx, r in enumerate(top_buys):
        rank = idx + 1
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                badge = ["🥇", "🥈", "🥉"][idx] if idx < 3 else f"#{rank}"
                st.markdown(f"### {badge} {r['ticker']}  —  **{r['signal']}**")
            with col2:
                st.metric("Harga", f"Rp {r['lastPrice']:,.0f}")

            # Bar skor + badge konfirmasi AI (jika ada)
            bar_len = int(abs(r['techScore']) * 10)
            bar_str = "█" * bar_len + "░" * (10 - bar_len)
            score_text = f"Tech Score: **{r['techScore']:.3f}**  {bar_str}"
            if r.get('ai_confirm'):
                score_text += f"  |  Hybrid: **{r.get('hybrid_score', r['techScore']):.3f}**"
                st.markdown("☑️ **Dikonfirmasi Tech + Scanner AI Re‑Rank**")
            st.caption(score_text)

            with st.expander("🔎 Detail Indikator"):
                # Baris 1
                ca, cb, cc = st.columns(3)
                ca.metric("Coppock", r['coppockLabel'])
                cb.metric("Est. Return", f"{r['muEst']*100:.2f}%")
                cc.metric("Regime", r['regime'])
                # Baris 2
                ca.metric("Confidence", f"{r['confidence']*100:.0f}%")
                cb.metric("Risk‑Adj (RRR)", f"{r['rrr']:.2f}")
                cc.metric("Likuiditas", r['likuiditas'])
                # Baris 3
                ca.metric("Est. TP Besok", f"Rp {r['tpEst']:,.0f}")
                cb.metric("Est. SL Besok", f"Rp {r['slEst']:,.0f}")
                cc.metric("Zona Entry", f"Rp {r['entryLow']:,.0f}-{r['entryHigh']:,.0f}")
                # Baris 4
                ca.metric("RSI-14", f"{r['rsi']:.1f}")
                cb.metric("Volume Surge", f"{r['volSurge']*100:.0f}%")
                cc.metric("Z‑Score", f"{r['zScore']:.2f}σ")
                # Baris 5
                ca.metric("Bollinger %B", f"{r['bbPct']:.2f}")
                cb.metric("Trend Consistency", f"{r['trendConsistency']:.0f}%")
                cc.metric("Beta | Mom", f"{r['beta']:.2f}β | {r['momScore']*100:.2f}%")
                if r['isCoppockTurningUp']:
                    st.info("⚡ **Coppock Turning Up** — Sinyal akumulasi terkuat!")
                if r.get('ai_reason'):
                    st.caption(f"🧠 AI Reason: {r['ai_reason']}")
            st.divider()

    # ---------- Kartu SELL (ringkas) ----------
    if top_sells:
        with st.container():
            for idx, r in enumerate(top_sells):
                rank = idx + 1
                st.markdown(f"#{rank} **{r['ticker']}** — {r['signal']} | Harga: Rp {r['lastPrice']:,.0f}")
                st.caption(f"Tech Score: {r['techScore']:.3f} | Est Return: {r['muEst']*100:.2f}% | Regime: {r['regime']}")
                st.divider()
    else:
        st.caption("(Tidak ada kandidat Jual yang memenuhi threshold)")

    # ==================== PERKUAT CROSS‑CHECK DENGAN AI (OPSIONAL) ====================
    if buy_signals:   # hanya tampil jika ada kandidat Beli
        st.markdown("---")
        reinforce_col, _ = st.columns([1, 3])
        with reinforce_col:
            if st.button("🛡️ Perkuat Cross‑Check dgn Sentimen AI (tambahan)", key="reinforce_ai"):
                if not st.session_state.get("gemini_api_key"):
                    st.error("API Key Gemini diperlukan.")
                else:
                    with st.spinner("🧠 Mengambil berita terbaru & menganalisis sentimen..."):
                        # Gunakan top_buys yang sudah tampil (maksimal 10)
                        candidates = sr.get('top_buys', [])
                        top_sell_candidates = sr.get('top_sells', [])
                        if not candidates and not top_sell_candidates:
                            st.warning("Tidak ada kandidat untuk dianalisis.")
                            st.stop()
                        # Ambil headline
                        headlines_map = {}
                        for r in candidates:
                            headlines_map[r['ticker']] = get_headlines_for_ticker(r['ticker'])

                        prompt = (
                            "Berikut hasil scan teknikal 15 saham. Verifikasi sinyal BUY dengan sentimen berita TERBARU yang saya berikan. "
                            "KELUARKAN HANYA JSON array, TANPA teks lain. "
                            "Format: [{\"ticker\": \"BBRI\", \"sentiment_score\": 0.0 (skala -1..1), \"note\": \"singkat berdasarkan berita\"}]\n\n"
                        )
                        for r in candidates:
                            tick = r['ticker']
                            headlines = headlines_map.get(tick, ["(tidak ada berita)"])
                            prompt += f"{tick} | Tech Score: {r['techScore']:.3f} | Est Return: {r['muEst']*100:.2f}% | Berita: {'; '.join(headlines)}\n"

                        model, err = dapatkan_model_gemini(st.session_state.gemini_api_key)
                        if model and not err:
                            try:
                                response = model.generate_content(prompt)
                                raw = response.text.strip()
                                start_idx = raw.rfind('[')
                                sentiments = []
                                if start_idx != -1:
                                    json_str = raw[start_idx:].strip()
                                    if json_str.startswith("```json"): json_str = json_str[7:]
                                    if json_str.endswith("```"): json_str = json_str[:-3]
                                    try:
                                        sentiments = json.loads(json_str)
                                    except json.JSONDecodeError:
                                        st.error("Gagal parse JSON dari akhir respons.")
                                        with st.expander("🔎 Debug: Raw Response"):
                                            st.code(raw)
                                else:
                                    st.error("Tidak ditemukan array JSON dalam respons AI.")
                                    with st.expander("🔎 Debug: Raw Response"):
                                        st.code(raw)

                                # --- Tampilkan hasil cross‑check ---
                                st.success("✅ Cross‑Check Sentimen AI berhasil!")
                                st.markdown("### 🧠 AI-Enhanced Cross-Check")
                                st.caption(
                                    "Quick Technical Cross-Check (7 faktor) + sentimen berita AI "
                                    "(berita terbaru dari Google News) "
                                    "= 8 dari 9 faktor Single Quant. "
                                    "MASIH bukan Single Quant penuh. Broker Summary tetap perlu input manual per-saham."
                                )

                                st.markdown("**TOP BELI**")
                                for item in sentiments:
                                    ticker = item.get("ticker", "").upper()
                                    stock = next((r for r in candidates if r['ticker'] == ticker), None)
                                    if stock is None:
                                        continue
                                    est_return = stock['muEst'] * 100
                                    sent_score = item.get("sentiment_score", 0.0)
                                    al_enhanced = sent_score * 100
                                    status = "☑️ Sejalan" if sent_score > 0 else "⛔ Berlawanan"
                                    note = item.get("note", "")
                                    st.markdown(f"""
                                    **{ticker}** {status}  
                                    Scanner: Beli (Est. Return {est_return:.2f}%) vs AI-Enhanced: {al_enhanced:.2f}% sentimen: {note}
                                    """)

                                # --- TOP JUAL (dengan berita juga) ---
                                sell_signals_list = sr.get('sell_signals', [])
                                if top_sell_candidates:
                                    st.markdown("**TOP JUAL**")
                                    headlines_sell = {}
                                    for r in top_sell_candidates:
                                        headlines_sell[r['ticker']] = get_headlines_for_ticker(r['ticker'])

                                    sell_prompt = (
                                        "Berikut hasil scan teknikal 3 saham dengan sinyal JUAL. "
                                        "Verifikasi sentimen berita TERBARU yang saya berikan. "
                                        "KELUARKAN HANYA JSON array: [{\"ticker\": \"BBRI\", \"sentiment_score\": -0.5..0.5, \"note\": \"singkat\"}]\n\n"
                                    )
                                    for r in top_sell_candidates:
                                        tick = r['ticker']
                                        headlines = headlines_sell.get(tick, ["(tidak ada berita)"])
                                        sell_prompt += f"{tick} | Tech Score: {r['techScore']:.3f} | Est Return: {r['muEst']*100:.2f}% | Berita: {'; '.join(headlines)}\n"

                                    model_s, err_s = dapatkan_model_gemini(st.session_state.gemini_api_key)
                                    if model_s and not err_s:
                                        try:
                                            resp_s = model_s.generate_content(sell_prompt)
                                            raw_s = resp_s.text.strip()
                                            start_s = raw_s.rfind('[')
                                            sell_ai = []
                                            if start_s != -1:
                                                json_s = raw_s[start_s:].strip()
                                                if json_s.startswith("```json"): json_s = json_s[7:]
                                                if json_s.endswith("```"): json_s = json_s[:-3]
                                                try:
                                                    sell_ai = json.loads(json_s)
                                                except json.JSONDecodeError:
                                                    sell_ai = []
                                        except:
                                            sell_ai = []
                                    else:
                                        sell_ai = []

                                    for item in sell_ai:
                                        ticker = item.get("ticker", "").upper()
                                        stock = next((r for r in top_sell_candidates if r['ticker'] == ticker), None)
                                        if stock is None:
                                            continue
                                        est_return = stock['muEst'] * 100
                                        sent_score = item.get("sentiment_score", 0.0)
                                        al_enhanced = sent_score * 100
                                        status = "☑️ Sejalan" if sent_score < 0 else "⛔ Berlawanan"
                                        note = item.get("note", "")
                                        st.markdown(f"""
                                        **{ticker}** {status}  
                                        Scanner: Jual (Est. Return {est_return:.2f}%) vs AI-Enhanced: {al_enhanced:.2f}% sentimen: {note}
                                        """)
                                else:
                                    st.caption("(Tidak ada kandidat Jual)")
                            except Exception as e:
                                st.error(f"Gagal memproses respons AI: {e}")
                        else:
                            st.error("Gagal mengakses Gemini.")
# ==================== TAMPILAN AWAL (SEBELUM ANALISIS) ====================
else:
    st.title("📊 Quant & Risk Engine Pro")
    st.markdown("""
    ## Selamat Datang di Dashboard Analisis Saham IHSG
    
    **Fitur Utama:**
    - 🔍 Analisis teknikal lengkap (EMA, ADX, RSI, Z-Score, Momentum, dll)
    - 📈 Sinyal trading adaptif (BUY/HOLD/AVOID) berdasarkan kondisi pasar
    - 🧠 V12 Adaptive Engine dengan self-learning untuk bobot indikator
    - 📰 Analisis sentimen berita dari berbagai sumber
    - 📊 Metrik fundamental (Market Cap, PER, PBV, ROE, D/E)
    - 🎲 Simulasi Monte Carlo untuk probabilitas naik & sentuh level
    - 🤖 AI Insight otomatis menggunakan Google Gemini (perlu API key)
    - 💾 Riwayat analisis tersimpan di Google Sheets (persisten)
    
    **Cara Memulai:**
    1. Pilih **Gaya Trading** di sidebar (Swing Trade mingguan / Day Trade harian)
    2. Masukkan **kode saham** IHSG (contoh: BBRI, TLKM, BMRI) – akhiran `.JK` otomatis ditambahkan
    3. Klik tombol **🚀 ANALISIS** dan tunggu beberapa detik
    
    > **Disclaimer:** Dashboard ini merupakan alat bantu analisis kuantitatif. Keputusan investasi tetap tanggung jawab masing-masing. Data historis tidak menjamin performa masa depan.
    """)

    st.markdown("---")
    st.subheader("📈 Informasi Pasar Terkini (IHSG)")

    periode_pilihan = st.selectbox(
        "Periode data IHSG:",
        options=["1d", "5d", "1mo"],
        format_func=lambda x: {"1d": "1 Hari", "5d": "5 Hari", "1mo": "1 Bulan"}[x],
        index=0,
        key="ihsg_period"
    )

    if periode_pilihan == "1d":
        interval_candidates = ["1m", "5m", "15m", "30m", "60m", "1d"]
    elif periode_pilihan == "5d":
        interval_candidates = ["5m", "15m", "30m", "60m", "1d"]
    else:
        interval_candidates = ["1d"]

    df_ihsg_preview = pd.DataFrame()
    interval_terpakai = None

    for interval in interval_candidates:
        temp_df = load_ihsg_data(period=periode_pilihan, interval=interval)
        if not temp_df.empty and len(temp_df) >= 2:
            df_ihsg_preview = temp_df
            interval_terpakai = interval
            break
        elif not temp_df.empty and len(temp_df) == 1 and interval == interval_candidates[-1]:
            df_ihsg_preview = temp_df
            interval_terpakai = interval
            break

    try:
        try:
            ihsg_info = yf.Ticker("^JKSE").info
            prev_close = ihsg_info.get('previousClose', None)
            open_price = ihsg_info.get('regularMarketOpen', None)
        except:
            prev_close = None
            open_price = None

        if not df_ihsg_preview.empty and len(df_ihsg_preview) >= 2:
            ihsg_close = float(df_ihsg_preview['Close'].iloc[-1])
            open_period = float(df_ihsg_preview['Open'].iloc[0])

            if periode_pilihan == "1d":
                if prev_close is not None and prev_close > 0:
                    ihsg_change = (ihsg_close - prev_close) / prev_close * 100
                else:
                    ihsg_prev = float(df_ihsg_preview['Close'].iloc[-2])
                    ihsg_change = (ihsg_close - ihsg_prev) / ihsg_prev * 100
            else:
                if open_period > 0:
                    ihsg_change = (ihsg_close - open_period) / open_period * 100
                else:
                    ihsg_change = 0.0

            ihsg_high = float(df_ihsg_preview['High'].max())
            ihsg_low = float(df_ihsg_preview['Low'].min())
            if open_price is None or open_price == 0:
                open_price = open_period

            if interval_terpakai in ("1m", "5m", "15m", "30m", "60m"):
                vol_val = df_ihsg_preview['Volume'].sum()
            else:
                vol_val = float(df_ihsg_preview['Volume'].iloc[-1])
            volume_str = f"{vol_val:,.0f}" if vol_val > 0 else "N/A"

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("IHSG", f"{ihsg_close:,.0f}", f"{ihsg_change:+.2f}%")
            col2.metric("Open", f"{open_price:,.0f}" if open_price else "N/A")
            col3.metric("High", f"{ihsg_high:,.0f}")
            col4.metric("Low", f"{ihsg_low:,.0f}")
            col5.metric("Volume", volume_str)

            if PLOTLY_AVAILABLE:
                line_color = '#26a69a' if ihsg_change >= 0 else '#ef5350'
                area_color = f"rgba({38 if ihsg_change >= 0 else 239}, {166 if ihsg_change >= 0 else 83}, {154 if ihsg_change >= 0 else 80}, 0.25)"

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_ihsg_preview.index,
                    y=df_ihsg_preview['Close'],
                    mode='lines',
                    line=dict(color=line_color, width=1.5),
                    fill='tozeroy',
                    fillcolor=area_color,
                    name='IHSG',
                    hovertemplate='<b>%{x|%d %b %H:%M WIB}</b><br>Close: %{y:,.0f}<extra></extra>'
                ))
                fig.add_hline(y=ihsg_high, line_dash='dot', line_color='rgba(255,255,255,0.4)')
                fig.add_annotation(x=0.5, y=ihsg_high, xref='paper', yref='y',
                                   text=f'H {ihsg_high:,.0f}', showarrow=False,
                                   font=dict(size=9, color='rgba(255,255,255,0.6)'),
                                   bgcolor='rgba(15, 17, 22, 0.7)',
                                   bordercolor='rgba(255,255,255,0.3)',
                                   borderwidth=1, borderpad=4, xanchor='center', yanchor='bottom')
                fig.add_hline(y=ihsg_low, line_dash='dot', line_color='rgba(255,255,255,0.4)')
                fig.add_annotation(x=0.5, y=ihsg_low, xref='paper', yref='y',
                                   text=f'L {ihsg_low:,.0f}', showarrow=False,
                                   font=dict(size=9, color='rgba(255,255,255,0.6)'),
                                   bgcolor='rgba(15, 17, 22, 0.7)',
                                   bordercolor='rgba(255,255,255,0.3)',
                                   borderwidth=1, borderpad=4, xanchor='center', yanchor='bottom')
                y_min = float(df_ihsg_preview['Low'].min()) * 0.998
                y_max = float(df_ihsg_preview['High'].max()) * 1.002
                fig.update_yaxes(range=[y_min, y_max])

                chart_title = {
                    "1d": "IHSG Hari Ini",
                    "5d": "IHSG 5 Hari Terakhir",
                    "1mo": "IHSG 1 Bulan Terakhir"
                }.get(periode_pilihan, "IHSG")

                fig.update_layout(
                    title=dict(text=chart_title, x=0.01, xanchor='left', font=dict(size=14, color='#e0e0e0')),
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=10, r=20, t=40, b=10),
                    dragmode='pan',
                    xaxis=dict(title=None, showgrid=False, zeroline=False, showline=True, linecolor='rgba(128,128,128,0.2)'),
                    yaxis=dict(title=None, showgrid=True, gridcolor='rgba(128,128,128,0.1)', zeroline=False, side='right'),
                    hovermode='x unified',
                    hoverlabel=dict(bgcolor='#1e293b', font_size=11, font_family="monospace"),
                    paper_bgcolor='#0f1116',
                    plot_bgcolor='#0f1116',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, config={
                    'modeBarButtonSize': 4,
                    'displaylogo': False
                })
                if interval_terpakai != "1m" and periode_pilihan == "1d":
                    st.info("ℹ️ Data 1 menit tidak tersedia, menggunakan interval yang lebih besar.")
            else:
                st.line_chart(df_ihsg_preview['Close'])
        elif not df_ihsg_preview.empty and len(df_ihsg_preview) == 1:
            ihsg_close = float(df_ihsg_preview['Close'].iloc[-1])
            if prev_close:
                ihsg_change = (ihsg_close - prev_close) / prev_close * 100
                st.metric("IHSG", f"{ihsg_close:,.0f}", f"{ihsg_change:+.2f}%")
            else:
                st.metric("IHSG", f"{ihsg_close:,.0f}")
            st.warning("Data IHSG hanya tersedia 1 titik (kemungkinan di luar jam bursa).")
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_ihsg_preview.index,
                    y=df_ihsg_preview['Close'],
                    mode='lines+markers',
                    marker=dict(color='#f59e0b', size=8),
                    line=dict(color='#f59e0b', width=2),
                    name='IHSG'
                ))
                fig.update_layout(title="IHSG (Data Terbatas)", template="plotly_dark", height=350, dragmode='pan')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df_ihsg_preview['Close'])
        else:
            st.warning("Data IHSG tidak tersedia untuk periode yang dipilih.")
    except Exception as e:
        st.error(f"Gagal memuat data IHSG: {e}")
        
# --- ANALISIS RIWAYAT DENGAN AI (TOMBOL SIDEBAR) ---
if ai_riwayat_btn:
    if not st.session_state.gemini_api_key: st.error("Masukkan API Key terlebih dahulu!")
    elif not st.session_state.riwayat: st.warning("Belum ada riwayat.")
    else:
        with st.spinner("🧠 AI menganalisis riwayat..."):
            hasil, error = analisis_riwayat_global(
                st.session_state.riwayat,
                st.session_state.riwayat_actual,
                st.session_state.gemini_api_key
            )
            if error:
                st.error(error)
            elif hasil:
                hasil_bersih = bersihkan_teks_ai(hasil)
                st.markdown(
                    f'<div class="ai-insight-card" style="border-left-color:#06b6d4;">'
                    f'<h3 style="color:#67e8f9;">📊 Insight AI dari Riwayat</h3>'
                    f'<p>{hasil_bersih}</p></div>',
                    unsafe_allow_html=True
                )
