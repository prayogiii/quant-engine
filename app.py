"""
IDX Cron Scraper — jalankan tiap hari jam 18:00 WIB.
Scrape foreign flow dari IDX untuk semua ticker yang ada di broksum_history,
lalu push ke sheet foreign_flow_history.
"""

import os
import sys
from datetime import datetime
import pytz
import time
import gspread
from google.oauth2.service_account import Credentials
from curl_cffi import requests as curl_requests

# ═══════════════════════════════════════════════════
# CONFIG — SESUAIKAN
# ═══════════════════════════════════════════════════
# Path ke file service account JSON (yang dipakai Streamlit)
CREDENTIALS_PATH = r"D:\stock_analysis\riwayat-analisis-1e97def5d873.json"

# Sheet ID (dari URL Google Sheets)
SHEET_ID = "14YPBBBU2Yzd7NGCLYHrubn5iUls-6NVZpFg53QmMpXg"   # ← GANTI dengan ID sheet kamu

# ═══════════════════════════════════════════════════
# FUNGSI
# ═══════════════════════════════════════════════════
def get_sheet():
    creds = Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_ID)


import time

def fetch_idx_all(max_retries=3):
    """Fetch semua saham dari IDX dengan retry logic untuk handle 403/429."""
    url = "https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=9999&start=0"
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9,id;q=0.8",
        "accept-encoding": "gzip, deflate, br",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "sec-ch-ua": '"Chromium";v="120", "Not_A Brand";v="8"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "referer": "https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-saham/",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "x-requested-with": "XMLHttpRequest",
    }

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[*] Fetch IDX (attempt {attempt}/{max_retries})...")
            r = curl_requests.get(
                url, headers=headers, timeout=30,
                impersonate="chrome120"
            )

            if r.status_code == 200:
                payload = r.json()
                if isinstance(payload, dict):
                    items = payload.get("data") or payload.get("Data") or []
                elif isinstance(payload, list):
                    items = payload
                else:
                    items = None
                if items:
                    return items
                print(f"[!] Response 200 tapi data kosong")
                return None

            elif r.status_code in (403, 429):
                wait_sec = 30 * attempt  # 30s, 60s, 90s
                print(f"[!] HTTP {r.status_code} — rate limited. "
                      f"Tunggu {wait_sec}s sebelum retry...")
                if attempt < max_retries:
                    time.sleep(wait_sec)
                    continue
                else:
                    print(f"[!] Gagal setelah {max_retries} percobaan")
                    return None
            else:
                print(f"[!] HTTP {r.status_code} — tidak terduga")
                return None

        except Exception as e:
            print(f"[!] Fetch error (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(10 * attempt)
                continue
            return None

    return None


def get_tickers_from_broksum(sheet):
    """Ambil daftar ticker unik dari broksum_history + riwayat."""
    tickers = set()

    # ── Sumber 1: broksum_history (ticker yang pernah di-upload broksumnya)
    try:
        ws = sheet.worksheet("broksum_history")
        records = ws.get_all_records()
        for r in records:
            t = str(r.get("ticker", "")).upper().replace(".JK", "").strip()
            if t:
                tickers.add(t)
        print(f"[✓] broksum_history: {len(tickers)} ticker")
    except Exception as e:
        print(f"[!] Get tickers from broksum error: {e}")

    # ── Sumber 2: riwayat (SEMUA ticker yang pernah dianalisis)
    try:
        before = len(tickers)
        ws = sheet.worksheet("riwayat")
        records = ws.get_all_records()
        for r in records:
            t = str(r.get("Saham", "")).upper().replace(".JK", "").strip()
            if t:
                tickers.add(t)
        print(f"[✓] riwayat: +{len(tickers) - before} ticker baru")
    except Exception as e:
        print(f"[!] Get tickers from riwayat error: {e}")

    return sorted(tickers)


def main():
    print("=" * 60)
    print(f"IDX Cron Scraper — {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%Y-%m-%d %H:%M WIB')}")
    print("=" * 60)

    # 1. Buka sheet
    try:
        sheet = get_sheet()
        print("[✓] Connected to Google Sheets")
    except Exception as e:
        print(f"[!] Gagal connect ke Sheets: {e}")
        sys.exit(1)

    # 2. Ambil ticker yang perlu di-scrape
    tickers = get_tickers_from_broksum(sheet)
    if not tickers:
        print("[!] Tidak ada ticker di broksum_history")
        sys.exit(0)
    print(f"[*] Ticker yang akan di-scrape: {len(tickers)}")
    print(f"    {', '.join(tickers[:20])}{' ...' if len(tickers) > 20 else ''}")

    # 3. Fetch IDX sekali (semua saham)
    print(f"\n[*] Fetching IDX...")
    items = fetch_idx_all()
    if not items:
        print("[!] Gagal fetch IDX. Exit.")
        sys.exit(1)
    print(f"[✓] IDX: {len(items)} saham")

    # 4. Build lookup ticker → data
    lookup = {}
    today_str = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d")

    for it in items:
        if not isinstance(it, dict):
            continue
        code = str(it.get("StockCode", "")).upper().strip()
        if not code:
            continue

        def _f(k):
            v = it.get(k)
            if v in (None, "", "N/A", "-"):
                return 0.0
            try:
                return float(str(v).replace(",", ""))
            except Exception:
                return 0.0

        fb = _f("ForeignBuy")
        fs = _f("ForeignSell")
        close = _f("Close")
        fb_rp = fb * close if close > 0 else 0.0
        fs_rp = fs * close if close > 0 else 0.0

        date_str = str(it.get("Date") or "")[:10] or today_str

        lookup[code] = {
            "ticker": code,
            "date": date_str,
            "close": close,
            "foreign_buy": fb_rp,
            "foreign_sell": fs_rp,
            "net_foreign": fb_rp - fs_rp,
            "source": "idx_cron",
        }

    # 5. Load existing records di foreign_flow_history
    try:
        ws = sheet.worksheet("foreign_flow_history")
    except Exception:
        # Buat kalau belum ada
        ws = sheet.add_worksheet("foreign_flow_history", rows=5000, cols=7)
        ws.update("A1:G1", [[
            "ticker", "date", "close",
            "foreign_buy", "foreign_sell", "net_foreign", "source"
        ]], value_input_option='RAW')

    existing = ws.get_all_records()
    existing_keys = set()
    for r in existing:
        t = str(r.get("ticker", "")).upper()
        d = str(r.get("date", ""))
        if t and d:
            existing_keys.add((t, d))

    # 6. Append yang belum ada
    to_add = []
    for tk in tickers:
        if tk not in lookup:
            print(f"    [!] {tk} — tidak ada di IDX")
            continue
        row = lookup[tk]
        key = (row["ticker"], row["date"])
        if key in existing_keys:
            print(f"    [~] {tk} — skip (sudah ada)")
            continue
        to_add.append([
            row["ticker"], row["date"], row["close"],
            row["foreign_buy"], row["foreign_sell"],
            row["net_foreign"], row["source"]
        ])
        print(f"    [+] {tk} — {row['date']} net={row['net_foreign']:,.0f}")

    if to_add:
        ws.append_rows(to_add, value_input_option='RAW')
        print(f"\n[✓] Added {len(to_add)} rows ke foreign_flow_history")
    else:
        print("\n[~] Tidak ada data baru untuk ditambahkan")

    print("\n[✓] Done!")


if __name__ == "__main__":
    main()
