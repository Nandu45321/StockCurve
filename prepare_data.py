"""
prepare_data.py — Run once to build data/windows.npy and data/meta.json.

Downloads historical price data for Nifty 500 stocks from yfinance,
slices into overlapping windows, normalizes each window, and saves
the result as a numpy array of dicts.
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Canonical normalization functions (copied verbatim from AGENTS.md)
# ---------------------------------------------------------------------------

def resample(arr: np.ndarray, target: int = 50) -> np.ndarray:
    """Resample arr to `target` points using linear interpolation."""
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target)
    return np.interp(x_new, x_old, arr)


def znorm(arr: np.ndarray) -> np.ndarray:
    """Z-normalize arr to zero mean and unit std (epsilon prevents div-by-zero)."""
    mean = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mean) / std


def smooth(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian smoothing with given sigma."""
    return gaussian_filter1d(arr, sigma=sigma)


def fingerprint(arr: np.ndarray, k: int = 8) -> np.ndarray:
    """Return first k magnitudes of the FFT of arr."""
    return np.abs(np.fft.fft(arr))[:k]


def sharpness(arr: np.ndarray) -> float:
    """Compute second-order difference mean as a sharpness metric."""
    return float(np.mean(np.abs(np.diff(arr, n=2))))


def sigmoid(x: float) -> float:
    """Standard sigmoid function."""
    import math
    return 1 / (1 + math.exp(-x))


# ---------------------------------------------------------------------------
# Stock universe (Nifty 500 representative sample)
# AGENTS.md provides the seed lists; expanded here for decent coverage.
# ---------------------------------------------------------------------------

LARGE_CAP = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "KOTAKBANK.NS",
    "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "COALINDIA.NS", "BPCL.NS",
    "TECHM.NS", "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "BAJAJFINSV.NS", "GRASIM.NS", "HEROMOTOCO.NS", "TATAMOTORS.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "INDUSINDBK.NS", "BRITANNIA.NS",
]

MID_CAP = [
    "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS", "LTTS.NS", "OFSS.NS",
    "PIIND.NS", "CROMPTON.NS", "VOLTAS.NS", "ESCORTS.NS", "TRENT.NS",
    "JUBLFOOD.NS", "INDIGO.NS", "IPCALAB.NS", "LALPATHLAB.NS", "METROPOLIS.NS",
    "AAVAS.NS", "CHOLAFIN.NS", "MFSL.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "PHOENIXLTD.NS", "OBEROIRLTY.NS", "GODREJPROP.NS", "PRESTIGE.NS", "BRIGADE.NS",
    "ABCAPITAL.NS", "MANAPPURAM.NS", "MUTHOOTFIN.NS", "CANFINHOME.NS", "LICHSGFIN.NS",
    "AUROPHARMA.NS", "TORNTPHARM.NS", "ALKEM.NS", "SYNGENE.NS", "NATCOPHARM.NS",
    "POLYCAB.NS", "HAVELLS.NS", "WHIRLPOOL.NS", "BLUESTARCO.NS", "SYMPHONY.NS",
    "ASTRAL.NS", "SUPREMEIND.NS", "NILKAMAL.NS", "CEATLTD.NS", "MRF.NS",
    "AMARAJABAT.NS", "EXIDEIND.NS", "MOTHERSON.NS", "BALKRISIND.NS", "APOLLOTYRE.NS",
]

SMALL_CAP = [
    "KPRMILL.NS", "WELCORP.NS", "MAHINDCIE.NS", "SPARC.NS", "NIACL.NS",
    "RBLBANK.NS", "IDFC.NS", "UJJIVANSFB.NS", "EQUITASBNK.NS", "SURYODAY.NS",
    "PVRINOX.NS", "INOXLEISUR.NS", "NAUKRI.NS", "INDIAMART.NS", "JUSTDIAL.NS",
    "NAZARA.NS", "EASEMYTRIP.NS", "CARTRADE.NS", "DELHIVERY.NS", "ZOMATO.NS",
    "NYKAA.NS", "PAYTM.NS", "POLICYBZR.NS", "MAPMYINDIA.NS", "ROUTE.NS",
    "TANLA.NS", "INTELLECT.NS", "MASTECH.NS", "NEWGEN.NS", "KFINTECH.NS",
]

# Map each symbol to its cap category
CAP_MAP: dict[str, str] = {}
for s in LARGE_CAP:
    CAP_MAP[s] = "large"
for s in MID_CAP:
    CAP_MAP[s] = "mid"
for s in SMALL_CAP:
    CAP_MAP[s] = "small"

ALL_SYMBOLS = LARGE_CAP + MID_CAP + SMALL_CAP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_DAYS = 60          # default window length
STEP_DAYS = 10            # sliding-window step
HISTORY_YEARS = 3         # how far back to download
DATA_DIR = "data"
WINDOWS_PATH = os.path.join(DATA_DIR, "windows.npy")
META_PATH = os.path.join(DATA_DIR, "meta.json")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def fetch_prices(symbol: str, period: str = f"{HISTORY_YEARS}y") -> pd.Series | None:
    """Download adjusted close prices for a symbol. Returns None on failure."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, auto_adjust=True)
        if hist.empty or len(hist) < WINDOW_DAYS + 5:
            print(f"  [SKIP] {symbol}: insufficient data ({len(hist)} rows)")
            return None
        return hist["Close"]
    except Exception as e:
        print(f"  [ERROR] {symbol}: {e}")
        return None


def fetch_info(symbol: str) -> dict:
    """Return best-effort metadata dict for a symbol from yfinance."""
    try:
        info = yf.Ticker(symbol).info
        return {
            "name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector") or "Unknown",
        }
    except Exception:
        return {"name": symbol, "sector": "Unknown"}


def make_windows(prices: pd.Series, symbol: str, cap: str, sector: str) -> list[dict]:
    """
    Slice a price series into overlapping windows of WINDOW_DAYS length.

    Returns a list of window dicts matching the AGENTS.md data schema.
    """
    windows = []
    closes = prices.values.astype(np.float64)
    dates = prices.index

    for start in range(0, len(closes) - WINDOW_DAYS + 1, STEP_DAYS):
        end = start + WINDOW_DAYS
        raw = closes[start:end]

        if np.any(np.isnan(raw)) or np.any(raw <= 0):
            continue  # skip corrupted windows

        std_val = float(np.std(raw))
        norm_arr = znorm(resample(raw, 50))
        smooth_arr = smooth(norm_arr, sigma=2.0)
        date_end = str(dates[end - 1].date())

        windows.append({
            "symbol": symbol,
            "date_end": date_end,
            "cap": cap,
            "sector": sector,
            "raw": raw.copy(),
            "norm": norm_arr,
            "smooth": smooth_arr,
            "std": std_val,
        })

    return windows


def build_database() -> tuple[list[dict], list[dict]]:
    """
    Download data for all symbols and build the windows list and meta list.

    Returns (all_windows, meta_list).
    """
    all_windows: list[dict] = []
    meta_list: list[dict] = []
    seen_symbols: set[str] = set()

    total = len(ALL_SYMBOLS)
    for i, symbol in enumerate(ALL_SYMBOLS, 1):
        print(f"[{i}/{total}] {symbol} ...", end=" ", flush=True)
        cap = CAP_MAP[symbol]

        prices = fetch_prices(symbol)
        if prices is None:
            continue

        info = fetch_info(symbol)
        sector = info["sector"]
        name = info["name"]

        windows = make_windows(prices, symbol, cap, sector)
        if not windows:
            print(f"no valid windows")
            continue

        all_windows.extend(windows)
        print(f"OK  {len(windows)} windows")

        if symbol not in seen_symbols:
            seen_symbols.add(symbol)
            meta_list.append({
                "symbol": symbol,
                "name": name,
                "sector": sector,
                "cap": cap,
            })

    return all_windows, meta_list


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Build and save data/windows.npy and data/meta.json."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Create .gitignore so windows.npy is never committed (too large)
    gitignore_path = os.path.join(DATA_DIR, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as f:
            f.write("windows.npy\n")
        print("Created data/.gitignore")

    print(f"Downloading data for {len(ALL_SYMBOLS)} symbols...")
    print(f"Window size: {WINDOW_DAYS} days | Step: {STEP_DAYS} days | History: {HISTORY_YEARS}y\n")

    all_windows, meta_list = build_database()

    if not all_windows:
        print("\n[ERROR] No windows generated. Check your internet connection.")
        return

    # Save windows as numpy array of dicts (allow_pickle=True required for dicts)
    np.save(WINDOWS_PATH, np.array(all_windows, dtype=object), allow_pickle=True)
    print(f"\nSaved {len(all_windows)} windows -> {WINDOWS_PATH}")

    with open(META_PATH, "w") as f:
        json.dump(meta_list, f, indent=2)
    print(f"Saved {len(meta_list)} symbols -> {META_PATH}")

    # Summary stats
    cap_counts: dict[str, int] = {}
    for w in all_windows:
        cap_counts[w["cap"]] = cap_counts.get(w["cap"], 0) + 1
    print("\nWindow breakdown by cap:")
    for cap, count in sorted(cap_counts.items()):
        print(f"  {cap:6s}: {count} windows")


if __name__ == "__main__":
    main()
