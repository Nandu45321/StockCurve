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
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS",
    "SBIN.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", 
    "MARUTI.NS", "SUNPHARMA.NS", "AXISBANK.NS", "TITAN.NS", "KOTAKBANK.NS", "ONGC.NS", 
    "NTPC.NS", "ADANIENT.NS", "ADANIPORTS.NS", "TATAMOTORS.NS", "TITAGARH.NS", "ASIANPAINT.NS", 
    "BAJFINSV.NS", "COALINDIA.NS", "BAJAJ-AUTO.NS", "POWERGRID.NS", "NESTLEIND.NS", "M&M.NS", 
    "GRASIM.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "SBILIFE.NS", "HDFCLIFE.NS", "INDUSINDBK.NS", 
    "BRITANNIA.NS", "TECHM.NS", "CIPLA.NS", "DMART.NS", "IOC.NS", "ZOMATO.NS", "ATGL.NS", 
    "HAL.NS", "SIEMENS.NS", "VBL.NS", "NAUKRI.NS", "DIVISLAB.NS", "EICHERMOT.NS", "APOLLOHOSP.NS", 
    "TATACONSUM.NS", "PIDILITIND.NS", "BEL.NS", "SHRIRAMFIN.NS", "GODREJCP.NS", "SBFC.NS", 
    "AMBUJACEM.NS", "BPCL.NS", "CHOLAFIN.NS", "CANBK.NS", "DABUR.NS", "SRF.NS", "TRENT.NS", 
    "INDIGO.NS", "ICICIGI.NS", "ICICIPRULI.NS", "VEDL.NS", "SBICARD.NS", "HINDALCO.NS", 
    "BERGEPAINT.NS", "ABB.NS", "TVSMOTOR.NS", "MUTHOOTFIN.NS", "SBILIFE.NS", "JSL.NS", 
    "LODHA.NS", "BHEL.NS", "PIIND.NS", "COLPAL.NS", "YESBANK.NS", "PERSISTENT.NS", 
    "AUROPHARMA.NS", "UNITDSPR.NS", "SHREECEM.NS", "SBILIFE.NS", "CONCOR.NS", "POLYCAB.NS", 
    "OFSS.NS", "LUPIN.NS", "JUBLFOOD.NS", "ZYDUSLIFE.NS", "MARICO.NS", "DALBHARAT.NS", 
    "BHARATFORG.NS", "IPCALAB.NS", "DIXON.NS", "SOLARINDS.NS", "APLAPOLLO.NS", "CUMMINSIND.NS",
]

MID_CAP = [
    "ACC.NS", "ADANITOTAL.NS", "ABCAPITAL.NS", "AJANTPHARM.NS", "ALKEM.NS", "APLLTD.NS", 
    "APOLLOTYRE.NS", "APARINDS.NS", "ASHOKLEY.NS", "ASTRAL.NS", "AUROPHARMA.NS", "AUBANK.NS", 
    "BALKRISIND.NS", "BANKINDIA.NS", "MAHABANK.NS", "BERGERPAINTS", "BHARTIHEXA.NS", "BHEL.NS", 
    "BIOCON.NS", "BLUESTARCO.NS", "BDL.NS", "BSE.NS", "COCHINSHIP.NS", "COFORGE.NS", "CONCOR.NS", 
    "COROMANDEL.NS", "CRISIL.NS", "DABUR.NS", "DIXON.NS", "ESCORTS.NS", "EXIDEIND.NS", 
    "FEDERALBNK.NS", "FORTIS.NS", "GMRINFRA.NS", "GLAXO.NS", "GLENMARK.NS", "GODREJIND.NS", 
    "GODREJPROP.NS", "HAVELLS.NS", "HONAUT.NS", "HUDCO.NS", "ICICIGI.NS", "ICICIPRULI.NS", 
    "IDFCFIRSTB.NS", "INDIANB.NS", "IREDA.NS", "IRFC.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", 
    "INFY.NS", "IPCALAB.NS", "JKCEMENT.NS", "JSWENERGY.NS", "JUBLFOOD.NS", "KALYANKNIL.NS", 
    "KEI.NS", "KPITTECH.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LINDEINDIA.NS", "LTIM.NS", 
    "LTF.NS", "LUPIN.NS", "M&MFIN.NS", "MANKIND.NS", "MARICO.NS", "MAXFINAN.NS", "MPHASIS.NS", 
    "MRF.NS", "MCX.NS", "NATCOPHARM.NS", "NATIONALUM.NS", "NAVINFLUOR.NS", "NEWINDIA.NS", 
    "NHPC.NS", "NLCINDIA.NS", "NMDC.NS", "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS", "PAYTM.NS", 
    "PAGEIND.NS", "PATANJALI.NS", "PVRINOX.NS", "PBFINTECH.NS", "PERSISTENT.NS", "PETRONET.NS", 
    "PHOENIXLTD.NS", "POLYCAB.NS", "PRESTIGE.NS", "RADICO.NS", "RVNL.NS", "RELAXO.NS", "SAIL.NS", 
    "SBICARD.NS", "SCHAEFFLER.NS", "SJVN.NS", "SKFINDIA.NS", "SONACOMS.NS", "SRF.NS", 
    "SUPREMEIND.NS", "SUZLON.NS", "TATACOMM.NS", "TATADELPHI.NS", "TATAELXSI.NS", "TATAINVEST.NS", 
    "THERMAX.NS", "TIMKEN.NS", "TORNTPOWER.NS", "TRENT.NS", "TIINDIA.NS", "UPL.NS", "UNOMINDA.NS", 
    "VGUARD.NS", "VBL.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "WIPRO.NS", "YESBANK.NS", "ZEEL.NS",
]

SMALL_CAP = [
    "ALOKINDS.NS", "AMARAJABAT.NS", "ANANTRAJ.NS", "APLLTD.NS", "ASKAUTOMA.NS", "AVANTIFEED.NS", 
    "BBL.NS", "BECTORFOOD.NS", "BIPOPO.NS", "BIRLACORPN.NS", "CAMPUS.NS", "CEATLTD.NS", 
    "CENTURYPLY.NS", "CESC.NS", "CIEINDIA.NS", "COCHINSHIP.NS", "CREST.NS", "CUB.NS", "CYIENT.NS", 
    "DATAPATTNS.NS", "DBREALTY.NS", "DCMSHRIRAM.NS", "DEEPAKFERT.NS", "DELHIVERY.NS", "EIDPARRY.NS", 
    "EIHOTEL.NS", "ELECTCAST.NS", "ELGIEQUIP.NS", "ERIS.NS", "FSL.NS", "GARFIBRES.NS", "GATEWAY.NS", 
    "GODREYPROP.NS", "GPIL.NS", "GRANULES.NS", "GRAPHITE.NS", "GREENPANEL.NS", "HAPPSTMINDS.NS", 
    "HATHWAY.NS", "HEG.NS", "HGINFRA.NS", "HINDCOPPER.NS", "HOMEFIRST.NS", "HSCL.NS", "HUDCO.NS", 
    "IBREALEST.NS", "IDFC.NS", "IFCI.NS", "IIFL.NS", "INFIBEAM.NS", "INOXWIND.NS", "IOB.NS", 
    "IRB.NS", "IRIS.NS", "ITDC.NS", "J&KBANK.NS", "JAIBALAJI.NS", "JINDALSAW.NS", "JKPAPER.NS", 
    "JKTYRE.NS", "JMFINANCIL.NS", "JSWENERGY.NS", "JUBLINGR.NS", "JUSTDIAL.NS", "JYOTHYLAB.NS", 
    "KALYANKJIL.NS", "KARURVYSYA.NS", "KFINTECH.NS", "KIMS.NS", "KNRCON.NS", "KRBL.NS", 
    "LEMONTREE.NS", "LINDEINDIA.NS", "MAHSEAMLES.NS", "MANAPPURAM.NS", "MAPMYINDIA.NS", 
    "MASTEK.NS", "MEDANTA.NS", "METROPOLIS.NS", "MHRIL.NS", "MIDHANI.NS", "MINDTECK.NS", "MMTC.NS", 
    "MOIL.NS", "MOTILALOFS.NS", "MRPL.NS", "MTARTECH.NS", "NATIONALUM.NS", "NAVA.NS", "NBCC.NS", 
    "NCC.NS", "NESCO.NS", "NETWORK18.NS", "NHPC.NS", "NLCINDIA.NS", "NMDC.NS", "NOCIL.NS", 
    "NURECA.NS", "NUVOCO.NS", "OLECTRA.NS", "ONMOBILE.NS", "ORIENTELEC.NS", "PARAS.NS", 
    "PATANJALI.NS", "PCJEWELLER.NS", "PEL.NS", "PERSISTENT.NS", "PFC.NS", "PNCINFRA.NS", 
    "PRAJMIND.NS", "PRINCEPIPE.NS", "PRUDENT.NS", "PVRINOX.NS", "QUESS.NS", "RADICO.NS", 
    "RAILTEL.NS", "RAIN.NS", "RAJESHEXPO.NS", "RALLIS.NS", "RBA.NS", "RCF.NS", "RECLTD.NS", 
    "RENUKA.NS", "RITES.NS", "ROUTE.NS", "SAIL.NS", "SANSERA.NS", "SAPPHIRE.NS", "SARDAEN.NS", 
    "SJVN.NS", "SKFINDIA.NS", "SONACOMS.NS", "SOUTHBANK.NS", "SPARC.NS", "STLTECH.NS", "SUBROS.NS", 
    "SUDARSCHEM.NS", "SUNTECK.NS", "SUPRAJIT.NS", "SUPREMEIND.NS", "SUZLON.NS", "SWANENERGY.NS", 
    "SYNGENE.NS", "TANLA.NS", "TATAINVEST.NS", "TEJASNET.NS", "TEXRAIL.NS", "THOMASCOOK.NS", 
    "THYROCARE.NS", "TIIL.NS", "TIMKEN.NS", "TRIDENT.NS", "TRITURBINE.NS", "TV18BRDCST.NS", 
    "UCOBANK.NS", "UJJIVANSFB.NS", "UNIONBANK.NS", "USHAMART.NS", "VAIBHAVGBL.NS", "VAKRANGEE.NS", 
    "VATECHWABAG.NS", "VENKEYS.NS", "VESUVIUS.NS", "VIJAYA.NS", "VIPIND.NS", "VOLTAMP.NS", 
    "WELCORP.NS", "WELSPUNIND.NS", "WESTLIFE.NS", "WOCKHARDT.NS", "ZENSARTECH.NS", "ZYDUSWELL.NS",
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

        # CONFLICT NOTE: AGENTS.md says "std of raw window" but thresholds 0.01–3.0
        # only make sense for a dimensionless measure. Raw price std for NSE stocks
        # ranges from ₹8 (penny stocks) to ₹700+ (MRF), so std_thresh_high=3.0 would
        # eliminate everything. We store coefficient of variation (std/mean) instead,
        # which is scale-free. CV ≈ 0.02–0.30 for typical 60-day windows.
        std_val = float(np.std(raw) / np.mean(raw))
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
