"""
NSE Option Chain Dashboard v3
Path: {BASE}/{YEAR}/{MON}/{YEAR}_{MM}_{DD}.parquet
Tickers: OLD format  NIFTY19JAN10900CE.NFO  |  Futures: NIFTY-I.NFO
Run: streamlit run option_chain_dashboard_v3.py
"""

import os, re, warnings, calendar
from datetime import date, timedelta, datetime
from pathlib import Path
from collections import Counter

from huggingface_hub import HfApi

import numpy as np
import pandas as pd
import duckdb
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — Remote Hugging Face Repository
# ══════════════════════════════════════════════════════════════════════════════
REPO_ID       = "Danish-for-TFU/Nifty-Option-Data-1min"
OPTIONS_BASE  = f"hf://datasets/{REPO_ID}"
EXPIRY_CSV    = f"{OPTIONS_BASE}/Nifty Expiries (2015-2026).csv"
SESSION_START = "09:15"
SESSION_END   = "15:30"

def get_hf_token():
    """Retrieve token from Streamlit Secrets or Environment Variable."""
    if "HF_TOKEN" in st.secrets:
        return st.secrets["HF_TOKEN"]
    return os.getenv("HF_TOKEN", "")

# Initialise HF Token
HF_TOKEN = get_hf_token()

MON_FWD = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
           7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}
MON_REV = {v:k for k,v in MON_FWD.items()}

PRIORITY_INSTRUMENTS = [
    "NIFTY","BANKNIFTY","NIFTYIT",
    "RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS",
    "BAJFINANCE","SBIN","AXISBANK","LT","TATAMOTORS",
    "WIPRO","MARUTI","KOTAKBANK","HINDUNILVR","BHARTIARTL",
]

DEMO_SPOT = {"NIFTY":10900,"BANKNIFTY":27500,"NIFTYIT":15000,
             "RELIANCE":1100,"HDFCBANK":2100,"INFY":700,"TCS":1900,
             "SBIN":280,"BAJFINANCE":2600}
DEMO_ITV  = {"NIFTY":50,"BANKNIFTY":100,"NIFTYIT":50,
             "RELIANCE":20,"HDFCBANK":20,"INFY":20,"TCS":50,
             "SBIN":10,"BAJFINANCE":20}

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Option Chain",page_icon="📊",
                   layout="wide",initial_sidebar_state="collapsed")

# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def parquet_path(base, d):
    # base is already hf://datasets/REPO_ID
    return f"{base}/{d.year}/{MON_FWD[d.month]}/{d.year}_{d.month:02d}_{d.day:02d}.parquet"


_MON_PAT = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
_RE_SERIES  = re.compile(r'^(.+?)-(I{1,3})$')
# NEW 2021+ :  INST DD MMM YY STRIKE CE/PE
_RE_NEW_OPT = re.compile(r'^(.+?)(\d{2})(' + _MON_PAT + r')(\d{2})(\d+)(CE|PE)$')
# OLD pre-2021: INST YY MMM STRIKE CE/PE   (no day)
_RE_OLD_OPT = re.compile(r'^(.+?)(\d{2})(' + _MON_PAT + r')(\d+)(CE|PE)$')
# NEW futures
_RE_NEW_FUT = re.compile(r'^(.+?)(\d{2})(' + _MON_PAT + r')(\d{2})FUT$')
# OLD futures
_RE_OLD_FUT = re.compile(r'^(.+?)(\d{2})(' + _MON_PAT + r')FUT$')


def parse_ticker(ticker):
    """Parse NSE option/futures ticker into components.

    Format disambiguation:
      OLD (pre-2021): INST + YY + MMM + STRIKE + CE/PE   e.g. NIFTY19JAN10900CE
      NEW (2021+)   : INST + DD + MMM + YY + STRIKE + CE/PE  e.g. NIFTY30JAN2524950CE

    Problem: RELIANCE19JAN1440CE matches NEW as dd=19,JAN,yy=14,strike=40 (wrong).
    Fix: try OLD first. If OLD yields strike >= 100, accept it.
         Only try NEW if OLD fails or gives strike < 100.
         This works because pre-2021 data is always OLD, and NEW-format strikes are always >= 100.
    """
    t = ticker.replace(".NFO","").replace(".BSE","").strip().upper()

    # Futures series  NIFTY-I  /  RELIANCE-II  /  ACC-III
    m = _RE_SERIES.match(t)
    if m:
        return dict(inst=m.group(1), exp=None, strike=None, otype="FUT_SERIES",
                    series=m.group(2))

    # Both OLD and NEW regexes can match the same ticker string.
    # Disambiguation: try both, pick the one with a reasonable strike.
    #   OLD: INST + YY + MMM + STRIKE    e.g. NIFTY19JAN10900CE  (strike=10900)
    #   NEW: INST + DD + MMM + YY + STRIKE  e.g. BANKNIFTY14FEB1927400CE (strike=27400)
    # Heuristic: real strikes are in [1, 200000]. If OLD gives strike > 200000,
    # it's actually a NEW-format ticker. If NEW gives strike < 100 for a number
    # that OLD would give >= 100, prefer OLD.

    m_old = _RE_OLD_OPT.match(t)
    m_new = _RE_NEW_OPT.match(t)

    old_result = None
    if m_old:
        yy,mon,strike = int(m_old.group(2)),m_old.group(3),int(m_old.group(4))
        fy = 2000+yy
        if 2014<=fy<=2035 and strike>0:
            try:
                last_day = calendar.monthrange(fy, MON_REV[mon])[1]
                exp = date(fy, MON_REV[mon], last_day)
                old_result = dict(inst=m_old.group(1), exp=exp, strike=strike,
                                  otype=m_old.group(5), exp_label=f"{mon}{yy:02d}")
            except (ValueError,KeyError): pass

    new_result = None
    if m_new:
        dd,mon,yy,strike = int(m_new.group(2)),m_new.group(3),int(m_new.group(4)),int(m_new.group(5))
        fy = 2000+yy
        if 1<=dd<=31 and 2014<=fy<=2035 and strike>0:
            try:
                exp = date(fy, MON_REV[mon], dd)
                new_result = dict(inst=m_new.group(1), exp=exp, strike=strike,
                                  otype=m_new.group(6), exp_label=f"{mon}{yy:02d}")
            except ValueError: pass

    if old_result and new_result:
        # If OLD strike is absurdly large (> 200000), it's actually NEW format
        if old_result["strike"] > 200000:
            return new_result
        # If both reasonable, prefer OLD (more common in pre-2021 data)
        return old_result
    if old_result:
        return old_result
    if new_result:
        return new_result

    # OLD futures  NIFTY19JANFUT  (try before NEW to avoid same ambiguity)
    m = _RE_OLD_FUT.match(t)
    if m:
        yy,mon = int(m.group(2)),m.group(3)
        fy = 2000+yy
        try:
            last_day = calendar.monthrange(fy, MON_REV[mon])[1]
            exp = date(fy, MON_REV[mon], last_day)
            return dict(inst=m.group(1), exp=exp, strike=None, otype="FUT",
                        exp_label=f"{mon}{yy:02d}")
        except (ValueError,KeyError): pass

    # NEW futures  NIFTY30JAN25FUT
    m = _RE_NEW_FUT.match(t)
    if m:
        dd,mon,yy = int(m.group(2)),m.group(3),int(m.group(4))
        fy = 2000+yy
        if 1<=dd<=31 and 2014<=fy<=2035:
            try:
                exp = date(fy, MON_REV[mon], dd)
                return dict(inst=m.group(1), exp=exp, strike=None, otype="FUT",
                            exp_label=f"{mon}{yy:02d}")
            except ValueError: pass

    return dict(inst=None, exp=None, strike=None, otype=None)


def _norm_time(t):
    """Normalise '09:15:59' -> '09:15' (drop seconds)."""
    s = str(t).strip()
    if len(s) >= 5:
        return s[:5]
    return s


def _detect_interval(strikes):
    if len(strikes) < 2: return 50
    s = sorted(set(int(x) for x in strikes if x and x > 0))
    diffs = [s[i+1]-s[i] for i in range(len(s)-1) if s[i+1]-s[i]>0]
    if not diffs: return 50
    return Counter(diffs).most_common(1)[0][0]


def _minutes_list():
    mins=[]; h,m=9,15
    while h*60+m<=15*60+30:
        mins.append(f"{h:02d}:{m:02d}"); m+=1
        if m==60: h+=1; m=0
    return mins


def fmt_chg(v):
    cls = "chg-pos" if v>0 else ("chg-neg" if v<0 else "chg-neu")
    sign = "+" if v>0 else ""
    return f'<span class="{cls}">{sign}{v:.1f}%</span>'


# ──────────────────────────────────────────────────────────────────────────────
#  CACHED LOADERS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, max_entries=1, show_spinner=False)
def load_expiry_df(csv_path):
    try:
        # csv_path is hf://...
        df = pd.read_csv(csv_path, storage_options={"token": HF_TOKEN})
        df["adjusted_expiry"] = pd.to_datetime(df["adjusted_expiry"], errors="coerce")
        df = df.dropna(subset=["adjusted_expiry"]).copy()
        df["exp_date"] = df["adjusted_expiry"].dt.date
        return df.sort_values("adjusted_expiry").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading expiry CSV: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=7200, max_entries=2, show_spinner=False)
def get_available_dates(repo_id):
    """Crawl the Hugging Face repo to find all available parquet dates."""
    try:
        api = HfApi(token=HF_TOKEN)
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        pat = re.compile(r"(\d{4})/([A-Z]{3})/(\d{4})_(\d{2})_(\d{2})\.parquet$")
        avail = []
        for f in files:
            mm = pat.match(f)
            if mm:
                try:
                    avail.append(date(int(mm.group(3)), int(mm.group(4)), int(mm.group(5))))
                except ValueError: pass
        return sorted(list(set(avail)))
    except Exception as e:
        # Fallback to demo mode dates if API fails
        st.sidebar.error(f"HF API Error: {e}")
        d = date.today(); avail = []
        while len(avail) < 60:
            if d.weekday() < 5: avail.append(d)
            d -= timedelta(days=1)
        return sorted(avail)


@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def load_parquet_raw(pp):
    """Load entire parquet once per file — cached. Returns (df, all_tickers, parsed_map)."""
    try:
        # pp is hf://datasets/REPO_ID/...
        df = pd.read_parquet(pp, storage_options={"token": HF_TOKEN})
    except Exception:
        return pd.DataFrame(), [], {}
    if df.empty:
        return df, [], {}

    # Normalise column names: map whatever casing to our standard names
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "ticker":          col_map[c] = "ticker"
        elif cl == "time":          col_map[c] = "time"
        elif cl == "date":          col_map[c] = "date"
        elif cl == "open":          col_map[c] = "open_"
        elif cl == "high":          col_map[c] = "high"
        elif cl == "low":           col_map[c] = "low"
        elif cl == "close":         col_map[c] = "close"
        elif cl == "volume":        col_map[c] = "volume"
        elif cl in ("open interest","openinterest","oi"):
            col_map[c] = "oi"
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    required = ["ticker","time","open_","high","low","close"]
    for r in required:
        if r not in df.columns:
            return pd.DataFrame(), [], {}

    if "volume" not in df.columns:
        df["volume"] = 0
    if "oi" not in df.columns:
        df["oi"] = 0

    # Cast float32 -> float64 to avoid precision loss
    for c in ["open_","high","low","close"]:
        if df[c].dtype == "float32":
            df[c] = df[c].astype("float64")

    # Sort and build minute column
    df = df.sort_values(["ticker","time"]).reset_index(drop=True)
    df["minute"] = df["time"].apply(_norm_time)

    all_tickers = sorted(df["ticker"].unique().tolist())
    parsed_map = {tk: parse_ticker(tk) for tk in all_tickers}
    return df, all_tickers, parsed_map


def get_instruments_from_parsed(parsed_map):
    """Extract sorted unique instruments, priority first."""
    insts = set()
    for p in parsed_map.values():
        if p["inst"]:
            insts.add(p["inst"])
    priority = [i for i in PRIORITY_INSTRUMENTS if i in insts]
    rest = sorted(i for i in insts if i not in set(PRIORITY_INSTRUMENTS))
    return priority + rest


# ──────────────────────────────────────────────────────────────────────────────
#  BUILD OPTION CHAIN from raw df
# ──────────────────────────────────────────────────────────────────────────────

def _format_exp_date(exp_date, sel_date):
    """Format expiry date like Sensibull: '17 Feb (0d)' or '24 Feb (7d)'."""
    if not exp_date or not hasattr(exp_date, 'strftime'):
        return str(exp_date)
    dte = (exp_date - sel_date).days
    return f"{exp_date.strftime('%d %b')} ({dte}d)"


def _exp_date_to_key(exp_date):
    """Convert date to string key for session_state storage."""
    if exp_date and hasattr(exp_date, 'isoformat'):
        return exp_date.isoformat()
    return str(exp_date)


def _key_to_exp_date(key):
    """Convert string key back to date."""
    if not key:
        return None
    try:
        return date.fromisoformat(key)
    except (ValueError, TypeError):
        return None


def build_chain(df, parsed_map, inst, time_str, sel_exp_key, depth, sel_date=None):
    """
    Build option chain for `inst` at `time_str`.
    sel_exp_key = ISO date string of selected expiry (e.g. '2022-02-17')
    Groups by actual expiry date — supports weekly expiries.
    """
    sel_exp_date = _key_to_exp_date(sel_exp_key)

    # Tag rows
    inst_mask = df["ticker"].map(lambda tk: parsed_map[tk]["inst"] == inst)
    otype_mask = df["ticker"].map(lambda tk: parsed_map[tk]["otype"] in ("CE","PE"))
    df_inst = df[inst_mask & otype_mask].copy()
    if df_inst.empty:
        return {}

    df_inst["strike"] = df_inst["ticker"].map(lambda tk: parsed_map[tk]["strike"])
    df_inst["otype"]  = df_inst["ticker"].map(lambda tk: parsed_map[tk]["otype"])
    df_inst["exp"]    = df_inst["ticker"].map(lambda tk: parsed_map[tk]["exp"])

    # All unique expiry dates, sorted chronologically
    all_exp_dates = sorted([e for e in df_inst["exp"].dropna().unique() if e is not None])
    if not all_exp_dates:
        return {}
    if sel_exp_date not in all_exp_dates:
        sel_exp_date = all_exp_dates[0]

    # Get the exact minute bar at time_str (minute-level OHLCV)
    df_at_min = df_inst[df_inst["minute"] == time_str].copy()

    # Fallback: if no data at exact minute, use last available minute <= time_str
    if df_at_min.empty:
        df_before = df_inst[df_inst["minute"] <= time_str]
        if df_before.empty:
            df_before = df_inst
        last_min = df_before["minute"].max()
        df_at_min = df_inst[df_inst["minute"] == last_min].copy()

    # For OI: use last available up to time_str (OI doesn't update every bar)
    df_oi = (df_inst[df_inst["minute"] <= time_str]
             .sort_values("minute")
             .groupby("ticker")["oi"].last()
             .reset_index(name="oi_latest"))

    # Snapshot: one row per ticker — the exact minute bar's OHLCV
    snap = (df_at_min.groupby("ticker")
            .agg(open_=("open_","last"), high=("high","last"),
                 low=("low","last"), close=("close","last"),
                 volume=("volume","last"),
                 strike=("strike","first"), otype=("otype","first"),
                 exp=("exp","first"))
            .reset_index())

    # Merge OI (keep minute bar's volume as-is)
    snap = pd.merge(snap, df_oi, on="ticker", how="left")
    snap["oi"] = snap["oi_latest"].fillna(0)
    snap = snap.drop(columns=["oi_latest"], errors="ignore")

    # Filter to selected expiry date
    df_exp = snap[snap["exp"] == sel_exp_date].copy()
    if df_exp.empty:
        return {}

    # Aggregate by strike to avoid duplicates
    ce = (df_exp[df_exp["otype"]=="CE"]
          .groupby("strike")
          .agg(ce_open=("open_","first"), ce_high=("high","max"),
               ce_low=("low","min"), ce_ltp=("close","last"),
               ce_oi=("oi","sum"), ce_vol=("volume","sum"))
          .reset_index())
    pe = (df_exp[df_exp["otype"]=="PE"]
          .groupby("strike")
          .agg(pe_open=("open_","first"), pe_high=("high","max"),
               pe_low=("low","min"), pe_ltp=("close","last"),
               pe_oi=("oi","sum"), pe_vol=("volume","sum"))
          .reset_index())

    chain = pd.merge(ce, pe, on="strike", how="outer").sort_values("strike").fillna(0)
    chain["ce_oi_lakh"] = (chain["ce_oi"]/1e5).round(1)
    chain["pe_oi_lakh"] = (chain["pe_oi"]/1e5).round(1)

    # OI change vs market open
    first_min = df_inst["minute"].min()
    open_oi = (df_inst[df_inst["minute"]==first_min]
               .groupby("ticker")["oi"].last().reset_index(name="oi_open"))
    snap2 = pd.merge(snap, open_oi, on="ticker", how="left").fillna(0)
    snap2["oi_chg"] = snap2["oi"] - snap2["oi_open"]

    for side, ot in [("ce","CE"),("pe","PE")]:
        chg = (snap2[(snap2["otype"]==ot)&(snap2["exp"]==sel_exp_date)]
               .groupby("strike")["oi_chg"].sum().reset_index()
               .rename(columns={"oi_chg":f"{side}_oi_chg"}))
        chain = pd.merge(chain, chg, on="strike", how="left").fillna(0)

    chain["ce_oi_base"] = (chain["ce_oi"]-chain["ce_oi_chg"]).clip(lower=0)
    chain["pe_oi_base"] = (chain["pe_oi"]-chain["pe_oi_chg"]).clip(lower=0)
    chain["ce_chg_pct"] = (chain["ce_oi_chg"]/chain["ce_oi_base"].clip(lower=1)*100).round(1)
    chain["pe_chg_pct"] = (chain["pe_oi_chg"]/chain["pe_oi_base"].clip(lower=1)*100).round(1)

    # Spot from futures
    fut_mask = df["ticker"].map(lambda tk: parsed_map[tk]["inst"]==inst and
                                 parsed_map[tk]["otype"] in ("FUT_SERIES","FUT"))
    df_fut = df[fut_mask].copy()
    spot = None
    if not df_fut.empty:
        df_fut_t = df_fut[df_fut["minute"]<=time_str]
        if not df_fut_t.empty:
            fut_snap = df_fut_t.sort_values("minute").groupby("ticker")["close"].last()
            near = [tk for tk in fut_snap.index if tk.upper().endswith("-I.NFO")]
            if near:
                spot = float(fut_snap[near[0]])
            else:
                spot = float(fut_snap.iloc[0])
    if spot is None:
        spot = float(chain["strike"].median())

    itv = _detect_interval(chain["strike"].tolist()) or 50
    atm = int(round(spot/itv)*itv)
    chain["atm_dist"] = ((chain["strike"]-atm)/itv).abs()
    if depth > 0:
        chain = chain[chain["atm_dist"]<=depth]
    chain["is_atm"]    = chain["strike"]==atm
    chain["is_itm_ce"] = chain["strike"]<atm
    chain["is_itm_pe"] = chain["strike"]>atm

    return dict(chain=chain, atm=atm, spot=spot, itv=itv,
                expiry_dates=all_exp_dates,
                sel_expiry_date=sel_exp_date)


# ──────────────────────────────────────────────────────────────────────────────
#  DEMO DATA (when no parquet)
# ──────────────────────────────────────────────────────────────────────────────

def make_demo_chain(inst, sel_date, time_str, sel_exp_key, depth):
    np.random.seed(hash(f"{inst}{sel_date}{time_str}") % (2**31))
    spot = DEMO_SPOT.get(inst,1000) + np.random.randint(-50,50)
    itv  = DEMO_ITV.get(inst,20)
    atm  = int(round(spot/itv)*itv)
    # Fake weekly expiry dates (Thu every week + monthly)
    exp_dates = []
    d = sel_date
    while len(exp_dates) < 8:
        if d.weekday() == 3:  # Thursday
            exp_dates.append(d)
        d += timedelta(days=1)
    sel_exp_date = _key_to_exp_date(sel_exp_key)
    if sel_exp_date not in exp_dates:
        sel_exp_date = exp_dates[0]
    n = depth if depth>0 else 15
    rows=[]
    for k in [atm+itv*i for i in range(-n,n+1)]:
        base_oi = 50000*np.exp(-abs(k-atm)/(itv*4))
        ce_oi = int(base_oi*np.random.uniform(0.5,1.5))
        pe_oi = int(base_oi*np.random.uniform(0.5,1.5))
        ce_ltp = max(0.5,(spot-k+np.random.uniform(20,80)) if k<spot
                     else np.exp(-abs(k-spot)/(spot*0.025))*np.random.uniform(10,200))
        pe_ltp = max(0.5,(k-spot+np.random.uniform(20,80)) if k>spot
                     else np.exp(-abs(k-spot)/(spot*0.025))*np.random.uniform(10,200))
        ce_o=round(ce_ltp*np.random.uniform(.92,1.08),2)
        pe_o=round(pe_ltp*np.random.uniform(.92,1.08),2)
        ce_h=round(max(ce_ltp,ce_o)*np.random.uniform(1.01,1.06),2)
        ce_l=round(min(ce_ltp,ce_o)*np.random.uniform(.94,.99),2)
        pe_h=round(max(pe_ltp,pe_o)*np.random.uniform(1.01,1.06),2)
        pe_l=round(min(pe_ltp,pe_o)*np.random.uniform(.94,.99),2)
        rows.append(dict(strike=k,
            ce_ltp=round(ce_ltp,2),ce_oi=ce_oi,ce_oi_base=int(ce_oi*.9),
            ce_open=ce_o,ce_high=ce_h,ce_low=ce_l,ce_vol=int(np.random.exponential(5000)),
            pe_ltp=round(pe_ltp,2),pe_oi=pe_oi,pe_oi_base=int(pe_oi*.9),
            pe_open=pe_o,pe_high=pe_h,pe_low=pe_l,pe_vol=int(np.random.exponential(5000))))
    df=pd.DataFrame(rows)
    df["ce_oi_lakh"]=(df["ce_oi"]/1e5).round(1)
    df["pe_oi_lakh"]=(df["pe_oi"]/1e5).round(1)
    df["ce_oi_chg"]=df["ce_oi"]-df["ce_oi_base"]
    df["pe_oi_chg"]=df["pe_oi"]-df["pe_oi_base"]
    df["ce_chg_pct"]=(df["ce_oi_chg"]/df["ce_oi_base"].clip(lower=1)*100).round(1)
    df["pe_chg_pct"]=(df["pe_oi_chg"]/df["pe_oi_base"].clip(lower=1)*100).round(1)
    df["is_atm"]=df["strike"]==atm
    df["is_itm_ce"]=df["strike"]<atm
    df["is_itm_pe"]=df["strike"]>atm
    return dict(chain=df,atm=atm,spot=spot,itv=itv,
                expiry_dates=exp_dates,sel_expiry_date=sel_exp_date)


def make_demo_ohlcv(ticker, date_str):
    parsed = parse_ticker(ticker)
    inst = parsed.get("inst") or "NIFTY"
    base = DEMO_SPOT.get(inst, 1000)
    if parsed.get("strike"):
        base = max(2, abs(base - parsed["strike"])*0.3 + np.random.uniform(5,60))
    np.random.seed(hash(f"{ticker}{date_str}") % (2**31))
    price = base; rows=[]
    for m in _minutes_list():
        o=price; chg=np.random.normal(0,base*0.002)
        h=o+abs(np.random.normal(0,base*0.003))
        l_=o-abs(np.random.normal(0,base*0.003))
        c=max(0.05,o+chg); h=max(h,o,c); l_=max(0.05,min(l_,o,c))
        vol=int(np.random.exponential(5000))
        rows.append(dict(time=m+":59",open_=round(o,2),high=round(h,2),
                         low=round(l_,2),close=round(c,2),volume=vol,
                         oi=0,minute=m))
        price=c
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────────────────────────────────────

TABLE_CSS = """
<style>
.oc-wrap{overflow-x:auto;border-radius:8px;border:1px solid #e4e0d8;background:#fff;}
.oc-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:0.73rem;}
.oc-table th{font-size:0.6rem;font-weight:600;color:#888;text-transform:uppercase;
  letter-spacing:.04em;padding:5px 6px;text-align:right;
  border-bottom:2px solid #e4e0d8;background:#fafaf7;white-space:nowrap;}
.oc-table th.ce-hdr{background:rgba(211,47,47,0.04);}
.oc-table th.pe-hdr{text-align:left;background:rgba(56,142,60,0.04);}
.oc-table th.sk-hdr{text-align:center;background:#f5f4ef;}
.calls-banner{background:rgba(211,47,47,0.07);color:#c62828;font-size:.65rem;
  font-weight:700;text-align:center;padding:4px;letter-spacing:.1em;
  border-bottom:1px solid rgba(211,47,47,0.14);}
.puts-banner{background:rgba(56,142,60,0.07);color:#2e7d32;font-size:.65rem;
  font-weight:700;text-align:center;padding:4px;letter-spacing:.1em;
  border-bottom:1px solid rgba(56,142,60,0.14);}
.oc-table td{padding:3px 6px;border-bottom:1px solid #f0ede8;
  font-variant-numeric:tabular-nums;white-space:nowrap;}
.oc-table tr:hover td{background:rgba(25,118,210,0.02)!important;}
.ce-td{background:rgba(211,47,47,0.015);text-align:right;}
.ce-itm .ce-td{background:rgba(211,47,47,0.05);}
.pe-td{background:rgba(56,142,60,0.015);text-align:left;}
.pe-itm .pe-td{background:rgba(56,142,60,0.05);}
.atm-row td{background:#fffde7!important;font-weight:600;}
.atm-row .sk-td{border-left:2px solid #f9a825!important;border-right:2px solid #f9a825!important;}
.sk-td{text-align:center;font-weight:600;font-size:.76rem;color:#222;
  background:#fafaf7;border-left:1px solid #e4e0d8;border-right:1px solid #e4e0d8;min-width:65px;}
.ce-ltp{color:#b71c1c;font-weight:600;}
.pe-ltp{color:#1b5e20;font-weight:600;}
.chg-pos{color:#2e7d32;font-size:.68rem;font-weight:500;}
.chg-neg{color:#c62828;font-size:.68rem;font-weight:500;}
.chg-neu{color:#aaa;font-size:.68rem;}
.oi-lakh{color:#444;font-size:.7rem;}
.bar-ce{height:5px;border-radius:3px 0 0 3px;background:#d32f2f;opacity:.7;min-width:1px;}
.bar-pe{height:5px;border-radius:0 3px 3px 0;background:#388e3c;opacity:.7;min-width:1px;}
.atm-badge{display:inline-block;background:#f9a825;color:#fff;border-radius:3px;
  padding:0 3px;font-size:.48rem;font-weight:700;margin-left:3px;vertical-align:middle;}
.ohlc-o{color:#1565c0;font-size:.68rem;}
.ohlc-h{color:#2e7d32;font-size:.68rem;}
.ohlc-l{color:#c62828;font-size:.68rem;}
.ohlc-c{color:#222;font-weight:600;font-size:.72rem;}
.vol-td{color:#888;font-size:.66rem;}
.oi-bar-wrap{display:flex;align-items:center;gap:3px;white-space:nowrap;}
.oi-bar-wrap.right{justify-content:flex-end;}
.oi-bar-wrap.left{flex-direction:row-reverse;justify-content:flex-end;}
.oi-bar-inner{width:50px;display:flex;}
.oi-bar-inner.right{justify-content:flex-end;}
.oi-bar-inner.left{justify-content:flex-start;}
</style>
"""

# ──────────────────────────────────────────────────────────────────────────────
#  RENDER CHAIN TABLE
# ──────────────────────────────────────────────────────────────────────────────

def render_chain_table(data):
    chain = data.get("chain", pd.DataFrame())
    if chain.empty:
        st.info("No option chain data for selected parameters.")
        return
    chain = chain.sort_values("strike",ascending=False).reset_index(drop=True)
    max_ce = max(chain["ce_oi"].max(),1)
    max_pe = max(chain["pe_oi"].max(),1)
    rows_html = ""
    for _,row in chain.iterrows():
        r_cls = ("atm-row" if row.get("is_atm") else
                 "ce-itm" if row.get("is_itm_ce") else
                 "pe-itm" if row.get("is_itm_pe") else "")
        cp = min(int(row["ce_oi"]/max_ce*100),100)
        pp = min(int(row["pe_oi"]/max_pe*100),100)
        atm_b = '<span class="atm-badge">ATM</span>' if row.get("is_atm") else ""
        ce_oi_cell = (f'<div class="oi-bar-wrap right">'
            f'<div class="oi-bar-inner right"><div class="bar-ce" style="width:{cp}%"></div></div>'
            f'<span class="oi-lakh">{row["ce_oi_lakh"]:.1f}L</span></div>')
        pe_oi_cell = (f'<div class="oi-bar-wrap left">'
            f'<div class="oi-bar-inner left"><div class="bar-pe" style="width:{pp}%"></div></div>'
            f'<span class="oi-lakh">{row["pe_oi_lakh"]:.1f}L</span></div>')
        rows_html += (
            f'<tr class="{r_cls}">'
            f'<td class="ce-td">{fmt_chg(row.get("ce_chg_pct",0))}</td>'
            f'<td class="ce-td vol-td">{int(row.get("ce_vol",0)):,}</td>'
            f'<td class="ce-td">{ce_oi_cell}</td>'
            f'<td class="ce-td ohlc-o">{row.get("ce_open",0):,.2f}</td>'
            f'<td class="ce-td ohlc-h">{row.get("ce_high",0):,.2f}</td>'
            f'<td class="ce-td ohlc-l">{row.get("ce_low",0):,.2f}</td>'
            f'<td class="ce-td ce-ltp ohlc-c">{row["ce_ltp"]:,.2f}</td>'
            f'<td class="sk-td">{int(row["strike"]):,}{atm_b}</td>'
            f'<td class="pe-td ohlc-o">{row.get("pe_open",0):,.2f}</td>'
            f'<td class="pe-td ohlc-h">{row.get("pe_high",0):,.2f}</td>'
            f'<td class="pe-td ohlc-l">{row.get("pe_low",0):,.2f}</td>'
            f'<td class="pe-td pe-ltp ohlc-c">{row["pe_ltp"]:,.2f}</td>'
            f'<td class="pe-td">{pe_oi_cell}</td>'
            f'<td class="pe-td vol-td">{int(row.get("pe_vol",0)):,}</td>'
            f'<td class="pe-td">{fmt_chg(row.get("pe_chg_pct",0))}</td>'
            f'</tr>')

    html = (TABLE_CSS +
        '<div class="oc-wrap"><table class="oc-table"><thead>'
        '<tr>'
        '<th colspan="7" class="calls-banner">CALLS</th>'
        '<th class="sk-hdr" rowspan="2" style="padding:0 10px;vertical-align:middle">STRIKE</th>'
        '<th colspan="7" class="puts-banner">PUTS</th></tr>'
        '<tr>'
        '<th class="ce-hdr">OI%</th>'
        '<th class="ce-hdr">Vol</th>'
        '<th class="ce-hdr" style="min-width:90px">OI</th>'
        '<th class="ce-hdr">O</th>'
        '<th class="ce-hdr">H</th>'
        '<th class="ce-hdr">L</th>'
        '<th class="ce-hdr">C</th>'
        '<th class="pe-hdr">O</th>'
        '<th class="pe-hdr">H</th>'
        '<th class="pe-hdr">L</th>'
        '<th class="pe-hdr">C</th>'
        '<th class="pe-hdr" style="min-width:90px">OI</th>'
        '<th class="pe-hdr">Vol</th>'
        '<th class="pe-hdr">OI%</th>'
        '</tr></thead><tbody>' +
        rows_html +
        '</tbody></table></div>')
    st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  OI CHANGE CHART
# ──────────────────────────────────────────────────────────────────────────────

def render_oi_chart(data, df_raw, parsed_map, inst, active_exp_dates,
                    cf, ct, sk_min, sk_max, depth_n, is_demo):
    atm  = data.get("atm",10900)
    spot = data.get("spot")
    itv  = data.get("itv",50)

    if not is_demo and not df_raw.empty:
        inst_mask = df_raw["ticker"].map(lambda tk: parsed_map[tk]["inst"]==inst)
        otype_mask = df_raw["ticker"].map(lambda tk: parsed_map[tk]["otype"] in ("CE","PE"))
        exp_mask = df_raw["ticker"].map(lambda tk: parsed_map[tk].get("exp") in active_exp_dates)
        df_i = df_raw[inst_mask & otype_mask & exp_mask].copy()
        df_i["strike"] = df_i["ticker"].map(lambda tk: parsed_map[tk]["strike"])
        df_i["otype"]  = df_i["ticker"].map(lambda tk: parsed_map[tk]["otype"])
        if sk_min and sk_max:
            df_i = df_i[(df_i["strike"]>=sk_min)&(df_i["strike"]<=sk_max)]
        oi_now  = df_i[df_i["minute"]<=ct].groupby("ticker")["oi"].last()
        oi_base = df_i[df_i["minute"]<=cf].groupby("ticker")["oi"].last()
        oi_chg  = (oi_now-oi_base).fillna(0).reset_index(); oi_chg.columns=["ticker","oi_chg"]
        meta = df_i[["ticker","strike","otype"]].drop_duplicates("ticker")
        merged = pd.merge(oi_chg,meta,on="ticker")
        agg = merged.groupby(["strike","otype"])["oi_chg"].sum().reset_index()
        ce_a = agg[agg["otype"]=="CE"].rename(columns={"oi_chg":"ce_chg"})[["strike","ce_chg"]]
        pe_a = agg[agg["otype"]=="PE"].rename(columns={"oi_chg":"pe_chg"})[["strike","pe_chg"]]
        oi_df = pd.merge(ce_a,pe_a,on="strike",how="outer").fillna(0).sort_values("strike")
        if depth_n and depth_n>0 and atm:
            oi_df = oi_df[((oi_df["strike"]-atm)/itv).abs()<=depth_n]
    else:
        np.random.seed(42)
        strikes = [atm+itv*i for i in range(-10,11)]
        oi_df = pd.DataFrame({"strike":strikes,
            "ce_chg":np.random.normal(0,30000,len(strikes)),
            "pe_chg":np.random.normal(5000,35000,len(strikes))})

    fig = go.Figure()
    fig.add_bar(x=oi_df["strike"],y=(oi_df["pe_chg"]/1e5).round(2),
                name="Put OI chg",marker_color="#43a047",marker_opacity=.82,
                hovertemplate="<b>%{x:,}</b><br>Put: %{y:.2f}L<extra></extra>")
    fig.add_bar(x=oi_df["strike"],y=(oi_df["ce_chg"]/1e5).round(2),
                name="Call OI chg",marker_color="#ef5350",marker_opacity=.82,
                hovertemplate="<b>%{x:,}</b><br>Call: %{y:.2f}L<extra></extra>")
    if spot and atm:
        fig.add_vline(x=atm,line_color="#1565c0",line_dash="dash",line_width=1.5,
                      annotation_text=f"{inst} {spot:,.2f}",
                      annotation_font_color="#1565c0",annotation_font_size=11,
                      annotation_position="top")
    fig.add_hline(y=0,line_color="#ccc",line_width=.8)
    fig.update_layout(
        barmode="group",paper_bgcolor="#fff",plot_bgcolor="#fff",
        font=dict(family="DM Sans",color="#444",size=11),
        legend=dict(orientation="h",x=0,y=-.14,bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#f0ede8",showline=True,linecolor="#e0ddd8",
                   tickfont=dict(size=10),title="Strike"),
        yaxis=dict(gridcolor="#f0ede8",showline=True,linecolor="#e0ddd8",
                   tickfont=dict(size=10),ticksuffix="L",title="OI Change (Lakh)",
                   zeroline=True,zerolinecolor="#bbb"),
        margin=dict(l=50,r=20,t=25,b=65),height=380)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})


# ──────────────────────────────────────────────────────────────────────────────
#  VIEW CHART TAB  — TradingView-style candlestick for any ticker
# ──────────────────────────────────────────────────────────────────────────────

def render_view_chart(df_raw, all_tickers, parsed_map, all_insts,
                      inst, date_str, time_str, expiry_dates, is_demo):

    # ── Global search bar — search ANY ticker across ALL instruments ──
    st.markdown(
        '<div style="font-size:.72rem;color:#666;margin-bottom:4px">'
        '<b>Quick Search</b> — type any ticker name directly '
        '(e.g. <code>RELIANCE-I</code>, <code>NIFTY19JAN10900CE</code>, '
        '<code>SBIN</code>, <code>27000CE</code>)</div>',
        unsafe_allow_html=True)

    global_search = st.text_input(
        "Search all tickers", value="",
        placeholder="Type ticker name to search across ALL instruments...",
        key="vc_global_search", label_visibility="collapsed")

    gs = global_search.strip().upper()
    if gs:
        # Search across ALL tickers, not just one instrument
        matched = [tk for tk in all_tickers if gs in tk.upper()]
        if matched:
            st.markdown(
                f'<div style="font-size:.68rem;color:#2e7d32;margin:2px 0 6px">'
                f'Found <b>{len(matched)}</b> tickers matching "<b>{gs}</b>"</div>',
                unsafe_allow_html=True)

            sel_ticker = st.selectbox(
                "Select from results", matched,
                key="vc_global_result", label_visibility="collapsed")
        else:
            st.warning(f"No tickers matching **'{gs}'**. Try a shorter search term.")
            # Fall through to instrument-based selection below
            gs = ""

    if not gs:
        st.markdown('<hr style="margin:8px 0;border-color:#e4e0d8"/>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:.68rem;color:#888;margin-bottom:4px">'
            'Or browse by instrument:</div>',
            unsafe_allow_html=True)

    # ── Instrument-based selector (used when no global search) ──
    if not gs:
        c1, c2, c3, c4 = st.columns([1.2, 0.9, 0.9, 2.0])
        with c1:
            vc_inst_idx = all_insts.index(inst) if inst in all_insts else 0
            vc_inst = st.selectbox("Instrument", all_insts, index=vc_inst_idx,
                                    key="vc_inst", label_visibility="collapsed")
        with c2:
            type_opts = ["All","Futures Only","CE Only","PE Only"]
            type_sel = st.selectbox("Type", type_opts, key="vc_type",
                                     label_visibility="collapsed")

        # Build ticker list for selected instrument (before expiry filter)
        inst_tickers = [tk for tk in all_tickers if parsed_map[tk]["inst"] == vc_inst]

        # Collect unique expiry dates for this instrument
        _vc_exp_dates = sorted(set(
            parsed_map[tk]["exp"] for tk in inst_tickers
            if parsed_map[tk].get("exp")
        ))
        _vc_sel_date = date.fromisoformat(date_str)
        _vc_exp_display = [_format_exp_date(e, _vc_sel_date) for e in _vc_exp_dates]

        with c3:
            exp_opts = ["All Expiries"] + _vc_exp_display
            exp_sel = st.selectbox("Expiry", exp_opts, key="vc_exp",
                                    label_visibility="collapsed")
        with c4:
            search = st.text_input("Filter", value="",
                                    placeholder=f"e.g. {vc_inst}-I or 10900",
                                    key="vc_search", label_visibility="collapsed")

        # Apply type filter
        if type_sel == "Futures Only":
            inst_tickers = [tk for tk in inst_tickers
                            if parsed_map[tk]["otype"] in ("FUT","FUT_SERIES")]
        elif type_sel == "CE Only":
            inst_tickers = [tk for tk in inst_tickers if parsed_map[tk]["otype"]=="CE"]
        elif type_sel == "PE Only":
            inst_tickers = [tk for tk in inst_tickers if parsed_map[tk]["otype"]=="PE"]

        # Apply expiry filter
        if exp_sel != "All Expiries" and exp_sel in _vc_exp_display:
            _sel_exp_d = _vc_exp_dates[_vc_exp_display.index(exp_sel)]
            inst_tickers = [tk for tk in inst_tickers
                            if parsed_map[tk].get("exp") == _sel_exp_d]

        # Apply text filter
        flt = search.strip().upper()
        if flt:
            inst_tickers = [tk for tk in inst_tickers if flt in tk.upper()]

        st.markdown(f'<div style="font-size:.68rem;color:#888;margin:2px 0 6px">'
                    f'{len(inst_tickers)} tickers for <b>{vc_inst}</b>'
                    f'{" (demo)" if is_demo else ""}</div>',
                    unsafe_allow_html=True)

        if not inst_tickers:
            st.warning(f"No tickers for **{vc_inst}** with current filters.")
            return

        # Default to futures -I
        default_idx = 0
        for i, tk in enumerate(inst_tickers):
            if tk.endswith("-I.NFO"):
                default_idx = i
                break

        sel_ticker = st.selectbox("Ticker", inst_tickers, index=default_idx,
                                   key="vc_sel_ticker", label_visibility="collapsed")

    # Time range
    tc1, tc2 = st.columns(2)
    with tc1:
        t_from = st.text_input("From", "09:15", key="vc_from")
        if not re.match(r'^\d{2}:\d{2}$', t_from): t_from = "09:15"
    with tc2:
        t_to = st.text_input("To", time_str, key="vc_to")
        if not re.match(r'^\d{2}:\d{2}$', t_to): t_to = time_str

    # Load data
    ticker = sel_ticker.strip()
    if not is_demo and not df_raw.empty:
        df_tk = df_raw[df_raw["ticker"]==ticker].copy()
    else:
        df_tk = pd.DataFrame()

    if df_tk.empty:
        df_tk = make_demo_ohlcv(ticker, date_str)

    df_plot = df_tk[(df_tk["minute"]>=t_from)&(df_tk["minute"]<=t_to)].copy()
    if df_plot.empty:
        st.warning(f"No data for **{ticker}** in {t_from}–{t_to}")
        return

    # Parse for subtitle
    p = parse_ticker(ticker)
    parts = [p.get("inst") or vc_inst]
    if p.get("strike"):
        parts.append(f"{p['strike']} {p.get('otype','')}")
    if p.get("exp_label"):
        parts.append(f"Exp: {p['exp_label']}")
    elif p.get("otype") == "FUT_SERIES":
        parts.append(f"Futures {p.get('series','I')}")
    subtitle = " | ".join(parts)

    # Build datetime for x-axis
    df_plot = df_plot.copy()
    df_plot["dt"] = pd.to_datetime(date_str + " " + df_plot["time"].astype(str).str[:8],
                                    errors="coerce")
    if df_plot["dt"].isna().all():
        df_plot["dt"] = pd.to_datetime(date_str + " " + df_plot["minute"] + ":00")

    first_open = float(df_plot["open_"].iloc[0])
    last_close = float(df_plot["close"].iloc[-1])
    day_high   = float(df_plot["high"].max())
    day_low    = float(df_plot["low"].min())
    pnl = last_close - first_open
    pnl_pct = (pnl/first_open*100) if first_open else 0
    pnl_color = "#2e7d32" if pnl>=0 else "#c62828"

    # ── Header bar (like TradingView) ──
    st.markdown(
        f'<div style="background:#fff;border:1px solid #e4e0d8;border-radius:8px;'
        f'padding:10px 14px;margin-bottom:8px;display:flex;align-items:center;'
        f'flex-wrap:wrap;gap:8px 20px">'
        f'<span style="font-size:1rem;font-weight:700;color:#222">{subtitle}</span>'
        f'<span style="font-size:.72rem;color:#888">1m</span>'
        f'<span style="font-size:.72rem;color:#888">NFO</span>'
        f'<span style="font-size:.85rem;font-weight:600;color:#222">C {last_close:,.2f}</span>'
        f'<span style="font-size:.75rem;color:{pnl_color};font-weight:500">'
        f'{"+" if pnl>=0 else ""}{pnl:.2f} ({pnl_pct:+.2f}%)</span>'
        f'<span style="font-size:.68rem;color:#1565c0">O {first_open:,.2f}</span>'
        f'<span style="font-size:.68rem;color:#2e7d32">H {day_high:,.2f}</span>'
        f'<span style="font-size:.68rem;color:#c62828">L {day_low:,.2f}</span>'
        f'</div>',
        unsafe_allow_html=True)

    # ── Candlestick ──
    fig = go.Figure()
    fig.add_candlestick(
        x=df_plot["dt"],
        open=df_plot["open_"],high=df_plot["high"],
        low=df_plot["low"],close=df_plot["close"],
        increasing_line_color="#26a69a",increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350",decreasing_fillcolor="#ef5350",
        name="Price")

    has_vol = "volume" in df_plot.columns and df_plot["volume"].sum() > 0
    if has_vol:
        # Color volume bars by direction
        vol_colors = ["#26a69a" if c >= o else "#ef5350"
                      for o, c in zip(df_plot["open_"], df_plot["close"])]
        fig.add_bar(x=df_plot["dt"],y=df_plot["volume"],
                    name="Volume",marker_color=vol_colors,
                    marker_opacity=0.35,yaxis="y2")

    # Last price line
    fig.add_hline(y=last_close,line_color="#1565c0",line_dash="dot",line_width=1,
                  annotation_text=f"{last_close:,.2f}",
                  annotation_font_color="#1565c0",annotation_font_size=10,
                  annotation_position="right")

    tick_every = max(1, len(df_plot)//20)
    fig.update_layout(
        paper_bgcolor="#fff",plot_bgcolor="#131722",
        font=dict(family="DM Sans",size=11,color="#d1d4dc"),
        xaxis=dict(
            type="category",
            tickvals=df_plot["dt"].iloc[::tick_every].tolist(),
            ticktext=df_plot["minute"].iloc[::tick_every].tolist(),
            gridcolor="#1e222d",showline=False,
            tickfont=dict(size=9,color="#787b86"),
            rangeslider_visible=False),
        yaxis=dict(
            gridcolor="#1e222d",showline=False,side="right",
            tickfont=dict(size=10,color="#787b86")),
        yaxis2=dict(
            overlaying="y",side="left",showgrid=False,
            tickfont=dict(size=8,color="#363a45"),
            range=[0,max(df_plot.get("volume",pd.Series([1])).max()*5,1)]) if has_vol else dict(),
        legend=dict(orientation="h",x=0,y=1.06,
                    bgcolor="rgba(0,0,0,0)",font=dict(size=10,color="#787b86")),
        margin=dict(l=10,r=60,t=10,b=30),
        height=520,hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e222d",bordercolor="#363a45",
                        font=dict(family="DM Mono",size=11,color="#d1d4dc")))
    st.plotly_chart(fig,use_container_width=True,
                    config={"displayModeBar":True,"scrollZoom":True})

    # ── Stats ──
    day_vol = int(df_plot["volume"].sum()) if has_vol else 0
    last_oi = int(df_plot["oi"].iloc[-1]) if "oi" in df_plot.columns else 0
    s1,s2,s3,s4,s5,s6,s7 = st.columns(7)
    with s1: st.metric("Open",f"{first_open:,.2f}")
    with s2: st.metric("High",f"{day_high:,.2f}")
    with s3: st.metric("Low",f"{day_low:,.2f}")
    with s4: st.metric("Close",f"{last_close:,.2f}")
    with s5: st.metric("Change",f"{pnl:+.2f}",delta=f"{pnl_pct:+.1f}%",delta_color="normal")
    with s6: st.metric("Volume",f"{day_vol:,}")
    with s7: st.metric("OI",f"{last_oi:,}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if "inst"      not in st.session_state: st.session_state.inst = "NIFTY"
    if "sel_exp"   not in st.session_state: st.session_state.sel_exp = None
    if "ch_depth"  not in st.session_state: st.session_state.ch_depth = 10

    # ── Global CSS ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
    :root{--bg:#f6f4ef;--surface:#fff;--border:#e4e0d8;--border2:#d0ccc2;
      --text1:#1a1a1a;--text2:#44443e;--text3:#888880;--accent:#1976d2;}
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;}
    .stApp,.main{background:var(--bg)!important;}
    .main .block-container{padding:.75rem 1.1rem!important;max-width:100%!important;}
    p,span,label,div,small,[class*="stMarkdown"],[class*="stCheckbox"] label,
    [class*="stRadio"] label,[class*="stSelectbox"] label,
    [data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] strong{color:var(--text1)!important;}
    .stCheckbox>label{color:var(--text1)!important;font-size:.82rem!important;}
    .stRadio>div{flex-direction:row!important;gap:.4rem!important;flex-wrap:wrap!important;}
    .stRadio>div>label{background:var(--surface)!important;border:1.5px solid var(--border2)!important;
      border-radius:20px!important;padding:.2rem .7rem!important;font-size:.76rem!important;
      color:var(--text1)!important;cursor:pointer!important;}
    .stRadio>div>label:has(input:checked){border-color:var(--accent)!important;
      color:var(--accent)!important;background:#e8f0fe!important;}
    .panel-label{font-size:.63rem;font-weight:700;color:var(--text3);
      text-transform:uppercase;letter-spacing:.08em;margin:10px 0 5px;display:block;}
    .stTabs [data-baseweb="tab-list"]{gap:0;background:var(--surface);
      border-bottom:2px solid var(--border);padding:0;}
    .stTabs [data-baseweb="tab"]{font-size:.82rem!important;font-weight:500!important;
      color:var(--text2)!important;padding:.5rem 1.2rem!important;border-radius:0!important;
      border-bottom:2px solid transparent!important;margin-bottom:-2px!important;
      background:transparent!important;}
    .stTabs [aria-selected="true"]{color:var(--accent)!important;
      border-bottom:2px solid var(--accent)!important;font-weight:600!important;}
    .stTabs [data-baseweb="tab-panel"]{padding:0!important;}
    .stButton>button{font-size:.78rem!important;font-weight:500!important;
      background:var(--surface)!important;border:1.5px solid var(--border2)!important;
      color:var(--text1)!important;border-radius:6px!important;
      padding:.28rem .75rem!important;transition:all .15s!important;
      cursor:pointer!important;opacity:1!important;}
    .stButton>button:hover{border-color:var(--accent)!important;
      color:var(--accent)!important;background:#e8f0fe!important;
      box-shadow:0 1px 4px rgba(25,118,210,0.15)!important;}
    .stButton>button:active{transform:scale(0.97)!important;
      background:#d4e4fc!important;}
    button[kind="primary"]{background:var(--accent)!important;color:#fff!important;
      border-color:var(--accent)!important;font-weight:600!important;}
    button[kind="primary"]:hover{background:#1565c0!important;
      border-color:#1565c0!important;color:#fff!important;}
    .stTextInput>div>div>input,.stDateInput>div>div>input{
      font-family:'DM Mono',monospace!important;font-size:.82rem!important;
      background:var(--surface)!important;border:1.5px solid var(--border2)!important;
      border-radius:6px!important;color:var(--text1)!important;padding:.32rem .6rem!important;}
    .stSelectbox>div>div{font-size:.8rem!important;background:var(--surface)!important;
      border:1.5px solid var(--border2)!important;border-radius:6px!important;
      color:var(--text1)!important;}
    .stNumberInput>div>div>input{font-family:'DM Mono',monospace!important;font-size:.8rem!important;
      background:var(--surface)!important;border:1.5px solid var(--border2)!important;
      border-radius:6px!important;color:var(--text1)!important;}
    [data-testid="stMetricValue"]{font-family:'DM Mono',monospace!important;
      font-size:1.05rem!important;font-weight:600!important;color:var(--text1)!important;}
    [data-testid="stMetricLabel"]{font-size:.58rem!important;color:var(--text3)!important;
      text-transform:uppercase!important;letter-spacing:.07em!important;}
    hr{border-color:var(--border)!important;margin:.35rem 0!important;}
    #MainMenu,footer,header{visibility:hidden;}
    .stDeployButton{display:none;}
    </style>""", unsafe_allow_html=True)

    # Check if we have remote access
    data_exists = bool(HF_TOKEN)
    expiry_df   = load_expiry_df(EXPIRY_CSV)

    # ── Header ──
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:1rem;padding:.45rem 0 .3rem;'
        f'border-bottom:2px solid #e4e0d8;margin-bottom:.7rem">'
        f'<span style="font-size:1.05rem;font-weight:700;color:#1565c0">Option Chain Analytics</span>'
        f'<span style="margin-left:auto;font-size:.63rem;font-weight:600;'
        f'color:{"#2e7d32" if data_exists else "#e65100"}">'
        f'{"Live Data" if data_exists else "Demo Mode"}</span></div>',
        unsafe_allow_html=True)

    # ── Controls ──
    avail_dates = get_available_dates(REPO_ID)
    max_d = avail_dates[-1] if avail_dates else date.today()
    min_d = avail_dates[0]  if avail_dates else date(2015,1,1)

    ctrl1,ctrl2,ctrl3,ctrl4 = st.columns([1.4,.85,.65,.65])
    with ctrl1:
        sel_date = st.date_input("Date",value=max_d,min_value=min_d,max_value=max_d,
                                  label_visibility="collapsed")
    with ctrl2:
        raw_time = st.text_input("HH:MM",value="15:30",placeholder="15:30",
                                  label_visibility="collapsed")
        raw_time = raw_time.strip()
        if not re.match(r'^\d{2}:\d{2}$',raw_time): raw_time="15:30"
        h_i,m_i = int(raw_time[:2]),int(raw_time[3:])
        h_i=max(9,min(15,h_i)); m_i=max(0,min(59,m_i))
        if h_i==9 and m_i<15: m_i=15
        if h_i==15 and m_i>30: m_i=30
        time_str = f"{h_i:02d}:{m_i:02d}"
    with ctrl3:
        depth_opts={"5":5,"10":10,"15":15,"20":20,"All":0}
        depth_sel=st.selectbox("Depth",list(depth_opts.keys()),index=1,
                                label_visibility="collapsed")
        depth=depth_opts[depth_sel]
    with ctrl4:
        show_itm = st.checkbox("ITM",value=True)

    date_str = sel_date.isoformat()

    # ── Load parquet (cached) ──
    pp = parquet_path(OPTIONS_BASE, sel_date)
    df_raw, all_tickers, parsed_map = load_parquet_raw(pp)
    is_demo = df_raw.empty

    if not is_demo:
        all_insts = get_instruments_from_parsed(parsed_map)
    else:
        all_insts = PRIORITY_INSTRUMENTS[:]

    # ── Instrument dropdown ──
    if st.session_state.inst not in all_insts:
        all_insts = [st.session_state.inst] + all_insts
    cur_idx = all_insts.index(st.session_state.inst) if st.session_state.inst in all_insts else 0

    # Get all expiry dates for the currently selected instrument
    if not is_demo:
        _inst_tickers = [tk for tk in all_tickers if parsed_map[tk]["inst"] == st.session_state.inst]
        _inst_exp_dates = sorted(set(
            parsed_map[tk]["exp"] for tk in _inst_tickers
            if parsed_map[tk].get("exp")
        ))
    else:
        # Demo: generate weekly Thursdays
        _inst_exp_dates = []
        d = sel_date
        while len(_inst_exp_dates) < 8:
            if d.weekday() == 3:
                _inst_exp_dates.append(d)
            d += timedelta(days=1)

    # Format expiry dates for dropdown: "17 Feb 22 (0d)"
    _exp_dd_display = [_format_exp_date(e, sel_date) for e in _inst_exp_dates]
    _exp_dd_keys    = [_exp_date_to_key(e) for e in _inst_exp_dates]

    # Instrument + Expiry dropdown on same row
    idd1, idd2 = st.columns([2, 1.5])
    with idd1:
        sel_dd = st.selectbox("Instrument", options=all_insts, index=cur_idx,
                               label_visibility="collapsed", key="inst_dd",
                               help=f"{len(all_insts)} instruments available")
    with idd2:
        # Expiry dropdown with DTE
        if _exp_dd_display:
            cur_exp_key = st.session_state.sel_exp
            if cur_exp_key in _exp_dd_keys:
                exp_dd_idx = _exp_dd_keys.index(cur_exp_key)
            else:
                exp_dd_idx = 0
            sel_exp_display = st.selectbox("Expiry", options=_exp_dd_display,
                                            index=exp_dd_idx,
                                            label_visibility="collapsed", key="exp_dd",
                                            help="Select expiry (days to expiry shown)")
            # Map display back to key
            sel_exp_key = _exp_dd_keys[_exp_dd_display.index(sel_exp_display)]
        else:
            sel_exp_key = None

    if sel_dd != st.session_state.inst:
        st.session_state.inst = sel_dd
        st.session_state.sel_exp = None
        st.rerun()
    if sel_exp_key != st.session_state.sel_exp:
        st.session_state.sel_exp = sel_exp_key
        st.rerun()
    inst = st.session_state.inst

    # ── Build chain ──
    if not is_demo:
        chain_data = build_chain(df_raw, parsed_map, inst, time_str,
                                  st.session_state.sel_exp, depth, sel_date)
        if not chain_data:
            is_demo = True
    if is_demo:
        chain_data = make_demo_chain(inst, sel_date, time_str,
                                      st.session_state.sel_exp, depth)

    expiry_dates  = chain_data.get("expiry_dates",[])
    sel_exp_date  = chain_data.get("sel_expiry_date")
    atm           = chain_data.get("atm")
    spot          = chain_data.get("spot")
    itv           = chain_data.get("itv",50)
    chain_df      = chain_data.get("chain",pd.DataFrame())

    if not show_itm and not chain_df.empty:
        chain_df = chain_df[chain_df["is_atm"]|~(chain_df["is_itm_ce"]|chain_df["is_itm_pe"])].copy()
        chain_data["chain"] = chain_df

    # ── Expiry quick-switch buttons with DTE ──
    if expiry_dates:
        show_n = min(len(expiry_dates),10)
        e_cols = st.columns(show_n)
        for i,ed in enumerate(expiry_dates[:show_n]):
            with e_cols[i]:
                dte = (ed - sel_date).days
                btn_lbl = f"{ed.strftime('%d %b')} ({dte}d)"
                is_sel = (ed == sel_exp_date)
                if st.button(btn_lbl,key=f"exp_{i}",use_container_width=True,
                              type="primary" if is_sel else "secondary"):
                    st.session_state.sel_exp = _exp_date_to_key(ed)
                    st.rerun()

    # ── Stats bar ──
    spot_disp = f"{spot:,.2f}" if spot else "---"
    atm_disp  = f"{atm:,}" if atm else "---"
    if not chain_df.empty:
        tot_ce=chain_df["ce_oi"].sum(); tot_pe=chain_df["pe_oi"].sum()
        pcr=round(tot_pe/max(tot_ce,1),2)
        m1,m2,m3,m4,m5,m6=st.columns(6)
        with m1: st.metric(f"{inst} Spot",spot_disp)
        with m2: st.metric("ATM Strike",atm_disp)
        with m3: st.metric("PCR",f"{pcr}")
        with m4: st.metric("CE OI (L)",f"{tot_ce/1e5:.1f}")
        with m5: st.metric("PE OI (L)",f"{tot_pe/1e5:.1f}")
        with m6: st.metric("Time",time_str)

    st.markdown("<hr/>",unsafe_allow_html=True)

    # ── Tabs ──
    tab1,tab2,tab3 = st.tabs(["Option Chain","OI Change Chart","View Chart"])

    with tab1:
        render_chain_table(chain_data)

    with tab2:
        left,right = st.columns([1,3.5])
        with left:
            st.markdown(
                f'<div style="background:#eef5ff;border:1.5px solid #bbdefb;'
                f'border-radius:8px;padding:9px 12px;margin-bottom:12px">'
                f'<div style="font-size:.6rem;font-weight:700;color:#888;'
                f'text-transform:uppercase;letter-spacing:.08em">SPOT</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:#1565c0;'
                f'font-family:DM Mono,monospace">{spot_disp}</div></div>',
                unsafe_allow_html=True)
            range_mode=st.radio("Range",["Intraday","Custom"],horizontal=True,
                                 label_visibility="collapsed")
            if range_mode=="Custom":
                cf=st.text_input("From","09:15",key="t2f")
                ct=st.text_input("To",time_str,key="t2t")
                if not re.match(r'^\d{2}:\d{2}$',cf): cf="09:15"
                if not re.match(r'^\d{2}:\d{2}$',ct): ct=time_str
            else:
                cf=SESSION_START; ct=time_str
            st.markdown('<span class="panel-label">Expiries</span>',unsafe_allow_html=True)
            active_exp=[]
            for i,ed in enumerate(expiry_dates):
                dte = (ed - sel_date).days
                elbl = f"{ed.strftime('%d %b')} ({dte}d)"
                if st.checkbox(elbl,value=(i==0),key=f"chk_e_{i}"):
                    active_exp.append(ed)
            if not active_exp and expiry_dates: active_exp=[expiry_dates[0]]
            st.markdown('<span class="panel-label">Strike Range</span>',unsafe_allow_html=True)
            sk_def_min=(atm or DEMO_SPOT.get(inst,1000))-10*itv
            sk_def_max=(atm or DEMO_SPOT.get(inst,1000))+10*itv
            sc1,sc2=st.columns(2)
            with sc1: sk_min=st.number_input("Min",value=float(sk_def_min),step=float(itv),key="skn")
            with sc2: sk_max=st.number_input("Max",value=float(sk_def_max),step=float(itv),key="skx")
            st.markdown('<span class="panel-label">Depth</span>',unsafe_allow_html=True)
            d_opts=[("All",0),("5",5),("10",10),("15",15),("20",20)]
            dn=st.columns(len(d_opts))
            for di,(lb,vl) in enumerate(d_opts):
                with dn[di]:
                    if st.button(lb,key=f"dn_{di}",use_container_width=True,
                                  type="primary" if st.session_state.ch_depth==vl else "secondary"):
                        st.session_state.ch_depth=vl; st.rerun()
            ch_depth=st.session_state.ch_depth
            slider_mins=[m for m in _minutes_list() if cf<=m<=ct]
            if slider_mins:
                ct=st.select_slider("Time",options=slider_mins,value=slider_mins[-1],
                                     label_visibility="collapsed")
        with right:
            render_oi_chart(chain_data, df_raw, parsed_map, inst, active_exp,
                            cf, ct, int(sk_min), int(sk_max), ch_depth, is_demo)

    # ── Tab 3: View Chart ──
    with tab3:
        render_view_chart(df_raw, all_tickers, parsed_map, all_insts,
                           inst, date_str, time_str, expiry_dates, is_demo)

    st.markdown(
        '<div style="margin-top:1.2rem;padding-top:.4rem;border-top:1px solid #e4e0d8;'
        'font-size:.56rem;color:#bbb;text-align:center">'
        'Option Chain Dashboard | 1-min Parquet OHLCV+OI</div>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()



