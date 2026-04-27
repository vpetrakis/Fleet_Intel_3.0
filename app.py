import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import math
import traceback
import base64
import warnings
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIES & SETUP
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from xgboost import XGBRegressor
    from sklearn.covariance import LedoitWolf
    from sklearn.model_selection import KFold
    import shap
    HAS_ML = True
except ImportError:
    HAS_ML = False

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="POSEIDON TITAN",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS LOADER
# ═══════════════════════════════════════════════════════════════════════════════
def load_local_css():
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ CSS not found at: {css_path} — Ensure your assets folder is pushed to GitHub.")

load_local_css()

# ═══════════════════════════════════════════════════════════════════════════════
# SVG ASSETS
# ═══════════════════════════════════════════════════════════════════════════════
def _u(s): return f"data:image/svg+xml;base64,{base64.b64encode(s.encode()).decode()}"

LOGO_SVG = base64.b64encode(b'<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="pg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#c9a84c"/><stop offset="50%" stop-color="#00e0b0"/><stop offset="100%" stop-color="#fff"/></linearGradient></defs><circle cx="24" cy="24" r="22" fill="none" stroke="url(#pg)" stroke-width="0.8" opacity=".3"/><path d="M24 6L24 42" stroke="url(#pg)" stroke-width="1.5" stroke-linecap="round"/><path d="M12 24Q24 32 36 24" fill="none" stroke="url(#pg)" stroke-width="1.5" stroke-linecap="round"/></svg>').decode()

ICONS = {
    "VERIFIED":     _u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><circle cx="14" cy="14" r="12" fill="none" stroke="#00e0b0" stroke-width="1" opacity=".2"/><circle cx="14" cy="14" r="7.5" fill="#061a14" stroke="#00e0b0" stroke-width="1.5"/><polyline points="10,14.5 12.8,17 18,10.5" fill="none" stroke="#00e0b0" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'),
    "GHOST BUNKER": _u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><circle cx="14" cy="14" r="12" fill="none" stroke="#ff2a55" stroke-width="1" stroke-dasharray="4 3"/><circle cx="14" cy="14" r="7.5" fill="#1a0508" stroke="#ff2a55" stroke-width="1.5"/><g stroke="#ff2a55" stroke-width="2.5" stroke-linecap="round"><line x1="11" y1="11" x2="17" y2="17"/><line x1="17" y1="11" x2="11" y2="17"/></g></svg>'),
    "STAT OUTLIER": _u('<svg viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="4" width="20" height="20" rx="5" fill="none" stroke="#c9a84c" stroke-width="1.2"/><circle cx="14" cy="14" r="4.5" fill="#0e0a1e" stroke="#c9a84c" stroke-width="1.5"/><circle cx="14" cy="14" r="1.8" fill="#c9a84c"/></svg>')
}

STATUS_COLORS = {
    "VERIFIED":     "#00e0b0",
    "GHOST BUNKER": "#ff2a55",
    "STAT OUTLIER": "#c9a84c"
}

REQUIRED_RAW_COLS = [
    'FO_A', 'FO_L', 'MGO_A', 'MGO_L',
    'Bunk_FO', 'Bunk_MGO', 'Bunk_MELO', 'Bunk_HSCYLO', 'Bunk_LSCYLO', 'Bunk_GELO', 'Bunk_CYLO',
    'MELO_R', 'HSCYLO_R', 'LSCYLO_R', 'GELO_R', 'CYLO_R',
    'Speed', 'DistLeg', 'TotalDist', 'CargoQty', 'Voy', 'Port', 'AD', 'Date', 'Time'
]

# ═══════════════════════════════════════════════════════════════════════════════
# FLEET MASTER DATABASE LOADER
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_fleet_master():
    db_path = Path(__file__).parent / 'fleet_master.csv'
    if db_path.exists():
        try:
            return pd.read_csv(db_path).set_index('Vessel_Name')
        except Exception:
            pass
    return pd.DataFrame(columns=['Min_Speed_kn', 'Ghost_Tol_Sea', 'Ghost_Tol_Port'])

fleet_db = load_fleet_master()

# ═══════════════════════════════════════════════════════════════════════════════
# FORENSIC UTILITIES & LEXICAL SIEVE
# ═══════════════════════════════════════════════════════════════════════════════
def _sn(val):
    """The Lexical Sieve & Nullification Protocol"""
    if pd.isna(val): return np.nan
    s = str(val).strip().upper()
    
    # Predefined Burn List for "Silent Strings" & Odometer "XXX"
    if s in ['NIL', 'N/A', 'NA', 'XXX', 'NONE', 'UNKNOWN', 'BLANK', '-', 'X', '', 'NULL']:
        return np.nan
        
    # Sieve: Extract only numbers and minus signs
    s = re.sub(r'[^\d.\-]', '', s)
    try:
        return float(s) if s and s not in ('.', '-', '-.') else np.nan
    except ValueError:
        return np.nan

def _sn0(val):
    v = _sn(val)
    return 0.0 if np.isnan(v) else v

def _parse_dt(d_val, t_val):
    try:
        if pd.isna(d_val) or str(d_val).strip() == '': return pd.NaT
        ds = str(d_val).strip()
        ds = re.sub(r'20224', '2024', ds)
        ds = re.sub(r'20023', '2023', ds)
        ds = re.sub(
            r'(\d+)\s+([A-Za-z]+)\.?\s+(\d{4})',
            lambda m: f"{m.group(3)}-{m.group(2)[:3]}-{m.group(1).zfill(2)}", ds
        )
        p = pd.to_datetime(ds, errors='coerce')
        if pd.isna(p): return pd.NaT
        d_str = p.strftime('%Y-%m-%d')
        t_str = '00:00'
        if pd.notna(t_val) and str(t_val).strip() != '':
            tr = re.sub(r'[HhLlTtUuCc\s]', '', str(t_val).strip())
            m = re.match(r'^(\d{1,2}):(\d{2})', tr)
            if m: t_str = f"{m.group(1).zfill(2)}:{m.group(2)}"
        return pd.to_datetime(f"{d_str} {t_str}", errors='coerce')
    except Exception:
        return pd.NaT

def compute_dqi(r1, r2, days, phys_burn, drift, ghost_tol):
    if days <= 0 or pd.isna(phys_burn): return 0
    scores = [100.0]
    if phys_burn >= ghost_tol:
        scores.append(100.0)
    else:
        scores.append(max(0.0, 100 - abs(phys_burn) * 5))
    tol = max(30.0, 0.03 * max(_sn0(r1.get('FO_A')), _sn0(r2.get('FO_A'))))
    if tol > 0:
        scores.append(math.exp(-0.5 * ((drift) / tol) ** 2) * 100)
    else:
        scores.append(0.0)
    return int(sum(scores) / len(scores))

# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC INGESTION: RAW TEXT ARMOR & GRAVITY LOCK
# ═══════════════════════════════════════════════════════════════════════════════
def semantic_parse(file_bytes, file_name):
    vn_raw = re.sub(r'\.[^.]+$', '', file_name).strip()
    vname  = re.sub(r'[_\-]+', ' ', vn_raw).upper()

    # RAW TEXT ARMOR: Ingest strictly as str to bypass openpyxl formatting crashes
    if file_name.lower().endswith('.xlsx'):
        df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None, engine='openpyxl', dtype=str)
    else:
        df_raw = pd.read_csv(
            io.StringIO(file_bytes.decode('latin-1', errors='replace')),
            header=None, on_bad_lines='skip', dtype=str
        )

    if df_raw.empty or len(df_raw) < 4:
        raise ValueError("File is empty or severely malformed.")

    # GRAVITY LOCK: Universal Data Density Algorithm
    candidates = []
    for i in range(min(150, len(df_raw))):
        vals = [str(x).upper() for x in df_raw.iloc[i].values if pd.notna(x)]
        if any(k in v for v in vals for k in ['DATE', 'DAY']) and \
           any(k in v for v in vals for k in ['PORT', 'LOC']):
            
            top_header    = df_raw.iloc[i].ffill()
            bottom_header = df_raw.iloc[i + 1] if i + 1 < len(df_raw) else pd.Series([np.nan] * len(df_raw.columns))
            cols_found    = {}

            # Map the columns for this specific candidate
            for j in range(len(df_raw.columns)):
                c1 = str(top_header.iloc[j]).upper().strip()   if pd.notna(top_header.iloc[j])   else ""
                c2 = str(bottom_header.iloc[j]).upper().strip() if pd.notna(bottom_header.iloc[j]) else ""
                c_comb = f"{c1} {c2}".strip()

                if   'VOY'  in c_comb:                             cols_found['Voy']       = j
                elif 'PORT' in c_comb or 'LOC' in c_comb:          cols_found['Port']      = j
                elif 'A/D'  in c_comb or c_comb == 'AD' or 'STATUS' in c_comb: cols_found['AD'] = j
                elif 'SPEED' in c_comb:                            cols_found['Speed']     = j
                elif 'CARGO' in c_comb or 'QTY' in c_comb:         cols_found['CargoQty']  = j
                elif 'DATE'  in c_comb or 'DAY' in c_comb:         cols_found['Date']      = j
                elif 'TIME'  in c_comb and 'TOTAL' not in c_comb:  cols_found['Time']      = j
                elif 'DIST'  in c_comb and 'LEG'   in c_comb:      cols_found['DistLeg']   = j
                elif 'DIST'  in c_comb and 'TOTAL' in c_comb:      cols_found['TotalDist'] = j
                elif 'BUNKER' in c1 or 'RECEIV' in c1:
                    if   'FO'     in c2 and 'MGO' not in c2:       cols_found['Bunk_FO']     = j
                    elif 'MGO'    in c2:                           cols_found['Bunk_MGO']    = j
                    elif 'MELO'   in c2:                           cols_found['Bunk_MELO']   = j
                    elif 'HSCYLO' in c2 or 'HS CYL' in c2:         cols_found['Bunk_HSCYLO'] = j
                    elif 'LSCYLO' in c2 or 'LS CYL' in c2:         cols_found['Bunk_LSCYLO'] = j
                    elif 'CYLO'   in c2 or 'CYL OIL' in c2:        cols_found['Bunk_CYLO']   = j
                    elif 'GELO'   in c2:                           cols_found['Bunk_GELO']   = j
                elif 'ROB' in c1 or 'REMAIN' in c1:
                    if   'FO A'   in c2 or 'FO ACT' in c2:         cols_found['FO_A']      = j
                    elif 'FO L'   in c2 or 'FO LED' in c2:         cols_found['FO_L']      = j
                    elif 'MGO A'  in c2:                           cols_found['MGO_A']     = j
                    elif 'MGO L'  in c2:                           cols_found['MGO_L']     = j
                    elif 'MELO'   in c2:                           cols_found['MELO_R']    = j
                    elif 'HSCYLO' in c2 or 'HS CYL' in c2:         cols_found['HSCYLO_R']  = j
                    elif 'LSCYLO' in c2 or 'LS CYL' in c2:         cols_found['LSCYLO_R']  = j
                    elif 'CYLO'   in c2 or 'CYL OIL' in c2:        cols_found['CYLO_R']    = j
                    elif 'GELO'   in c2:                           cols_found['GELO_R']    = j
            
            # LITMUS TEST: Measure the "Gravity" (Data Density) of this header candidate
            valid_count = 0
            if 'Date' in cols_found:
                date_idx = cols_found['Date']
                time_idx = cols_found.get('Time', -1)
                test_rows = df_raw.iloc[i + 2 : i + 32] # Sample the 30 rows beneath
                for _, row in test_rows.iterrows():
                    d_val = row.iloc[date_idx] if date_idx < len(row) else np.nan
                    t_val = row.iloc[time_idx] if time_idx != -1 and time_idx < len(row) else "00:00"
                    if pd.notna(_parse_dt(d_val, t_val)):
                        valid_count += 1 
            
            candidates.append({'idx': i, 'cols': cols_found, 'gravity': valid_count})

    if not candidates:
        raise ValueError("Matrix Lock Failed: Could not locate any valid 'DATE' and 'PORT' anchor.")

    # Mathematical Winner: The header with the highest valid density
    winner     = max(candidates, key=lambda x: x['gravity'])
    header_idx = winner['idx']
    cols_found = winner['cols']

    df = df_raw.iloc[header_idx + 1:].copy().reset_index(drop=True)
    
    for std_name, exc_idx in cols_found.items():
        df[std_name] = df.iloc[:, exc_idx]

    missing = [col for col in REQUIRED_RAW_COLS if col not in df.columns]
    for req in missing:
        df[req] = np.nan

    # LEXICAL SIEVE 
    MATH_COLS = [
        'FO_A', 'FO_L', 'MGO_A', 'MGO_L', 'Bunk_FO', 'Bunk_MGO', 'Bunk_MELO', 
        'Bunk_HSCYLO', 'Bunk_LSCYLO', 'Bunk_GELO', 'Bunk_CYLO', 'MELO_R', 
        'HSCYLO_R', 'LSCYLO_R', 'GELO_R', 'CYLO_R', 'Speed', 'DistLeg', 
        'TotalDist', 'CargoQty'
    ]
    for col in MATH_COLS:
        df[col] = df[col].apply(_sn)

    # STRUCTURAL ANCHOR: Prevent float/str crashes
    STRING_COLS = ['Voy', 'Port', 'AD', 'Date', 'Time']
    for col in STRING_COLS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df['Datetime'] = df.apply(lambda r: _parse_dt(r.get('Date'), r.get('Time')), axis=1)
    df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
    df['AD'] = df['AD'].apply(
        lambda v: 'D' if v.upper() in ['D','DEP','SBE','FAOP']
        else ('A' if v.upper().startswith('A') else v)
    )
    return df, vname

# ═══════════════════════════════════════════════════════════════════════════════
# TRI-STATE AD-TO-AD STATE MACHINE (KINEMATIC IMPUTATION PROTOCOL)
# ═══════════════════════════════════════════════════════════════════════════════
def build_state_machine(df, min_speed, ghost_sea, ghost_port):
    ad_events = df[df['AD'].isin(['A', 'D'])].copy()
    if len(ad_events) < 2:
        raise ValueError("Insufficient A/D events to construct a timeline.")

    ad_events['Prev_AD'] = ad_events['AD'].shift(1)
    ad_events = ad_events[ad_events['AD'] != ad_events['Prev_AD']].drop(columns=['Prev_AD']).copy()

    trips, cum_drift = [], []
    for i in range(len(ad_events) - 1):
        r1, r2   = ad_events.iloc[i], ad_events.iloc[i + 1]
        idx1, idx2 = r1.name, r2.name
        status, flags = 'VERIFIED', []
        phys_burn, log_burn, drift, daily_burn, days = np.nan, np.nan, np.nan, np.nan, 0.0

        phase = 'SEA' if r1['AD'] == 'D' else 'PORT'
        days  = (r2['Datetime'] - r1['Datetime']).total_seconds() / 86400.0
        if days <= 0:
            days = 0.02
            flags.append("Time Delta Fallback Applied")

        start_rob, end_rob = r1.get('FO_A'), r2.get('FO_A')
        if pd.isna(start_rob) or pd.isna(end_rob):
            status = 'QUARANTINE_ROB'
            flags.append("Missing Physical Tank Sounding")

        if r1['AD'] == 'D' and not pd.isna(start_rob):
            fol = r1.get('FO_L')
            cum_drift.append({
                'dt':   r1['Datetime'],
                'gap':  start_rob - (fol if not pd.isna(fol) else start_rob),
                'port': r1.get('Port', '')[:20]
            })

        window = df.loc[idx1 + 1:idx2]
        
        # Safe aggregation (MATH_COLS are guaranteed floats)
        if phase == 'PORT':
            bfo      = df.loc[idx1:idx2, 'Bunk_FO'].sum(skipna=True)
            b_melo   = df.loc[idx1:idx2, 'Bunk_MELO'].sum(skipna=True)
            b_hscylo = df.loc[idx1:idx2, 'Bunk_HSCYLO'].sum(skipna=True)
            b_lscylo = df.loc[idx1:idx2, 'Bunk_LSCYLO'].sum(skipna=True)
            b_cylo   = df.loc[idx1:idx2, 'Bunk_CYLO'].sum(skipna=True)
            b_gelo   = df.loc[idx1:idx2, 'Bunk_GELO'].sum(skipna=True)
        else:
            bfo      = window['Bunk_FO'].sum(skipna=True)
            b_melo   = window['Bunk_MELO'].sum(skipna=True)
            b_hscylo = window['Bunk_HSCYLO'].sum(skipna=True)
            b_lscylo = window['Bunk_LSCYLO'].sum(skipna=True)
            b_cylo   = window['Bunk_CYLO'].sum(skipna=True)
            b_gelo   = window['Bunk_GELO'].sum(skipna=True)

        speed = window['Speed'].replace(0, np.nan).mean() if not window['Speed'].empty else np.nan
        dist  = window['DistLeg'].sum(skipna=True)

        # THE "XXX" ODOMETER FALLBACK (Kinematic Imputation)
        if dist <= 0 and phase == 'SEA':
            dist = max(0, _sn0(r2.get('TotalDist')) - _sn0(r1.get('TotalDist')))
            
            # If both are missing ("XXX" -> 0), impute physically from speed and time
            if dist <= 0 and not pd.isna(speed):
                dist = speed * (days * 24.0)
                flags.append("Distance Imputed from Speed/Time Kinematics")

        if pd.isna(speed):
            speed = dist / (days * 24.0) if days > 0 else 0.0

        melo_c     = max(0, (_sn0(r1.get('MELO_R'))   - _sn0(r2.get('MELO_R')))   + b_melo)
        hscylo_c   = max(0, (_sn0(r1.get('HSCYLO_R')) - _sn0(r2.get('HSCYLO_R'))) + b_hscylo)
        lscylo_c   = max(0, (_sn0(r1.get('LSCYLO_R')) - _sn0(r2.get('LSCYLO_R'))) + b_lscylo)
        cylo_gen_c = max(0, (_sn0(r1.get('CYLO_R'))   - _sn0(r2.get('CYLO_R')))   + b_cylo)
        gelo_c     = max(0, (_sn0(r1.get('GELO_R'))   - _sn0(r2.get('GELO_R')))   + b_gelo)

        dqi = 0
        if status == 'VERIFIED' or 'QUARANTINE' not in status: 
            phys_burn  = (start_rob - end_rob) + bfo
            log_start  = r1.get('FO_L') if not pd.isna(r1.get('FO_L')) else start_rob
            log_end    = r2.get('FO_L') if not pd.isna(r2.get('FO_L')) else end_rob
            log_burn   = (log_start - log_end) + bfo
            drift      = phys_burn - log_burn
            daily_burn = phys_burn / days

            # TRIANGULATION PROTOCOL: Physics Constraints
            if bfo < 0:
                status = 'QUARANTINE'
                flags.append("Negative Bunker Input Detected")
            
            if abs(drift) > 20 and abs(abs(drift) - abs(bfo)) < 5.0:
                status = 'QUARANTINE'
                flags.append("Mass Imbalance (Drift-Mirror Detection)")
                
            if daily_burn > 250:
                status = 'QUARANTINE'
                flags.append("Thermodynamic Ceiling Exceeded (MCR Limit)")

            if phase == 'PORT' and phys_burn < ghost_port and 'QUARANTINE' not in status:
                status = 'GHOST BUNKER'
                flags.append("Missing Port Bunker Receipt")
            elif phase == 'SEA' and phys_burn < ghost_sea and 'QUARANTINE' not in status:
                status = 'GHOST BUNKER'
                flags.append("Negative Sea Burn Impossibility")

            dqi = compute_dqi(
                r1, r2, days, phys_burn, drift,
                ghost_tol=(ghost_port if phase == 'PORT' else ghost_sea)
            )

        trips.append({
            'Indicator':    ICONS.get(status, ICONS['VERIFIED']) if 'QUARANTINE' not in status else '⛔',
            'Timeline':     f"{r1['Datetime'].strftime('%d %b %y')} → {r2['Datetime'].strftime('%d %b %y')}",
            'Date_Start_TS': r1['Datetime'],
            'Phase':        phase,
            'Condition':    'LADEN' if _sn0(r1.get('CargoQty', 0)) > 100 else 'BALLAST',
            'Voy':          r1.get('Voy', ''),
            'Route':        f"{r1.get('Port','')[:15]} → {r2.get('Port','')[:15]}"
                            if phase == 'SEA' else f"Port Idle: {r1.get('Port','')[:15]}",
            'Days':         round(days, 2),
            'Dist_NM':      round(dist, 0),
            'Speed_kn':     round(speed, 1),
            'CargoQty':     _sn0(r1.get('CargoQty', 0)),
            'FO_A_Start':   start_rob  if status == 'VERIFIED' else np.nan,
            'Bunk_FO':      bfo,
            'FO_A_End':     end_rob    if status == 'VERIFIED' else np.nan,
            'Phys_Burn':    round(phys_burn,  1),
            'Log_Burn':     round(log_burn,   1),
            'Drift_MT':     round(drift,      1),
            'Daily_Burn':   round(daily_burn, 1) if status == 'VERIFIED' else np.nan,
            'MELO_L':       round(melo_c,     0),
            'HSCYLO_L':     round(hscylo_c,   0),
            'LSCYLO_L':     round(lscylo_c,   0),
            'CYLO_GEN_L':   round(cylo_gen_c, 0),
            'GELO_L':       round(gelo_c,     0),
            'Total_CYLO':   round(hscylo_c + lscylo_c + cylo_gen_c, 0),
            'DQI':          int(dqi),
            'Status':       status,
            'Flags':        ', '.join(flags) if flags else ''
        })

    trip_df = pd.DataFrame(trips)

    if len(trip_df) >= 4:
        for cond in ['LADEN', 'BALLAST']:
            ver = trip_df[
                (trip_df['Status']    == 'VERIFIED') &
                (trip_df['Phase']     == 'SEA') &
                (trip_df['Phys_Burn'] >  0) &
                (trip_df['Condition'] == cond)
            ]
            if len(ver) >= 4:
                q1, q3 = ver['Daily_Burn'].quantile(0.25), ver['Daily_Burn'].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 2.0 * iqr, q3 + 2.0 * iqr
                    mask = (
                        (trip_df['Status']    == 'VERIFIED') &
                        (trip_df['Phase']     == 'SEA') &
                        (trip_df['Condition'] == cond) &
                        ((trip_df['Daily_Burn'] < lo) | (trip_df['Daily_Burn'] > hi))
                    )
                    trip_df.loc[mask, 'Status']    = 'STAT OUTLIER'
                    trip_df.loc[mask, 'Indicator'] = ICONS['STAT OUTLIER']

    return trip_df, cum_drift

# ═══════════════════════════════════════════════════════════════════════════════
# FULL DATA-DRIVEN PIML — Ledoit-Wolf + K-Fold Conformal
# ═══════════════════════════════════════════════════════════════════════════════
def execute_ai_physics(trip_df, min_speed):
    ai_status_msg = "Enterprise AI Optimized."
    if not HAS_ML:
        return trip_df, "AI Offline: Missing scikit-learn or xgboost."
    if trip_df.empty:
        return trip_df, "AI Offline: Empty ledger."

    for col in ['AI_Exp','HM_Base','Stoch_Var','SHAP_Base','SHAP_Propulsion','SHAP_Mass',
                'SHAP_Kinematics','SHAP_Season','SHAP_Degradation',
                'Exp_Lower','Exp_Upper','Mahalanobis','MD_Threshold','P_Value']:
        if col not in trip_df.columns:
            trip_df[col] = np.nan

    try:
        sea_mask = (
            (trip_df['Phase']    == 'SEA') &
            (trip_df['Status']   == 'VERIFIED') &
            (trip_df['Speed_kn'] >= min_speed)
        )
        if sea_mask.sum() < 8:
            raise ValueError(f"Insufficient valid Sea Legs ({sea_mask.sum()}). Minimum 8 required.")

        ml = trip_df.loc[sea_mask].copy()
        
        ml['True_Mass']     = (ml['CargoQty'].fillna(0) + ml['FO_A_Start'].fillna(0)).clip(lower=0.1)
        ml['SOG']           = ml['Dist_NM'] / np.maximum(ml['Days'] * 24, 0.1)
        ml['Kin_Delta']     = (ml['Speed_kn'] - ml['SOG']).clip(-3.0, 3.0)
        ml['Accel_Penalty'] = ml['Speed_kn'].diff().fillna(0.0).clip(-2.0, 2.0)
        ml['Speed_Cubed']   = ml['Speed_kn'] ** 3
        ml['Season_Sin']    = np.sin(2 * np.pi * ml['Date_Start_TS'].dt.month.fillna(6) / 12.0)
        ml['Season_Cos']    = np.cos(2 * np.pi * ml['Date_Start_TS'].dt.month.fillna(6) / 12.0)

        epoch = trip_df['Date_Start_TS'].min()
        ml['Days_Since_Epoch'] = (ml['Date_Start_TS'] - epoch).dt.total_seconds() / 86400.0

        features      = ['Speed_kn','Speed_Cubed','True_Mass','Kin_Delta','Accel_Penalty',
                         'Season_Sin','Season_Cos','Days_Since_Epoch']
        maha_features = ['Speed_kn','True_Mass','Accel_Penalty',
                         'Season_Sin','Season_Cos','Days_Since_Epoch']
        ml[features]  = ml[features].fillna(0.0)

        # Admiralty-anchored baseline
        k_array  = ml['Daily_Burn'] / ((ml['True_Mass'] ** (2/3)) * ml['Speed_Cubed'] + 1e-6)
        q25      = np.percentile(k_array, 25)
        best_k   = np.median(k_array[k_array <= q25])
        ml['HM_Base'] = best_k * (ml['True_Mass'] ** (2/3)) * ml['Speed_Cubed']
        trip_df.loc[sea_mask, 'HM_Base'] = ml['HM_Base']

        y_delta = ml['Daily_Burn'] - ml['HM_Base']
        X_train = ml[features]
        weights = ml['Days'].clip(0.1, 30.0)
        if y_delta.var() < 0.05:
            raise ValueError("Target variance too low.")

        # K-Fold cross-conformal calibration
        kf         = KFold(n_splits=min(5, len(X_train)), shuffle=True, random_state=42)
        oof_preds  = np.zeros(len(X_train))
        for train_idx, val_idx in kf.split(X_train):
            m = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.06, random_state=42)
            m.fit(X_train.iloc[train_idx], y_delta.iloc[train_idx],
                  sample_weight=weights.iloc[train_idx])
            oof_preds[val_idx] = m.predict(X_train.iloc[val_idx])

        oof_residuals = np.abs(y_delta - oof_preds)

        # Full-data final model
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.06, random_state=42)
        model.fit(X_train, y_delta, sample_weight=weights)
        preds = ml['HM_Base'] + model.predict(X_train)

        # Heteroscedastic variance model
        var_model = XGBRegressor(n_estimators=40, max_depth=2, learning_rate=0.05, random_state=42)
        var_model.fit(X_train, oof_residuals, sample_weight=weights)
        var_preds_train  = np.maximum(var_model.predict(X_train), 0.01)
        conformal_scores = oof_residuals / var_preds_train

        n     = len(conformal_scores)
        q_val = min(1.0, np.ceil((n + 1) * 0.90) / n) if n > 0 else 0.90
        q90   = np.quantile(conformal_scores, q_val)
        stoch_margin = np.maximum(var_model.predict(X_train) * q90, 0.5)

        p_vals = []
        for i, row_idx in enumerate(ml.index):
            current_score     = np.abs(ml.loc[row_idx, 'Daily_Burn'] - preds.iloc[i]) / var_preds_train[i]
            prob_less_extreme = np.sum(conformal_scores <= current_score) / len(conformal_scores)
            p_vals.append((1.0 - prob_less_extreme) * 100)
        trip_df.loc[sea_mask, 'P_Value'] = p_vals

        # Ledoit-Wolf Mahalanobis
        X_maha = ml[maha_features].values
        lw     = LedoitWolf().fit(X_maha)
        md     = np.sqrt(np.maximum(lw.mahalanobis(X_maha), 0))
        trip_df.loc[sea_mask, 'Mahalanobis']  = md
        trip_df.loc[sea_mask, 'MD_Threshold'] = np.percentile(md, 95)

        explainer = shap.TreeExplainer(model)
        sv        = explainer.shap_values(X_train)
        base_val  = explainer.expected_value[0] \
                    if isinstance(explainer.expected_value, np.ndarray) \
                    else explainer.expected_value

        trip_df.loc[sea_mask, 'AI_Exp']           = preds.round(1)
        trip_df.loc[sea_mask, 'Stoch_Var']        = stoch_margin.round(1)
        trip_df.loc[sea_mask, 'SHAP_Base']        = base_val
        trip_df.loc[sea_mask, 'SHAP_Propulsion']  = sv[:, 0] + sv[:, 1]
        trip_df.loc[sea_mask, 'SHAP_Mass']        = sv[:, 2]
        trip_df.loc[sea_mask, 'SHAP_Kinematics']  = sv[:, 3] + sv[:, 4]
        trip_df.loc[sea_mask, 'SHAP_Season']      = sv[:, 5] + sv[:, 6]
        trip_df.loc[sea_mask, 'SHAP_Degradation'] = sv[:, 7]
        trip_df.loc[sea_mask, 'Exp_Lower']        = preds - stoch_margin
        trip_df.loc[sea_mask, 'Exp_Upper']        = preds + stoch_margin

        outlier_mask = sea_mask & (
            (trip_df['Daily_Burn'] < trip_df['Exp_Lower']) |
            (trip_df['Daily_Burn'] > trip_df['Exp_Upper'])
        )
        trip_df.loc[outlier_mask, 'Status'] = 'STAT OUTLIER'

    except ValueError as e:
        ai_status_msg = f"AI Offline: {str(e)}"
    except Exception as e:
        ai_status_msg = f"AI Critical Exception: {str(e)}"
        print(traceback.format_exc())

    return trip_df, ai_status_msg

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes, filename, min_speed, ghost_sea, ghost_port):
    try:
        parsed_df, vname   = semantic_parse(file_bytes, filename)
        trip_df, cum_drift = build_state_machine(parsed_df, min_speed, ghost_sea, ghost_port)
        trip_df, ai_msg    = execute_ai_physics(trip_df, min_speed)

        quarantined = len(trip_df[trip_df['Status'].str.contains('QUARANTINE')])
        valid_sea   = trip_df[(trip_df['Phase'] == 'SEA') & (trip_df['Status'] == 'VERIFIED')]
        avg_sea     = valid_sea['Phys_Burn'].sum() / valid_sea['Days'].sum() \
                      if valid_sea['Days'].sum() > 0 else 0.0

        trip_df['Total_CYLO'] = (
            trip_df.get('HSCYLO_L',  pd.Series([0], dtype=float)) +
            trip_df.get('LSCYLO_L',  pd.Series([0], dtype=float)) +
            trip_df.get('CYLO_GEN_L',pd.Series([0], dtype=float))
        )

        summary = {
            'vname':       vname,
            'integrity':   round((len(trip_df) - quarantined) / len(trip_df) * 100, 1)
                           if not trip_df.empty else 0,
            'avg_dqi':     round(trip_df['DQI'].mean(), 0) if not trip_df.empty else 0,
            'total_fuel':  round(trip_df['Phys_Burn'].sum(skipna=True), 1),
            'avg_sea_burn':round(avg_sea, 1),
            'total_nm':    round(trip_df['Dist_NM'].sum(), 0),
            'total_days':  round(trip_df['Days'].sum(), 1),
            'total_melo':  round(trip_df.get('MELO_L', pd.Series([0])).sum(), 0),
            'total_cylo':  round(trip_df['Total_CYLO'].sum(), 0),
            'cycles':      len(trip_df),
            'quarantined': quarantined,
            'anomalies':   len(trip_df[trip_df['Status'].isin(['GHOST BUNKER','STAT OUTLIER'])]),
            'ai_msg':      ai_msg
        }
        return trip_df, summary, cum_drift, None
    except ValueError as e:
        return pd.DataFrame(), None, None, f"Parsing Rejected: {str(e)}"
    except Exception as e:
        return pd.DataFrame(), None, None, f"System Crash: {str(e)}"

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY RENDER ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
_BL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor="rgba(6,12,18,0.97)",
        bordercolor="rgba(0,224,176,0.55)",
        font=dict(family='Geist Mono', color='#f8fafc', size=13)
    ),
    font=dict(family='Hanken Grotesk', color='#f8fafc'),
    transition=dict(duration=800, easing='cubic-in-out')
)
_M = dict(l=15, r=15, t=85, b=30)
_AX = dict(
    gridcolor='rgba(255,255,255,0.02)',
    zerolinecolor='rgba(255,255,255,0.05)',
    tickfont=dict(family='Geist Mono', size=11, color='#475569'),
    showspikes=True,
    spikecolor="rgba(0,224,176,0.6)",
    spikethickness=1,
    spikedash="solid"
)

def chart_fuel(df):
    sea  = df[(df['Phase'] == 'SEA')  & (~df['Status'].str.contains('QUARANTINE'))]
    port = df[(df['Phase'] == 'PORT') & (~df['Status'].str.contains('QUARANTINE'))]
    fig  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.7, 0.3], vertical_spacing=0.08)
    if not sea.empty:
        fig.add_trace(go.Bar(
            x=sea['Timeline'], y=sea['Phys_Burn'], name='Sea Fuel',
            marker_color='rgba(0,224,176,0.15)',
            marker_line_color='#00e0b0', marker_line_width=1.5
        ), row=1, col=1)
    if not port.empty:
        fig.add_trace(go.Bar(
            x=port['Timeline'], y=port['Phys_Burn'], name='Port Fuel',
            marker_color='rgba(255,42,85,0.15)',
            marker_line_color='#ff2a55', marker_line_width=1.5
        ), row=1, col=1)
    if not sea.empty:
        fig.add_trace(go.Scatter(
            x=sea['Timeline'], y=sea['Daily_Burn'], name='Sea MT/day',
            mode='lines+markers',
            line=dict(color='#00e0b0', width=3, shape='spline'),
            fill='tozeroy', fillcolor='rgba(0,224,176,0.05)',
            marker=dict(size=8, color="#051014", line=dict(color="#00e0b0", width=2))
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sea['Timeline'], y=sea['Speed_kn'], name='Sea Speed',
            mode='lines+markers',
            line=dict(color='#c9a84c', width=3, shape='spline'),
            fill='tozeroy', fillcolor='rgba(201,168,76,0.05)',
            marker=dict(size=8, color="#051014", line=dict(color="#c9a84c", width=2))
        ), row=2, col=1)
    fig.update_layout(
        **_BL,
        margin=_M,
        title=dict(text='Tri-State Fuel Consumption & Kinematics',
                   font=dict(size=24, family='Bricolage Grotesque', color="#fff")),
        barmode='group', showlegend=True, height=700,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.update_xaxes(tickangle=-45, automargin=True, **_AX)
    fig.update_yaxes(**_AX)
    return fig

def chart_lube(df):
    fig = go.Figure()
    if df.get('MELO_L',    pd.Series([0])).sum() > 0:
        fig.add_trace(go.Bar(x=df['Timeline'], y=df['MELO_L'],
            name='MELO',      marker_color='rgba(0,224,176,0.15)',
            marker_line_color='#00e0b0', marker_line_width=1.5))
    if df.get('Total_CYLO',pd.Series([0])).sum() > 0:
        fig.add_trace(go.Bar(x=df['Timeline'], y=df['Total_CYLO'],
            name='CYLO (All)',marker_color='rgba(255,42,85,0.15)',
            marker_line_color='#ff2a55', marker_line_width=1.5))
    if df.get('GELO_L',    pd.Series([0])).sum() > 0:
        fig.add_trace(go.Bar(x=df['Timeline'], y=df['GELO_L'],
            name='GELO',      marker_color='rgba(201,168,76,0.15)',
            marker_line_color='#c9a84c', marker_line_width=1.5))
    fig.update_layout(
        **_BL,
        margin=_M,
        title=dict(text='Lubricant Consumption (Liters)',
                   font=dict(size=24, family='Bricolage Grotesque', color="#fff")),
        barmode='group', showlegend=True, height=500,
        yaxis=dict(title='L', **_AX),
        xaxis=dict(automargin=True, **_AX)
    )
    fig.update_xaxes(tickangle=-45)
    return fig

def chart_cum_drift(cum_drift):
    if not cum_drift: return None
    cdf = pd.DataFrame(cum_drift)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cdf['dt'], y=cdf['gap'], mode='lines+markers',
        name='A−L Gap',
        line=dict(color='#c9a84c', width=3),
        marker=dict(size=8, color="#051014", line=dict(color="#c9a84c", width=2)),
        fill='tozeroy', fillcolor='rgba(201,168,76,0.08)'
    ))
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.15)', width=1))
    fig.update_layout(
        **_BL,
        margin=_M,
        title=dict(text='Physical vs Logged Mass Drift',
                   font=dict(size=24, family='Bricolage Grotesque', color="#fff")),
        height=500,
        yaxis=dict(title='FO_A − FO_L (MT)', **_AX),
        xaxis=dict(automargin=True, **_AX)
    )
    fig.update_xaxes(tickangle=-45)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# HUD RENDERER
# ═══════════════════════════════════════════════════════════════════════════════
def render_hud(sum_data):
    svg_fuel  = '<svg viewBox="0 0 24 24"><path d="M12 2c-5.33 4.55-8 8.48-8 11.8 0 4.98 3.8 8.2 8 8.2s8-3.22 8-8.2c0-3.32-2.67-7.25-8-11.8zM12 20c-3.35 0-6-2.57-6-6.2 0-2.34 1.95-5.44 6-9.14 4.05 3.7 6 6.79 6 9.14 0 3.63-2.65 6.2-6 6.2z"/></svg>'
    svg_speed = '<svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/></svg>'
    svg_lube  = '<svg viewBox="0 0 24 24"><path d="M19.36 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.36 8.04A5.994 5.994 0 0 0 4 20h14c3.31 0 6-2.69 6-6 0-3.15-2.44-5.74-5.64-5.96z"/></svg>'
    svg_alert = '<svg viewBox="0 0 24 24"><path d="M12 2L1 21h22M12 6l7.53 13H4.47M11 10v4h2v-4m-2 6v2h2v-2"/></svg>'
    svg_lock  = '<svg viewBox="0 0 24 24"><path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zM9 6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9V6zm9 14H6V10h12v10zm-6-3c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2z"/></svg>'

    w_class = " hud-warn" if sum_data['anomalies']   > 0 else ""
    q_class = " hud-warn" if sum_data['quarantined'] > 0 else ""

    html = f"""
    <div class="hud-grid">
        <div class="hud-card">
            <div class="hud-header">
                <div class="hud-title">Verified Fuel</div>
                <div class="hud-icon">{svg_fuel}</div>
            </div>
            <div class="hud-val">{sum_data['total_fuel']:,.1f}</div>
            <div class="hud-sub">Metric Tons</div>
        </div>
        <div class="hud-card">
            <div class="hud-header">
                <div class="hud-title">Avg Sea Burn</div>
                <div class="hud-icon">{svg_speed}</div>
            </div>
            <div class="hud-val">{sum_data['avg_sea_burn']:.1f}</div>
            <div class="hud-sub">MT / Day</div>
        </div>
        <div class="hud-card">
            <div class="hud-header">
                <div class="hud-title">Total MELO</div>
                <div class="hud-icon">{svg_lube}</div>
            </div>
            <div class="hud-val">{int(sum_data['total_melo']):,}</div>
            <div class="hud-sub">Liters</div>
        </div>
        <div class="hud-card">
            <div class="hud-header">
                <div class="hud-title">Total CYLO</div>
                <div class="hud-icon">{svg_lube}</div>
            </div>
            <div class="hud-val">{int(sum_data['total_cylo']):,}</div>
            <div class="hud-sub">Liters</div>
        </div>
        <div class="hud-card{w_class}">
            <div class="hud-header">
                <div class="hud-title">Anomalies</div>
                <div class="hud-icon">{svg_alert}</div>
            </div>
            <div class="hud-val" style="color:{'#ff2a55' if sum_data['anomalies'] > 0 else '#fff'}">{sum_data['anomalies']}</div>
            <div class="hud-sub">Flagged Deviations</div>
        </div>
        <div class="hud-card{q_class}">
            <div class="hud-header">
                <div class="hud-title">Quarantined</div>
                <div class="hud-icon">{svg_lock}</div>
            </div>
            <div class="hud-val" style="color:{'#ff2a55' if sum_data['quarantined'] > 0 else '#fff'}">{sum_data['quarantined']}</div>
            <div class="hud-sub">Missing Data Legs</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FRONTEND
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <div class="hero-left">
        <img src="data:image/svg+xml;base64,{LOGO_SVG}" class="hero-logo" alt=""/>
        <div class="hero-title-container">
            <div class="hero-title">POSEIDON TITAN</div>
            <div class="hero-sub">Enterprise Forensic Engine</div>
        </div>
    </div>
    <div class="hero-badge">
        <span style="color:#fff">KERNEL</span>&ensp;Ledoit-Wolf Matrix<br>
        <span style="color:#fff">PIPELINE</span>&ensp;Cinematic VDOM<br>
        <span style="color:var(--acc)">BUILD</span>&ensp;v10.0.0 The Zenith Edition
    </div>
</div>
""", unsafe_allow_html=True)

files = st.file_uploader(
    'Upload vessel telemetry',
    accept_multiple_files=True,
    type=['xlsx', 'csv'],
    label_visibility='collapsed'
)

if not files:
    st.info("Drop vessel noon-report files to execute the Multi-Dimensional Forensic Audit.")
    st.stop()

fleet_results = []
for f in files:
    with st.spinner(f'Auditing {f.name}…'):
        file_bytes = f.getvalue()
        try:
            _, vname = semantic_parse(file_bytes, f.name)
            if vname in fleet_db.index:
                v_props   = fleet_db.loc[vname]
                min_speed = float(v_props.get('Min_Speed_kn',  4.0))
                ghost_sea = float(v_props.get('Ghost_Tol_Sea', -3.0))
                ghost_port= float(v_props.get('Ghost_Tol_Port',-5.0))
            else:
                min_speed, ghost_sea, ghost_port = 4.0, -3.0, -5.0

            trip_df, sum_data, cum_drift, err = run_pipeline(
                file_bytes, f.name, min_speed, ghost_sea, ghost_port
            )
        except Exception as e:
            err     = f"Initialization Error: {str(e)}"
            trip_df = pd.DataFrame()

    if err:
        st.error(f"**Rejected {f.name}:** {err}")
        continue
    if trip_df.empty:
        st.warning(f"No valid events extracted from {f.name}. Check template schema.")
        continue

    fleet_results.append({'name': sum_data['vname'], 'summary': sum_data, 'df': trip_df})

    ic = STATUS_COLORS['VERIFIED']     if sum_data['integrity'] >= 80 else \
         STATUS_COLORS['STAT OUTLIER'] if sum_data['integrity'] >= 50 else \
         STATUS_COLORS['GHOST BUNKER']

    # Vessel header card
    r, g, b = int(ic[1:3],16), int(ic[3:5],16), int(ic[5:7],16)
    st.markdown(f"""
    <div class="vcard" style="
        background: var(--glass-bg);
        backdrop-filter: blur(40px);
        border: 1px solid var(--glass-border);
        border-radius: var(--r);
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: var(--glass-shadow);">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <div style="font-family:var(--fd);font-weight:800;font-size:1.8rem;
                            color:#fff;letter-spacing:-0.03em;">
                    {sum_data['vname']}
                </div>
                <div style="font-family:var(--fm);font-size:.63rem;color:var(--t2);
                            margin-top:6px;letter-spacing:0.08em">
                    {sum_data['cycles']} LEGS&ensp;·&ensp;{sum_data['total_days']:.0f} DAYS&ensp;·&ensp;{int(sum_data['total_nm']):,} NM
                </div>
                <div style="font-family:var(--fm);font-size:.58rem;color:var(--acc2);
                            margin-top:5px;letter-spacing:0.04em">
                    {sum_data['ai_msg']}
                </div>
            </div>
            <div style="text-align:right">
                <div style="font-family:var(--fd);font-weight:800;font-size:2.8rem;
                            color:{ic};line-height:1;
                            text-shadow:0 0 28px rgba({r},{g},{b},0.45);">
                    {sum_data['integrity']:.0f}%
                </div>
                <div style="font-family:var(--fm);font-size:.58rem;color:var(--t2);
                            text-transform:uppercase;letter-spacing:.15em;margin-top:6px">
                    Audit Integrity
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_hud(sum_data)
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        'IMMUTABLE LEDGER', 'COMMERCIAL P&L',
        'LUBE & DRIFT',     'AI DIGITAL TWIN',
        'FORENSIC PROOF',   'QUARANTINE LOG'
    ])

    # ── Tab 1 · Immutable Ledger ──────────────────────────────────────────────
    with t1:
        st.markdown(
            '<div style="margin-bottom:14px">'
            '<span style="font-family:var(--fm);font-size:.72rem;color:var(--acc);'
            'background:rgba(0,224,176,0.05);padding:6px 12px;border-radius:6px;'
            'border:1px solid rgba(0,224,176,0.14);">'
            '[START ROB] + [BUNKERS] − [END ROB] = [PHYSICAL BURN]'
            '</span></div>',
            unsafe_allow_html=True
        )
        dcfg = {
            'Indicator':  st.column_config.ImageColumn(' '),
            'Timeline':   st.column_config.TextColumn('TIMELINE',   width='medium'),
            'Phase':      st.column_config.TextColumn('LEG'),
            'Days':       st.column_config.NumberColumn('DAYS',      format='%.2f'),
            'Speed_kn':   st.column_config.NumberColumn('SPD',       format='%.1f'),
            'FO_A_Start': st.column_config.NumberColumn('START ROB', format='%.1f'),
            'Bunk_FO':    st.column_config.NumberColumn('+ BUNKERS', format='%.1f'),
            'FO_A_End':   st.column_config.NumberColumn('− END ROB', format='%.1f'),
            'Phys_Burn':  st.column_config.NumberColumn('= BURN',    format='%.1f'),
            'Log_Burn':   st.column_config.NumberColumn('LOG BURN',  format='%.1f'),
            'DQI':        st.column_config.ProgressColumn('DQI', format='%d', min_value=0, max_value=100),
            'Daily_Burn': st.column_config.NumberColumn('MT/DAY',    format='%.1f'),
            'Total_CYLO': st.column_config.NumberColumn('CYLO (ALL)',format='%d'),
            'Status':     st.column_config.TextColumn('STATUS',      width='medium'),
        }
        st.dataframe(
            trip_df[['Indicator','Timeline','Phase','Days','Speed_kn',
                     'FO_A_Start','Bunk_FO','FO_A_End','Phys_Burn',
                     'Log_Burn','Drift_MT','Daily_Burn',
                     'Total_CYLO','MELO_L','GELO_L','DQI','Status']],
            column_config=dcfg, hide_index=True,
            use_container_width=True, height=500
        )
        buf = io.BytesIO()
        exp = trip_df.drop(columns=['Indicator','Date_Start_TS'], errors='ignore')
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            exp.to_excel(w, index=False, sheet_name='Audit')
        buf.seek(0)
        st.download_button(
            'Export Tri-State Ledger', data=buf,
            file_name=f"{sum_data['vname'].replace(' ','_')}_LEDGER.xlsx",
            key=f"dl_{sum_data['vname']}"
        )

    # ── Tab 2 · Commercial P&L ────────────────────────────────────────────────
    with t2:
        voy = (
            trip_df[~trip_df['Status'].str.contains('QUARANTINE')]
            .groupby('Voy', dropna=False)
            .agg(
                Total_Fuel  =('Phys_Burn', 'sum'),
                Sea_Days    =('Days',      lambda x: x[trip_df.loc[x.index,'Phase']=='SEA'].sum()),
                Port_Days   =('Days',      lambda x: x[trip_df.loc[x.index,'Phase']=='PORT'].sum()),
                Sea_Fuel    =('Phys_Burn', lambda x: x[trip_df.loc[x.index,'Phase']=='SEA'].sum()),
                Bunkers     =('Bunk_FO',   'sum'),
                Dist        =('Dist_NM',   'sum'),
                HSCYLO      =('HSCYLO_L',  'sum'),
                LSCYLO      =('LSCYLO_L',  'sum'),
            )
            .reset_index()
        )
        voy['Sea MT/Day'] = np.where(voy['Sea_Days'] > 0, voy['Sea_Fuel'] / voy['Sea_Days'], 0.0)
        st.dataframe(voy, hide_index=True, use_container_width=True)

    # ── Tab 3 · Lube & Drift ─────────────────────────────────────────────────
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            if trip_df.get('MELO_L', pd.Series([0])).sum() + \
               trip_df.get('Total_CYLO', pd.Series([0])).sum() > 0:
                st.plotly_chart(chart_lube(trip_df),
                                use_container_width=True, config={'displayModeBar': False})
            else:
                st.info('No lubricant consumption data detected.')
        with c2:
            if cum_drift:
                st.plotly_chart(chart_cum_drift(cum_drift),
                                use_container_width=True, config={'displayModeBar': False})

    # ── Tab 4 · AI Digital Twin ───────────────────────────────────────────────
    with t4:
        st.plotly_chart(chart_fuel(trip_df),
                        use_container_width=True, config={'displayModeBar': False})
        sea_df = trip_df[(trip_df['Phase'] == 'SEA') & (trip_df['Status'] == 'VERIFIED')]
        if 'AI_Exp' in sea_df.columns and sea_df['AI_Exp'].abs().sum() > 0:
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=sea_df['Timeline'].tolist() + sea_df['Timeline'].tolist()[::-1],
                y=sea_df['Exp_Upper'].tolist() + sea_df['Exp_Lower'].tolist()[::-1],
                fill='toself', fillcolor='rgba(123,104,238,0.14)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip', name='90% Conformal Interval'
            ))
            fig_c.add_trace(go.Scatter(
                x=sea_df['Timeline'], y=sea_df['AI_Exp'],
                name='Expected Mean',
                line=dict(color='#7b68ee', width=2, dash='dash')
            ))
            fig_c.add_trace(go.Scatter(
                x=sea_df['Timeline'], y=sea_df['Daily_Burn'],
                name='Audited Burn', mode='lines+markers',
                line=dict(color='#00e0b0', width=3),
                marker=dict(size=8, color='#051014', line=dict(color='#00e0b0', width=2))
            ))
            fig_c.update_layout(
                **_BL,
                margin=_M,
                title=dict(text='Conformal Propulsion Bounds (Verified Sea Legs)',
                           font=dict(size=22, family='Bricolage Grotesque', color='#fff')),
                height=700,
                yaxis=dict(title='MT/day', **_AX),
                xaxis=dict(tickangle=-45, automargin=True, **_AX)
            )
            st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar': False})

    # ── Tab 5 · Forensic Proof ────────────────────────────────────────────────
    with t5:
        sea = trip_df[(trip_df['Phase'] == 'SEA') & (trip_df['Status'] == 'VERIFIED')]
        if 'HM_Base' in sea.columns and sea['HM_Base'].abs().sum() > 0:
            sel = st.selectbox(
                'Select Verified Sea Passage',
                sea['Timeline'].tolist(),
                key=f'shap_{sum_data["vname"]}'
            )
            tr = sea[sea['Timeline'] == sel].iloc[0]
            eb = tr['AI_Exp']

            # SHAP waterfall
            fig_w = go.Figure(go.Waterfall(
                name="SHAP", orientation="v",
                measure=["absolute","relative","relative","relative",
                         "relative","relative","relative","total"],
                x=["Robust Baseline","Fleet Bias","Res. Speed","Mass",
                   "Kinematics","Season Spline","Degradation","AI Expected"],
                textposition="outside",
                text=[f"{tr['HM_Base']:.1f}", f"{tr['SHAP_Base']:+.1f}",
                      f"{tr['SHAP_Propulsion']:+.1f}", f"{tr['SHAP_Mass']:+.1f}",
                      f"{tr['SHAP_Kinematics']:+.1f}", f"{tr['SHAP_Season']:+.1f}",
                      f"{tr['SHAP_Degradation']:+.1f}", f"{eb:.1f}"],
                y=[tr['HM_Base'], tr['SHAP_Base'], tr['SHAP_Propulsion'],
                   tr['SHAP_Mass'], tr['SHAP_Kinematics'], tr['SHAP_Season'],
                   tr['SHAP_Degradation'], 0],
                connector={"line": {"color": "rgba(255,255,255,0.08)", "width": 2, "dash": "dot"}},
                decreasing={"marker": {"color": "#00e0b0"}},
                increasing={"marker": {"color": "#ff2a55"}},
                totals={"marker": {"color": "#7b68ee"}}
            ))
            fig_w.update_layout(
                **_BL, height=500,
                title=dict(
                    text=f"Mathematical Delta Breakdown: {tr['Route']} ({tr['Speed_kn']}kn)",
                    font=dict(size=20, family='Bricolage Grotesque', color='#fff')
                ),
                yaxis=dict(**_AX),
                margin=dict(t=80, b=30, l=10, r=10)
            )
            st.plotly_chart(fig_w, use_container_width=True, config={'displayModeBar': False})

            # PDF chart
            sigma   = max(tr['Stoch_Var'] / 1.645, 0.1)
            x_vals  = np.linspace(eb - 4 * sigma, eb + 4 * sigma, 100)
            y_vals  = np.exp(-0.5 * ((x_vals - eb) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            x_fill  = np.linspace(tr['Exp_Lower'], tr['Exp_Upper'], 50)
            y_fill  = np.exp(-0.5 * ((x_fill - eb) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(
                x=np.concatenate([x_fill, x_fill[::-1]]),
                y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
                fill='toself', fillcolor='rgba(123,104,238,0.18)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip', showlegend=False
            ))
            fig_stoch.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='lines',
                line=dict(color='rgba(123,104,238,0.9)', width=3),
                showlegend=False
            ))
            fig_stoch.add_trace(go.Scatter(
                x=[eb, eb], y=[0, max(y_vals)], mode='lines',
                line=dict(color='#7b68ee', width=2, dash='dash'),
                showlegend=False
            ))
            fig_stoch.add_trace(go.Scatter(
                x=[eb], y=[max(y_vals) * 1.15], mode='text',
                text=['AI Mean'],
                textfont=dict(color='#7b68ee', family='Geist Mono', size=13),
                showlegend=False
            ))
            actual_color = '#00e0b0' \
                if (tr['Daily_Burn'] >= tr['Exp_Lower'] and tr['Daily_Burn'] <= tr['Exp_Upper']) \
                else '#ff2a55'
            y_actual = np.exp(-0.5 * ((tr['Daily_Burn'] - eb) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            fig_stoch.add_trace(go.Scatter(
                x=[tr['Daily_Burn'], tr['Daily_Burn']], y=[0, y_actual], mode='lines',
                line=dict(color=actual_color, width=2), showlegend=False
            ))
            fig_stoch.add_trace(go.Scatter(
                x=[tr['Daily_Burn']], y=[y_actual + max(y_vals) * 0.15],
                mode='markers+text',
                marker=dict(color=actual_color, size=14, symbol='diamond'),
                text=['Actual'],
                textfont=dict(color=actual_color, family='Geist Mono', size=13),
                textposition='top center', showlegend=False
            ))
            fig_stoch.update_layout(
                **_BL,
                title=dict(text='Empirical Probability Density (Cross-Conformal)',
                           font=dict(size=18, family='Bricolage Grotesque', color='#fff')),
                height=400,
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                xaxis=dict(title='MT/day', **_AX),
                margin=dict(t=70, b=40, l=20, r=20)
            )
            st.plotly_chart(fig_stoch, use_container_width=True, config={'displayModeBar': False})

            p_val = tr['P_Value']
            if p_val < 5.0:
                st.error(
                    f"**Forensic Proof:** The Audited Burn falls at the absolute tail of the "
                    f"Conformal distribution. Empirical probability of natural occurrence: "
                    f"**{p_val:.2f}%**. High probability of physical anomaly or mass extraction."
                )
            else:
                st.success(
                    f"**Forensic Proof:** The Audited Burn is statistically nominal. "
                    f"Empirical probability of natural occurrence: **{p_val:.2f}%**."
                )

            # Mahalanobis gauge
            md_val    = tr['Mahalanobis']
            md_thresh = tr['MD_Threshold']
            md_color  = '#00e0b0' if md_val <= md_thresh else '#ff2a55'
            fig_md = go.Figure(go.Indicator(
                mode="number+gauge", value=md_val,
                number={'font': {'color': md_color, 'size': 45, 'family': 'Bricolage Grotesque'}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': 'Kinematic Plausibility Matrix',
                       'font': {'size': 18, 'color': '#f8fafc', 'family': 'Bricolage Grotesque'}},
                gauge={
                    'axis': {'range': [None, max(md_val, md_thresh) * 1.2],
                             'tickwidth': 2, 'tickcolor': 'rgba(255,255,255,0.35)'},
                    'bar':  {'color': md_color, 'thickness': 0.3},
                    'bgcolor': 'rgba(255,255,255,0.04)', 'borderwidth': 0,
                    'steps': [
                        {'range': [0, md_thresh],                          'color': 'rgba(0,224,176,0.14)'},
                        {'range': [md_thresh, max(md_val,md_thresh)*1.2],  'color': 'rgba(255,42,85,0.14)'}
                    ],
                    'threshold': {'line': {'color': '#fff', 'width': 3},
                                  'thickness': 0.82, 'value': md_thresh}
                }
            ))
            fig_md.update_layout(**_BL, height=300, margin=dict(t=60, b=20, l=30, r=30))
            st.plotly_chart(fig_md, use_container_width=True, config={'displayModeBar': False})

            if md_val <= md_thresh:
                st.success(
                    f"**Kinematic Audit: PASS.** The engine confirms this report — "
                    f"Mahalanobis distance ({md_val:.1f}) is within the historical "
                    f"threshold ({md_thresh:.1f})."
                )
            else:
                st.error(
                    f"⚠️ **Kinematic Audit: FAIL.** Mahalanobis distance ({md_val:.1f}) "
                    f"exceeds the statistical threshold ({md_thresh:.1f}). "
                    f"The reported kinematic inputs are inconsistent with historical baseline."
                )
        else:
            st.warning("AI Explainability Offline: Minimum 8 Sea Legs required.")

    # ── Tab 6 · Quarantine Log ────────────────────────────────────────────────
    with t6:
        quar = trip_df[trip_df['Status'].str.contains('QUARANTINE|GHOST')]
        if quar.empty:
            st.success("Zero anomalies. All timelines and mass balances intact.")
        else:
            for _, r in quar.iterrows():
                c = STATUS_COLORS.get(r['Status'], '#ff2a55')
                st.markdown(
                    f'<div class="q-card">'
                    f'<span style="color:{c};font-weight:800;font-size:.78rem;'
                    f'letter-spacing:.1em">{r["Status"]}</span>'
                    f'<span style="color:var(--t2);margin-left:14px;font-size:.78rem;'
                    f'font-family:var(--fm)">{r["Timeline"]}</span>'
                    f'<div style="color:var(--t1);font-size:.82rem;margin-top:8px;'
                    f'font-weight:500">Exception: {r["Flags"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# FLEET COMPARISON MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
if len(fleet_results) > 1:
    st.markdown(
        '<h2 style="color:#fff;font-family:var(--fd);margin-top:20px;font-size:2rem">'
        'Fleet Comparison Matrix</h2>',
        unsafe_allow_html=True
    )
    fleet_rows = [{
        'Vessel':    r['name'],
        'Legs':      r['summary']['cycles'],
        'Verified':  f"{r['summary']['integrity']:.1f}%",
        'DQI':       int(r['summary']['avg_dqi']),
        'Fuel MT':   r['summary']['total_fuel'],
        'Sea Burn':  r['summary']['avg_sea_burn'],
        'Anomalies': r['summary']['anomalies'],
        'NM':        int(r['summary']['total_nm'])
    } for r in fleet_results]
    st.dataframe(
        pd.DataFrame(fleet_rows),
        hide_index=True, use_container_width=True, height=350
    )
