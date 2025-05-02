import numpy as np
import pandas as pd

# Required columns across both pipelines (union of all features)
REQ_COLS = {
    'sin_month':        lambda idx: np.sin(2*np.pi*(idx.month-1)/12),
    'cos_month':        lambda idx: np.cos(2*np.pi*(idx.month-1)/12),
    'discount_flag':    0,
    'discount_pct':     0.0,
    'discount_amount':  lambda df: -df['Discounts'] if 'Discounts' in df else 0.0,
    'promotion':        0,
    'weather_score':    0.0,
    'web_traffic':      0.0
}

def ensure_schema(df):
    df = df.copy()
    for col, default in REQ_COLS.items():
        if col not in df.columns:
            if callable(default):
                try:
                    df[col] = default(df.index) if col in ['sin_month', 'cos_month'] else default(df)
                except Exception:
                    df[col] = 0.0  # Fallback if something breaks
            else:
                df[col] = default
    # Reorder: features first, target last (for LSTM prep)
    if 'Net items sold' in df.columns:
        features = [c for c in df.columns if c != 'Net items sold']
        df = df[features + ['Net items sold']]
    return df
