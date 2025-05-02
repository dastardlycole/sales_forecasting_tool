import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory for saving plots - add this like in doghouse_predictor2.py
plots_dir = "final_plots"
os.makedirs(plots_dir, exist_ok=True)

############################################################################
#                           SYNTHETIC HISTORY HELPER
############################################################################

def make_fake_history_like(real_df,
                           start="2014-01-01",
                           extra_years=6,
                           base=500,
                           trend=2,
                           seasonal_amp=80,
                           disc_lift=0.25,
                           promo_prob=0.10,
                           noise_sd=40):
    """Return a DataFrame that mimics the *numeric* schema of real_df and
    contains engineered exogenous variables (discount_flag, promotion) plus
    synthetic values for 'Net items sold'.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real dataset already indexed by Month.
    start : str
        First month (YYYY-MM-DD) of the synthetic history.
    extra_years : int
        How many *additional* years of data to fabricate.
    Other parameters control the synthetic sales process.
    """
    # Timeline limited so we don't overlap real data
    months = pd.date_range(start, periods=extra_years * 12, freq="MS")
    months = months[months < real_df.index.min()]

    m = months.month                      # 1‒12 for seasonal functions
    rng = np.random.default_rng(42)

    # Engineered flags -----------------------------------------------------
    discount_flag = m.isin([7, 12]).astype(int)
    promotion = rng.choice([0, 1], size=len(months), p=[1 - promo_prob, promo_prob])

    # Synthetic Net items sold --------------------------------------------
    # build a month‑by‑month seasonal fingerprint from the first real year
    month_avg = real_df.groupby(real_df.index.month)['Net items sold'].mean()
    amplitude = (month_avg / month_avg.mean()).values      # length‑12 array

    seasonal_shape = amplitude[m - 1]                      # map each month

    sales = ((base + trend * np.arange(len(months))[::-1])   # upward trend
            * seasonal_shape                                # multiply by pattern
            + rng.normal(0, noise_sd, len(months)))         # add noise
    sales = np.maximum(0, sales).round().astype(int)

    # Build dataframe with all numeric columns found in real_df ------------
    num_cols = [c for c in real_df.columns if pd.api.types.is_numeric_dtype(real_df[c])]
    fake = pd.DataFrame(0, index=months, columns=num_cols)

    # Populate mandatory target column
    fake["Net items sold"] = sales

    # Populate other financial fields if they exist ------------------------
    if "Gross sales" in fake.columns:
        unit_price = 30  # assumption – adjust as needed
        fake["Gross sales"] = fake["Net items sold"] * unit_price
    if "Discounts" in fake.columns:
        fake["Discounts"] = -discount_flag * fake["Gross sales"] * 0.15
    if "Returns" in fake.columns:
        fake["Returns"] = -rng.poisson(0.03, len(months))
    if "Net sales" in fake.columns:
        fake["Net sales"] = fake["Gross sales"] + fake["Discounts"] + fake["Returns"]
    if "Taxes" in fake.columns:
        fake["Taxes"] = fake["Net sales"] * 0.2
    if "Total sales" in fake.columns:
        fake["Total sales"] = fake["Net sales"] + fake["Taxes"]

    # Attach engineered features
    fake["discount_flag"] = discount_flag
    fake["promotion"] = promotion

    fake.index.name = "Month"
    return fake

############################################################################
#                            LOADING DATA
############################################################################

def load_data(csv_file, target_col='Net items sold'):
    df = pd.read_csv(csv_file, parse_dates=['Month'], index_col='Month')
    df = df.sort_index()
    df = df.groupby(level=0).sum()  # sums numeric cols; drops non‑numeric
    df = df.asfreq('MS', method='ffill')
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in {csv_file}.")
    return df

############################################################################
#                         HELPER FUNCTIONS
############################################################################

def create_sequences_multicol(data, lookback=12):
    X, y = [], []
    for i in range(len(data) - lookback):
        seq_x = data[i:i + lookback, :-1]  # all but last col => features
        seq_y = data[i + lookback, -1]  # last col => target
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def run_sarima_in_sample(df, target_col='Net items sold'):
    train_size = int(len(df) * 0.9)
    df_train = df.iloc[:train_size]
    model = sm.tsa.statespace.SARIMAX(
        df_train[target_col],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit_res = model.fit(disp=False)
    in_sample = fit_res.predict(start=df.index[0], end=df.index[-1])
    in_sample = in_sample.reindex(df.index).fillna(method='bfill')
    return in_sample, fit_res, train_size


def create_sequences_singlecol(data, lookback=12):
    X, y = [], []
    for i in range(len(data) - lookback):
        seq_x = data[i:i + lookback, 0].reshape(lookback, 1)
        seq_y = data[i + lookback, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def run_lstm_in_sample(df, target_col='Net items sold', lookback=12):
    total_len = len(df)
    train_size = int(total_len * 0.9)

    split_idx   = int(len(df) * 0.9)        # same train fraction used later
    scaler      = RobustScaler().fit(df.values[:split_idx])
    scaled_vals = scaler.transform(df.values)

    X_all, y_all = create_sequences_multicol(scaled_vals, lookback=lookback)
    X_train = X_all[:train_size - lookback]
    y_train = y_all[:train_size - lookback]
    X_val = X_all[train_size - lookback:]
    y_val = y_all[train_size - lookback:]

    model = Sequential()
    num_features = X_train.shape[2]
    model.add(LSTM(64, return_sequences=True, input_shape=(lookback, num_features)))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(16))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=16, callbacks=[es], verbose=0)

    # Validation predictions back‑scaled -----------------------------------
    val_preds = model.predict(X_val).flatten()
    X_val_last = X_val[:, -1, :]
    recon_preds = np.hstack([X_val_last, val_preds.reshape(-1, 1)])
    recon_actual = np.hstack([X_val_last, y_val.reshape(-1, 1)])

    inv_scaler = RobustScaler()
    inv_scaler.fit(df.values)
    inv_preds = inv_scaler.inverse_transform(recon_preds)[:, -1]
    inv_actual = inv_scaler.inverse_transform(recon_actual)[:, -1]

    return inv_preds, inv_actual, model, scaler, train_size

############################################################################
#                     FUTURE FORECAST: SARIMA / LSTM
############################################################################

def forecast_sarima_future(df, target_col='Net items sold', steps=12):
    full_model = sm.tsa.statespace.SARIMAX(
        df[target_col],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit_res = full_model.fit(disp=False)
    fc = fit_res.get_forecast(steps=steps)
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')
    return pd.Series(fc.predicted_mean.values, index=future_dates)


def forecast_lstm_future_singlecol(df, target_col='Net items sold', steps=12, lookback=12):
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df.values)
    X_all, y_all = create_sequences_singlecol(scaled, lookback=lookback)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_all, y_all, epochs=30, batch_size=16, verbose=0, callbacks=[es])

    init_seq = scaled[-lookback:, :]
    init_seq = init_seq[np.newaxis, ...]
    preds = []
    for _ in range(steps):
        p = model.predict(init_seq)[0, 0]
        preds.append(p)
        init_seq = np.roll(init_seq, -1, axis=1)
        init_seq[0, -1, 0] = p

    preds = np.array(preds).reshape(-1, 1)
    future_dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq='MS')
    out = []
    for val in preds:
        row = np.array([[val[0]]])
        unscaled = scaler.inverse_transform(row)
        out.append(unscaled[0, 0])
    return pd.Series(np.array(out), index=future_dates)


def forecast_lstm_future_multicol(df, target_col='Net items sold', steps=12, lookback=12):
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df.values)

    def create_seq_mc(data, lb=12):
        X, y = [], []
        for i in range(len(data) - lb):
            x_ = data[i:i + lb, :-1]
            y_ = data[i + lb, -1]
            X.append(x_)
            y.append(y_)
        return np.array(X), np.array(y)

    X_all, y_all = create_seq_mc(scaled, lb=lookback)
    nfeat = df.shape[1] - 1

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(lookback, nfeat)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_all, y_all, epochs=30, batch_size=16, verbose=0, callbacks=[es])

    init_seq = scaled[-lookback:, :-1]          # shape (lookback, nfeat)
    init_seq = init_seq[np.newaxis, ...]        # shape (1, lookback, nfeat)

    preds = []

    # 2 · Build feature grid for each of the next `steps` months
    future_dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(1),
                                 periods=steps, freq='MS')

    future_exog = pd.DataFrame(index=future_dates)
    future_exog["sin_month"]     = np.sin(2*np.pi*(future_exog.index.month-1)/12)
    future_exog["cos_month"]     = np.cos(2*np.pi*(future_exog.index.month-1)/12)
    future_exog["discount_flag"] = future_exog.index.month.isin([7, 12]).astype(int)
    hist_p = df['promotion'].mean() if 'promotion' in df else 0.1   # fall back to 10 %
    future_exog['promotion'] = np.random.binomial(1, hist_p, len(future_exog))

    # carry last real discount_pct into every future step
    last_disc = df['discount_pct'].iloc[-1]
    future_exog['discount_pct'] = last_disc


    

    for _, row in future_exog.iterrows():
        # roll window left by 1 step
        init_seq = np.roll(init_seq, -1, axis=1)

        # scale and insert the new exogenous feature vector
        placeholder_target = 0                     # last element is dummy
        # get last known SARIMA values
        last_sarima_fit  = df['sarima_fitted'].iloc[-1]
        last_sarima_res  = 0          # assume zero residual going forward

        # choose placeholders for weather & traffic (carry last observed)
        last_weather     = df['weather_score'].iloc[-1]
        last_traffic     = df['web_traffic'].iloc[-1]

        new_vec = np.hstack([
            last_sarima_fit,           # 1
            last_sarima_res,           # 2
            row['sin_month'],          # 3
            row['cos_month'],          # 4
            row['discount_flag'],      # 5
            row['promotion'],          # 6 ← include promotion here
            row['discount_pct'],       # 7
            last_weather,              # 8
            last_traffic,              # 9
            placeholder_target         # 10 dummy target
        ]).reshape(1, -1)


        init_seq[0, -1, :] = scaler.transform(new_vec)[0, :-1]

        # predict next target
        p = model.predict(init_seq)[0, 0]
        preds.append(p)

        # put the scaled prediction into the feature slot reserved for target
        init_seq[0, -1, -1] = p

    # 3 · Inverse‑transform predictions to original scale
    preds = np.array(preds).reshape(-1, 1)
    out = []
    for val in preds:
        # concatenate with last feature row so inverse_transform works
        row = np.hstack([scaled[-1, :-1], val[0]]).reshape(1, -1)
        unscaled = scaler.inverse_transform(row)
        out.append(unscaled[0, -1])

    return pd.Series(out, index=future_dates)

############################################################################
#                            MAIN ANALYSIS
############################################################################

def analyze_csv(file_path, target_col='Net items sold', lookback=24, log=None):
    """Load CSV, run SARIMA & LSTM in‑sample, forecast future, plot & metrics."""
    df = load_data(file_path, target_col=target_col)

    # ------------------------------------------------------------------
    # Ensure engineered features exist (for real as well as synthetic)
    # ------------------------------------------------------------------
    if 'sin_month' not in df.columns:
        df['sin_month'] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    if 'cos_month' not in df.columns:
        df['cos_month'] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
    if 'discount_flag' not in df.columns:
        df['discount_flag'] = df.index.month.isin([7, 12]).astype(int)
    if 'promotion' not in df.columns:
        df['promotion'] = 0  # assume no promo in the real history unless supplied
    df['discount_pct']   = df.get('discount_pct',   0.0)

    # In‑Sample SARIMA ---------------------------------------------------
    sarima_in, sarima_fit, train_size = run_sarima_in_sample(df, target_col)
    df['sarima_fitted'] = sarima_in
    df['sarima_resid'] = df[target_col] - df['sarima_fitted']
    df['sarima_resid_std']  = df['sarima_resid'] / df['sarima_resid'].std(ddof=0)
    sarima_rmse = np.sqrt(mean_squared_error(df[target_col], sarima_in))
    sarima_mae = mean_absolute_error(df[target_col], sarima_in)

    # In‑Sample pure LSTM (target only) ----------------------------------
    df_pure = df[[target_col]].dropna()
    pure_preds, pure_actual, _, _, _ = run_lstm_in_sample(df_pure, target_col, lookback)
    pure_idx = df_pure.index[-len(pure_preds):]
    pure_rmse = np.sqrt(mean_squared_error(df[target_col].reindex(pure_idx), pure_preds))
    pure_mae = mean_absolute_error(df[target_col].reindex(pure_idx), pure_preds)

    # In‑Sample combined LSTM (SARIMA + exogenous) -----------------------
    df_comb = df[['sarima_fitted', 'sarima_resid_std',
                  'sin_month', 'cos_month', 'discount_flag', 'promotion','discount_pct','weather_score','web_traffic',
                  target_col]].dropna()
    comb_preds, comb_actual, _, _, _ = run_lstm_in_sample(df_comb, target_col, lookback)
    comb_idx = df_comb.index[-len(comb_preds):]
    comb_rmse = np.sqrt(mean_squared_error(df[target_col].reindex(comb_idx), comb_preds))
    comb_mae = mean_absolute_error(df[target_col].reindex(comb_idx), comb_preds)
    
    # Add SMAPE metric calculation for consistency with company version
    def smape(y_true, y_pred):
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_true - y_pred)
        return np.mean(np.divide(diff, denominator, out=np.zeros_like(diff), where=denominator!=0)) * 100
    
    sarima_smape = smape(df[target_col], sarima_in)
    pure_smape = smape(df[target_col].reindex(pure_idx), pure_preds)
    comb_smape = smape(df[target_col].reindex(comb_idx), comb_preds)

    # ------------------------------------------------------------------
    # Future forecast (12 months)
    # ------------------------------------------------------------------
    future_sarima = forecast_sarima_future(df, target_col, steps=12)
    future_pure = forecast_lstm_future_singlecol(df_pure, target_col, steps=12, lookback=lookback)
    future_comb = forecast_lstm_future_multicol(df_comb, target_col, steps=12, lookback=lookback)

    # Clamp any negative predictions to zero (like company version)
    future_sarima = future_sarima.clip(lower=0)
    future_pure = future_pure.clip(lower=0)
    future_comb = future_comb.clip(lower=0)

    # Build Series for in‑sample LSTM lines -----------------------------
    pure_line = pd.Series(np.nan, index=df.index)
    pure_line.loc[pure_idx] = pure_preds
    comb_line = pd.Series(np.nan, index=df.index)
    comb_line.loc[comb_idx] = comb_preds

    # Plot #1: In‑sample
    plt.figure(figsize=(10, 5))
    
    filename = os.path.splitext(os.path.basename(file_path))[0]
    plt.title(f"{filename} - Historical (in-sample)")
    
    plt.plot(df.index, df[target_col], label='Actual')
    plt.plot(df.index, sarima_in, label='SARIMA', linestyle='--')
    plt.plot(pure_line.index, pure_line.values, label='Pure LSTM', linestyle='--')
    plt.plot(comb_line.index, comb_line.values, label='Comb. LSTM', linestyle='--')
    plt.axvline(df.index[train_size], color='red', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plots_dir = "final_plots"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f"{os.path.basename(file_path)}_in_sample.png"))
    plt.close()

    # Plot #2: Historical + forecast
    plt.figure(figsize=(10, 5))
    plt.title(f"{filename} - 12-month Forecast")
    plt.plot(df.index, df[target_col], label='Actual (Hist)')
    plt.plot(df.index, sarima_in, label='SARIMA (Hist)', linestyle='--')
    plt.plot(pure_line.index, pure_line, label='Pure LSTM (Hist)', linestyle='--')
    plt.plot(comb_line.index, comb_line, label='Comb. LSTM (Hist)', linestyle='--')
    plt.plot(future_sarima.index, future_sarima.values, label='SARIMA (Future)', marker='o')
    plt.plot(future_pure.index, future_pure.values, label='Pure LSTM (Future)', marker='o')
    plt.plot(future_comb.index, future_comb.values, label='Comb. LSTM (Future)', marker='o')
    plt.axvline(df.index[train_size], color='red', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{os.path.basename(file_path)}_future.png"))
    plt.close()

    # Format message like company version does with metrics
    msg = f"\n===== {file_path} RESULTS =====\n"
    msg += f"SARIMA -> RMSE: {sarima_rmse:.2f}, MAE: {sarima_mae:.2f}\n"
    msg += f"Pure   -> RMSE: {pure_rmse:.2f}, MAE: {pure_mae:.2f}\n"
    msg += f"Comb   -> RMSE: {comb_rmse:.2f}, MAE: {comb_mae:.2f}\n"
    msg += f"SARIMA -> SMAPE: {sarima_smape:.2f}\n"
    msg += f"Pure   -> SMAPE: {pure_smape:.2f}\n"
    msg += f"Comb   -> SMAPE: {comb_smape:.2f}\n"
    
    # Output metrics using log if provided, otherwise print
    if log:
        log(msg)
    else:
        print(msg)
    
    # 12-Month Forecast Table (like company version)
    forecast_table = pd.DataFrame({
        "Product": [filename] * len(future_sarima),
        "Month": future_sarima.index.strftime("%Y-%m"),
        "SARIMA": future_sarima.values,
        "Pure_LSTM": future_pure.values,
        "Combined_LSTM": future_comb.values
    })

    # Save forecast to CSV
    forecast_csv_path = os.path.join(plots_dir, "forecast_values.csv")
    forecast_table.to_csv(forecast_csv_path, index=False)

    # Optional: show forecast table in log or terminal
    table_msg = "\n📅 12-Month Forecast (units sold):\n"
    table_msg += forecast_table.to_string(index=False)

    if log:
        log(table_msg)
    else:
        print(table_msg)
    
    # Save metrics as a CSV (simplified version for both)
    df_res = pd.DataFrame([{
        "File": file_path,
        "SARIMA_RMSE": sarima_rmse,
        "SARIMA_MAE": sarima_mae,
        "Pure_LSTM_RMSE": pure_rmse,
        "Pure_LSTM_MAE": pure_mae,
        "Comb_LSTM_RMSE": comb_rmse,
        "Comb_LSTM_MAE": comb_mae
    }])
    df_res.to_csv(os.path.join(plots_dir, "performance_summary.csv"), index=False)

    # Return full metrics including SMAPE (like company version)
    return {
        "File": file_path,
        "SARIMA_RMSE": sarima_rmse,
        "SARIMA_MAE": sarima_mae,
        "SARIMA_SMAPE": sarima_smape,
        "Pure_LSTM_RMSE": pure_rmse,
        "Pure_LSTM_MAE": pure_mae,
        "Pure_LSTM_SMAPE": pure_smape,
        "Comb_LSTM_RMSE": comb_rmse,
        "Comb_LSTM_MAE": comb_mae,
        "Comb_LSTM_SMAPE": comb_smape,
    }

############################################################################
#                           ENTRY POINT
############################################################################

if __name__ == "__main__":
    uploads_dir = "uploads"
    plots_dir = "final_plots"
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # List CSV files -----------------------------------------------------
    csv_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in '{uploads_dir}'. Place files and re‑run.")
        sys.exit(1)

    print("Available CSV files in 'uploads':")
    for i, fname in enumerate(csv_files, 1):
        print(f"{i}. {fname}")

    try:
        choice = int(input("\nEnter the number of the file to analyze: "))
        selected_file = csv_files[choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        sys.exit(1)

    selected_path = os.path.join(uploads_dir, selected_file)
    print(f"\nYou selected: {selected_file}\nGenerating synthetic history…")

    # ------------------------------------------------------------------
# Fabricate and stitch  (NEW VERSION)
# ------------------------------------------------------------------
    EXTRA_YEARS = 0

    first_real  = load_data(selected_path, target_col='Net items sold').index.min()
    start_fake  = (first_real - pd.DateOffset(years=EXTRA_YEARS)).strftime("%Y-%m-%d")

    df_real   = load_data(selected_path, target_col='Net items sold')
    real_mean = df_real.iloc[:12]['Net items sold'].mean()

    df_fake = make_fake_history_like(
        df_real,
        start        = start_fake,
        extra_years  = EXTRA_YEARS,
        base         = real_mean * 0.3,
        trend        = (real_mean * 1.1 - real_mean * 0.3) / (EXTRA_YEARS*12 - 1),
        seasonal_amp = real_mean * 0.25
    )

    df_real['discount_flag'] = df_real.get('discount_flag',
                        df_real.index.month.isin([7, 12]).astype(int))
    df_real['promotion']     = df_real.get('promotion', 0)
    df_real['sin_month']     = np.sin(2*np.pi*(df_real.index.month-1)/12)
    df_real['cos_month']     = np.cos(2*np.pi*(df_real.index.month-1)/12)

    df_full  = pd.concat([df_fake, df_real]).sort_index()
    tmp_path = os.path.join(uploads_dir, "tmp_with_synth.csv")
    df_full.to_csv(tmp_path)
    print(f"Synthetic history added: {start_fake} → {(first_real - pd.DateOffset(months=1)):%Y-%m}")


    # Run analysis -------------------------------------------------------
    metrics = analyze_csv(tmp_path)

    # Store metrics ------------------------------------------------------
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(os.path.join(plots_dir, "performance_summary.csv"), index=False)
    print("\nMetrics stored in performance_summary.csv and plots saved to 'final_plots'.")
