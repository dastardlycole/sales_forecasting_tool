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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from bridge_utils import ensure_schema

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory for saving plots
plots_dir = "final_plots"
os.makedirs(plots_dir, exist_ok=True)

############################################################################
#                            LOADING DATA
############################################################################

def load_data(csv_file, target_col='Net items sold'):
    df = pd.read_csv(csv_file, parse_dates=['Month'], index_col='Month')
    df = df.sort_index()
    df = df.groupby(level=0).sum()
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
        seq_x = data[i:i+lookback, :-1]  # all but last col => features
        seq_y = data[i+lookback, -1]     # last col => target
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def run_sarima_in_sample(df, target_col='Net items sold'):
    train_size = int(len(df)*0.8)
    df_train = df.iloc[:train_size]
    model = sm.tsa.statespace.SARIMAX(
        df_train[target_col],
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit_res = model.fit(disp=False)
    in_sample = fit_res.predict(start=df.index[0], end=df.index[-1])
    in_sample = in_sample.reindex(df.index).fillna(method='bfill')
    return in_sample, fit_res, train_size

def run_lstm_in_sample(df, target_col='Net items sold', lookback=12):
    total_len = len(df)
    train_size = int(total_len*0.8)

    scaler = MinMaxScaler()
    scaled_vals = scaler.fit_transform(df.values)

    X_all, y_all = create_sequences_multicol(scaled_vals, lookback=lookback)
    X_train = X_all[:train_size - lookback]
    y_train = y_all[:train_size - lookback]
    X_val   = X_all[train_size - lookback:]
    y_val   = y_all[train_size - lookback:]

    model = Sequential()
    num_features = X_train.shape[2]
    model.add(LSTM(64, return_sequences=True, input_shape=(lookback, num_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=50, batch_size=16, callbacks=[es], verbose=1)

    val_preds = model.predict(X_val).flatten()
    X_val_last = X_val[:, -1, :]
    recon_preds = np.hstack([X_val_last, val_preds.reshape(-1,1)])
    recon_actual = np.hstack([X_val_last, y_val.reshape(-1,1)])
    inv_scaler = MinMaxScaler()
    inv_scaler.fit(df.values)
    inv_preds  = inv_scaler.inverse_transform(recon_preds)[:, -1]
    inv_actual = inv_scaler.inverse_transform(recon_actual)[:, -1]

    return inv_preds, inv_actual, model, scaler, train_size

############################################################################
#                     FUTURE FORECAST: SARIMA / LSTM
############################################################################

def forecast_sarima_future(df, target_col='Net items sold', steps=12):
    full_model = sm.tsa.statespace.SARIMAX(
        df[target_col],
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit_res = full_model.fit(disp=False)
    fc = fit_res.get_forecast(steps=steps)
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                                 periods=steps, freq='MS')
    return pd.Series(fc.predicted_mean.values, index=future_dates)

def create_sequences_singlecol(data, lookback=12):
    X, y = [], []
    for i in range(len(data) - lookback):
        seq_x = data[i:i+lookback, 0].reshape(lookback,1)
        seq_y = data[i+lookback, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def forecast_lstm_future_singlecol(df, target_col='Net items sold', steps=12, lookback=12):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    X_all, y_all = create_sequences_singlecol(scaled, lookback=lookback)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(lookback,1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_all, y_all, epochs=30, batch_size=16, verbose=0, callbacks=[es])

    init_seq = scaled[-lookback:, :]  # shape (lookback,1)
    init_seq = init_seq[np.newaxis,...]  # shape (1,lookback,1)
    preds = []
    for _ in range(steps):
        p = model.predict(init_seq)[0,0]
        preds.append(p)
        init_seq = np.roll(init_seq, -1, axis=1)
        init_seq[0,-1,0] = p

    preds = np.array(preds).reshape(-1,1)
    future_dates = pd.date_range(df.index[-1]+pd.offsets.MonthBegin(1), periods=steps, freq='MS')
    out = []
    for val in preds:
        row = np.array([[val[0]]])
        unscaled = scaler.inverse_transform(row)
        out.append(unscaled[0,0])
    return pd.Series(np.array(out), index=future_dates)

def forecast_lstm_future_multicol(df, target_col='Net items sold', steps=12, lookback=12):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    def create_seq_mc(data, lb=12):
        X, y = [], []
        for i in range(len(data) - lb):
            x_ = data[i:i+lb, :-1]
            y_ = data[i+lb, -1]
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

    init_seq = scaled[-lookback:, :-1]  # shape (lookback, nfeat)
    init_seq = init_seq[np.newaxis,...]  # (1,lookback,nfeat)
    preds = []
    for _ in range(steps):
        p = model.predict(init_seq)[0,0]
        preds.append(p)
        init_seq = np.roll(init_seq, -1, axis=1)

    preds = np.array(preds).reshape(-1,1)
    future_dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq='MS')
    out = []
    for val in preds:
        row = np.hstack([scaled[-1,:-1], val[0]]).reshape(1,-1)
        unscaled = scaler.inverse_transform(row)
        out.append(unscaled[0,-1])
    return pd.Series(np.array(out), index=future_dates)

############################################################################
#                            MAIN ANALYSIS
############################################################################

def analyze_csv(file_path, target_col='Net items sold', lookback=12,log=None):
    """Load the CSV, run SARIMA and LSTM in-sample, forecast future steps,
    then plot results and print metrics."""
    df = load_data(file_path, target_col=target_col)
    df = ensure_schema(df)

    # ── OPTIONAL DISCOUNT / PROMO COLUMNS ─────────────────────────────
    # If the CSV already contains them we keep them;
    # if not we create zero‑filled columns so code runs unchanged.

    

    # ── AUTOMATIC DISCOUNT FEATURES from the “Discounts” column ───────
    # total discounts in $ is a negative number in the CSV, so flip sign:
    if 'Discounts' in df.columns:
        df['discount_amount'] = -df['Discounts']
    else:
        df['discount_amount'] = 0.0

    # percentage‑off = discount_amount / gross_sales × 100 (safe for zero gross)
    if 'Gross sales' in df.columns:
        df['discount_pct'] = (df['discount_amount'] 
                              / df['Gross sales'].replace(0, np.nan) * 100
                             ).fillna(0.0)
    else:
        df['discount_pct'] = 0.0

    # binary flag: did any discount happen this month?
    df['discount_flag'] = (df['discount_amount'] > 0).astype(int)



    # Seasonality encodings (safe to add even if already present)
    if "sin_month" not in df.columns:
        df["sin_month"] = np.sin(2*np.pi*(df.index.month-1)/12)
    if "cos_month" not in df.columns:
        df["cos_month"] = np.cos(2*np.pi*(df.index.month-1)/12)


    # In-Sample SARIMA
    sarima_in, sarima_fit, train_size = run_sarima_in_sample(df, target_col)
    df['sarima_fitted'] = sarima_in
    df['sarima_resid'] = df[target_col] - df['sarima_fitted']
    sarima_rmse = np.sqrt(mean_squared_error(df[target_col], sarima_in))
    sarima_mae  = mean_absolute_error(df[target_col], sarima_in)

    # In-Sample PURE LSTM
    df_pure = df[[target_col]].dropna()
    pure_preds, pure_actual, _, _, pure_train_size = run_lstm_in_sample(df_pure, target_col, lookback)
    pure_idx = df_pure.index[-len(pure_preds):]
    pure_rmse = np.sqrt(mean_squared_error(df[target_col].reindex(pure_idx), pure_preds))
    pure_mae  = mean_absolute_error(df[target_col].reindex(pure_idx), pure_preds)

    # In-Sample COMBINED LSTM
    df_comb = df[['sarima_fitted', 'sarima_resid',
              'sin_month', 'cos_month',
              'discount_flag', 'discount_amount','discount_pct',      # new features
              target_col]].dropna()
    comb_preds, comb_actual, _, _, comb_train_size = run_lstm_in_sample(df_comb, target_col, lookback)
    comb_idx = df_comb.index[-len(comb_preds):]
    comb_rmse = np.sqrt(mean_squared_error(df[target_col].reindex(comb_idx), comb_preds))
    comb_mae  = mean_absolute_error(df[target_col].reindex(comb_idx), comb_preds)
    def smape(y_true, y_pred):
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_true - y_pred)
        return np.mean(np.divide(diff, denominator, out=np.zeros_like(diff), where=denominator!=0)) * 100
    sarima_smape = smape(df[target_col], sarima_in)
    pure_smape = smape(df[target_col].reindex(pure_idx), pure_preds)
    comb_smape = smape(df[target_col].reindex(comb_idx), comb_preds)

    # Future 12 months
    future_sarima = forecast_sarima_future(df, target_col, steps=12)
    future_pure   = forecast_lstm_future_singlecol(df_pure, target_col, steps=12, lookback=lookback)
    future_comb   = forecast_lstm_future_multicol(df_comb, target_col, steps=12, lookback=lookback)

    # Clamp any negative predictions to zero
    future_sarima = future_sarima.clip(lower=0)
    future_pure   = future_pure.clip(lower=0)
    future_comb   = future_comb.clip(lower=0)


    # Build Series for in-sample LSTM lines
    pure_line = pd.Series(np.nan, index=df.index)
    pure_line.loc[pure_idx] = pure_preds
    comb_line = pd.Series(np.nan, index=df.index)
    comb_line.loc[comb_idx] = comb_preds

    # Plot #1: In-sample lines
    plt.figure(figsize=(10,5))
   
    filename = os.path.splitext(os.path.basename(file_path))[0]
    plt.title(f"{filename} - Historical (in-sample)")

    plt.plot(df.index, df[target_col], label='Actual')
    plt.plot(df.index, sarima_in, label='SARIMA', linestyle='--')
    plt.plot(pure_line.index, pure_line.values, label='Pure LSTM', linestyle='--')
    plt.plot(comb_line.index, comb_line.values, label='Comb. LSTM', linestyle='--')
    plt.axvline(df.index[train_size], color='red', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{os.path.basename(file_path)}_in_sample.png"))
    plt.show()

    # Plot #2: Historical + 12-month future
    plt.figure(figsize=(10,5))
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
    plt.show()

    # print(f"\n===== {file_path} RESULTS =====")
    # print(f"SARIMA -> RMSE: {sarima_rmse:.2f}, MAE: {sarima_mae:.2f}")
    # print(f"Pure   -> RMSE: {pure_rmse:.2f}, MAE: {pure_mae:.2f}")
    # print(f"Comb   -> RMSE: {comb_rmse:.2f}, MAE: {comb_mae:.2f}")

    msg = f"\n===== {file_path} RESULTS =====\n"
    msg += f"SARIMA -> RMSE: {sarima_rmse:.2f}, MAE: {sarima_mae:.2f}\n"
    msg += f"Pure   -> RMSE: {pure_rmse:.2f}, MAE: {pure_mae:.2f}\n"
    msg += f"Comb   -> RMSE: {comb_rmse:.2f}, MAE: {comb_mae:.2f}\n"
    msg += f"SARIMA -> SMAPE: {sarima_smape:.2f}\n"
    msg += f"Pure   -> SMAPE: {pure_smape:.2f}\n"
    msg += f"Comb   -> SMAPE: {comb_smape:.2f}\n"


    if log:
        log(msg)
    else:
        print(msg)

        # === Add this right before the return statement ===

    # 12-Month Forecast Table
    forecast_table = pd.DataFrame({
    "Product": [filename] * len(future_sarima),
    "Month": future_sarima.index.strftime("%Y-%m"),
    "SARIMA": future_sarima.values,
    "Pure_LSTM": future_pure.values,
    "Combined_LSTM": future_comb.values
})

    # Save forecast to CSV
    forecast_csv_path = os.path.join(plots_dir, "forecast_values.csv")
    # forecast_table.to_excel(forecast_csv_path, index=False)
    forecast_table.to_csv(forecast_csv_path, index=False)

    # Optional: show forecast table in log or terminal
    table_msg = "\n📅 12-Month Forecast (units sold):\n"
    table_msg += forecast_table.to_string(index=False)

    if log:
        log(table_msg)
    else:
        print(table_msg)


    
    # Save metrics as a CSV
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

