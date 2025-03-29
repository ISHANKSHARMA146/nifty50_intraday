import sqlite3
import pandas as pd
import numpy as np
import ast

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def compute_sma(series, window):
    return series.rolling(window=window).mean()

def compute_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def compute_macd(close, span_short=12, span_long=26, span_signal=9):
    ema_short = compute_ema(close, span_short)
    ema_long = compute_ema(close, span_long)
    macd_line = ema_short - ema_long
    macd_signal = compute_ema(macd_line, span_signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(close, window=20):
    sma = compute_sma(close, window)
    std = close.rolling(window=window).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return sma, upper_band, lower_band

def compute_atr(df, window=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def add_features(df):
    df = flatten_columns(df)
    # Ensure we have a 'Close' column (check for lowercase 'close' or 'adj close')
    columns_lower = {col.lower(): col for col in df.columns}
    if 'close' in columns_lower:
        df.rename(columns={columns_lower['close']: 'Close'}, inplace=True)
    elif 'adj close' in columns_lower:
        df.rename(columns={columns_lower['adj close']: 'Close'}, inplace=True)
    else:
        raise KeyError("No 'Close' column found")
    
    # Compute technical indicators on daily data
    df['SMA_20'] = compute_sma(df['Close'], 20)
    df['SMA_50'] = compute_sma(df['Close'], 50)
    df['EMA_20'] = compute_ema(df['Close'], 20)
    df['EMA_50'] = compute_ema(df['Close'], 50)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['Close'])
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = compute_bollinger_bands(df['Close'], 20)
    df['ATR_14'] = compute_atr(df, 14)
    
    # Create target labels:
    # OpenTarget: 1 if next day's Open > today's Open, else 0.
    # CloseTarget: 1 if next day's Close > today's Close, else 0.
    df['OpenTarget'] = (df['Open'].shift(-1) > df['Open']).astype(int)
    df['CloseTarget'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

def process_all_tickers():
    conn = sqlite3.connect("db/database.db")
    df = pd.read_sql_query("SELECT * FROM historical_daily", conn)
    conn.close()
    df = flatten_columns(df)
    
    # Rename 'Date' column to 'Datetime' if needed.
    columns_lower = {col.lower(): col for col in df.columns}
    if 'datetime' not in df.columns and 'date' in columns_lower:
        df.rename(columns={columns_lower['date']: 'Datetime'}, inplace=True)
    
    # Standardize Ticker column name (case-insensitive)
    if 'ticker' in columns_lower and columns_lower['ticker'] != 'Ticker':
        df.rename(columns={columns_lower['ticker']: 'Ticker'}, inplace=True)
    if 'Ticker' not in df.columns:
        raise KeyError("Column 'Ticker' not found in the historical data. Available columns: " + str(df.columns.tolist()))
    
    processed_list = []
    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values('Datetime')
        try:
            group_features = add_features(group)
            processed_list.append(group_features)
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")
    if processed_list:
        return pd.concat(processed_list, ignore_index=True)
    else:
        return None

def save_engineered_features(df, db_path="db/database.db", table_name="engineered_features"):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Engineered features saved to table '{table_name}' in {db_path}")

if __name__ == "__main__":
    print("Processing all tickers from daily historical data...")
    df_all = process_all_tickers()
    if df_all is not None:
        print("Saving engineered features...")
        save_engineered_features(df_all)
    else:
        print("No processed data to save.")
