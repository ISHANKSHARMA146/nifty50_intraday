import yfinance as yf
import pandas as pd
import sqlite3
import os

# List of Nifty 50 tickers on Yahoo Finance (commonly using ".NS" suffix)
nifty50_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "HINDUNILVR.NS", "AXISBANK.NS", "ITC.NS", "SBIN.NS",
    "HDFC.NS", "LT.NS", "MARUTI.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS",
    "HCLTECH.NS", "TITAN.NS", "WIPRO.NS", "ONGC.NS", "POWERGRID.NS",
    "NTPC.NS", "BPCL.NS", "TATAMOTORS.NS", "TATACONSUM.NS", "TATASTEEL.NS",
    "DIVISLAB.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "DRREDDY.NS", "BRITANNIA.NS",
    "HEROMOTOCO.NS", "GRASIM.NS", "SHREECEM.NS", "EICHERMOT.NS", "ADANIPORTS.NS",
    "ADANIGREEN.NS", "ADANIPOWER.NS", "COALINDIA.NS", "INDUSINDBK.NS", "SBILIFE.NS",
    "M&M.NS", "HAVELLS.NS", "VEDL.NS", "UPL.NS", "CIPLA.NS",
    "JSWSTEEL.NS", "BAJAJ-AUTO.NS", "ICICIGI.NS", "PIDILITIND.NS", "TORNTPHARM.NS"
]

def flatten_columns(df):
    # If columns are a MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def download_data_for_ticker(ticker):
    print(f"Downloading daily data for {ticker}...")
    df = yf.download(ticker, period="max", interval="1d")
    if df.empty:
        print(f"No data for {ticker}")
        return None
    df = flatten_columns(df)
    df.reset_index(inplace=True)
    df['Ticker'] = ticker
    return df

def download_all_data():
    all_data = []
    for ticker in nifty50_tickers:
        df = download_data_for_ticker(ticker)
        if df is not None:
            all_data.append(df)
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    return None

def save_to_csv(df, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Daily data saved to {filename}")

def save_to_sqlite(df, db_path="db/database.db", table_name="historical_daily"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Daily data saved to SQLite database at {db_path} in table '{table_name}'")

if __name__ == "__main__":
    data = download_all_data()
    if data is not None:
        save_to_csv(data, "data/historical/nifty50_daily.csv")
        save_to_sqlite(data)
