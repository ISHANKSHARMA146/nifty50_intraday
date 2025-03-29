import os
import pickle
import sqlite3
import pandas as pd
import tkinter as tk
from tkinter import ttk

def load_latest_features(db_path="db/database.db", table_name="engineered_features"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    # If a "Date" column exists instead of "Datetime", rename it
    if 'Datetime' not in df.columns and 'Date' in df.columns:
        df.rename(columns={'Date': 'Datetime'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.sort_values(['Ticker', 'Datetime'], inplace=True)
    latest_dict = {}
    for ticker, group in df.groupby('Ticker'):
        latest_row = group.iloc[-1]  # Get the latest row for each ticker
        latest_dict[ticker] = latest_row
    return latest_dict

def load_model(ticker, target, model_dir="models"):
    # target is "open" or "close"
    model_path = os.path.join(model_dir, f"{ticker}_{target}_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    return None

def predict_signals(latest_features):
    # Feature columns used in daily training
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'RSI_14', 'BB_Middle', 'BB_Upper', 'BB_Lower',
        'ATR_14'
    ]
    signals = {}
    for ticker, row in latest_features.items():
        model_open = load_model(ticker, "open")
        model_close = load_model(ticker, "close")
        if model_open is not None and model_close is not None:
            data_point = pd.DataFrame([row[feature_cols]])
            pred_open = model_open.predict(data_point)[0]
            pred_close = model_close.predict(data_point)[0]
            # Final decision: if both predict upward then BUY, if both predict downward then SELL, else HOLD.
            if pred_open == 1 and pred_close == 1:
                signal = "BUY"
            elif pred_open == 0 and pred_close == 0:
                signal = "SELL"
            else:
                signal = "HOLD"
            signals[ticker] = signal
        else:
            signals[ticker] = "No Model"
    return signals

class SignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Daily Signals for Nifty 50 Stocks")
        self.create_widgets()
        self.update_signals()  # initial update

    def create_widgets(self):
        # Create a frame for the treeview and scrollbar
        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        # Create the Treeview with two columns and set headings to be centered.
        self.tree = ttk.Treeview(frame, columns=("Ticker", "Signal"), show="headings")
        self.tree.heading("Ticker", text="Ticker", anchor="center")
        self.tree.heading("Signal", text="Signal", anchor="center")
        self.tree.column("Ticker", anchor="center", width=150)
        self.tree.column("Signal", anchor="center", width=100)

        # Add a vertical scrollbar
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Configure grid so the Treeview expands with the window.
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def update_signals(self):
        latest_features = load_latest_features()
        signals = predict_signals(latest_features)
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        # Insert new items sorted by ticker
        for ticker in sorted(signals.keys()):
            self.tree.insert("", "end", values=(ticker, signals[ticker]))
        # Tag rows for coloring
        for item in self.tree.get_children():
            vals = self.tree.item(item, "values")
            sig = vals[1]
            if sig == "BUY":
                self.tree.item(item, tags=("buy",))
            elif sig == "SELL":
                self.tree.item(item, tags=("sell",))
            else:
                self.tree.item(item, tags=("hold",))
        self.tree.tag_configure("buy", background="lightgreen")
        self.tree.tag_configure("sell", background="salmon")
        self.tree.tag_configure("hold", background="lightgrey")
        # Schedule next update in 60 seconds
        self.root.after(60000, self.update_signals)

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalApp(root)
    root.mainloop()
