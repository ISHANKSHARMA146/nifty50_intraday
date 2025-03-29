import sqlite3
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

def load_engineered_features(db_path="db/database.db", table_name="engineered_features"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def train_model_for_ticker(df, ticker, target_col):
    # Feature columns (using our daily technical indicators)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
                    'MACD', 'MACD_Signal', 'MACD_Hist',
                    'RSI_14', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                    'ATR_14']
    X = df[feature_cols]
    y = df[target_col]
    if len(df) < 50:
        print(f"Not enough data for {ticker} to train {target_col} model")
        return None, None
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.7, 1.0]
    }
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy', verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{ticker} ({target_col}) - Best params: {grid_search.best_params_}, Test Accuracy: {accuracy:.4f}")
    return best_model, accuracy

def train_models_all():
    df = load_engineered_features()
    models = {}   # keys: (ticker, "open") and (ticker, "close")
    accuracies = {}
    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values('Datetime')
        print(f"Training models for {ticker}...")
        model_open, acc_open = train_model_for_ticker(group, ticker, "OpenTarget")
        model_close, acc_close = train_model_for_ticker(group, ticker, "CloseTarget")
        if model_open is not None:
            models[(ticker, "open")] = model_open
            accuracies[(ticker, "open")] = acc_open
            model_path = os.path.join("models", f"{ticker}_open_model.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model_open, f)
            print(f"Model for {ticker} (open) saved to {model_path}")
        if model_close is not None:
            models[(ticker, "close")] = model_close
            accuracies[(ticker, "close")] = acc_close
            model_path = os.path.join("models", f"{ticker}_close_model.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model_close, f)
            print(f"Model for {ticker} (close) saved to {model_path}")
    return models, accuracies

if __name__ == "__main__":
    models, accuracies = train_models_all()
