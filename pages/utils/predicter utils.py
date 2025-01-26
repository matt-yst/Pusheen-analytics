import pandas as pd
import os

def load_all_data(directory, stocks):
    all_data = []
    periods = sorted(os.listdir(directory))
    for period in periods:
        for stock in stocks:
            period_path = os.path.join(directory, period, stock)
            column_names = ['bidVolume', 'bidPrice', 'askVolume', 'askPrice', 'timestamp']
            if os.path.exists(period_path):
                for file in sorted(os.listdir(period_path)):
                    if file.startswith("market_data"):
                        data = pd.read_csv(
                            os.path.join(period_path, file),
                            header=None if "market_data_A_1.csv" in file else "infer",
                        )
                        if "market_data_A_1.csv" in file or len(data.columns) == 5:
                            data.columns = column_names
                        if "timestamp" in data.columns:
                            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                            data = data.dropna(subset=["timestamp"])
                            data["stock"] = stock
                            data["period"] = period
                            all_data.append(data)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def generate_features(data):
    data["midPrice"] = (data["bidPrice"] + data["askPrice"]) / 2
    data["rolling_avg_30"] = data["midPrice"].rolling(window=30).mean()
    data["rolling_avg_60"] = data["midPrice"].rolling(window=60).mean()
    data["rolling_std_30"] = data["midPrice"].rolling(window=30).std()
    data["rolling_std_60"] = data["midPrice"].rolling(window=60).std()
    data["momentum"] = data["midPrice"].pct_change()
    data["sharp_change"] = (abs(data["momentum"]) > 0.05).astype(int)
    return data.dropna()
