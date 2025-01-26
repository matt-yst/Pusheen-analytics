import os
import pandas as pd

def load_and_preprocess_data(directory):
    """
    Load and preprocess market data from a specified directory.
    Combines data for multiple stocks and periods.
    """
    all_data = []
    periods = sorted(os.listdir(directory))
    column_names = ['bidVolume', 'bidPrice', 'askVolume', 'askPrice', 'timestamp']

    for period in periods:
        for stock in os.listdir(os.path.join(directory, period)):
            stock_path = os.path.join(directory, period, stock)
            if os.path.isdir(stock_path):
                for file in sorted(os.listdir(stock_path)):
                    if file.startswith("market_data"):
                        data = pd.read_csv(
                            os.path.join(stock_path, file),
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


def feature_engineering(data):
    """
    Generate features for stock price data:
    - Momentum
    - Binary target column for sharp changes
    """
    # Compute midPrice
    data["midPrice"] = (data["bidPrice"] + data["askPrice"]) / 2

    # Calculate momentum
    data["momentum"] = data["midPrice"].pct_change()

    # Create binary target column for sharp changes
    data["sharp_change"] = (abs(data["momentum"]) > 0.05).astype(int)

    return data.dropna()


# Example usage
if __name__ == "__main__":
    # Path to the TrainingData directory
    training_data_dir = "./TrainingData"

    # Load and preprocess data
    combined_data = load_and_preprocess_data(training_data_dir)

    if not combined_data.empty:
        # Generate features
        processed_data = feature_engineering(combined_data)
        print(processed_data.head())
    else:
        print("No data found in the specified directory.")
