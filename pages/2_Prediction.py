import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import plotly.graph_objects as go
import os


def load_all_data(directory, stocks):
    """
    Load and combine market data for all periods and selected stocks.
    Args:
        directory (str): Path to the TrainingData directory.
        stocks (list): List of stock symbols (e.g., ["A", "B", "C"]).

    Returns:
        pd.DataFrame: Combined DataFrame for all stocks across all periods.
    """
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
    """
    Add rolling averages, standard deviations, and momentum features to the dataset.
    """
    data["midPrice"] = (data["bidPrice"] + data["askPrice"]) / 2
    data["rolling_avg_30"] = data["midPrice"].rolling(window=30).mean()
    data["rolling_avg_60"] = data["midPrice"].rolling(window=60).mean()
    data["rolling_std_30"] = data["midPrice"].rolling(window=30).std()
    data["rolling_std_60"] = data["midPrice"].rolling(window=60).std()
    data["momentum"] = data["midPrice"].pct_change()
    data["sharp_change"] = (abs(data["momentum"]) > 0.05).astype(int)  # Sharp change threshold
    return data.dropna()


st.title("Improved Stock Movement Prediction with All Data")

# Directory setup
training_data_dir = "./TrainingData"
stocks = ["A", "B", "C", "D", "E"]

if os.path.exists(training_data_dir):
    # Load all data for selected stocks
    st.write("Loading and combining data...")
    data = load_all_data(training_data_dir, stocks)

    if not data.empty:
        # Generate features
        st.write("Generating features...")
        data = generate_features(data)

        # Prepare the dataset for modeling
        feature_columns = [
            "rolling_avg_30",
            "rolling_avg_60",
            "rolling_std_30",
            "rolling_std_60",
            "momentum",
        ]
        target_column = "sharp_change"
        X = data[feature_columns]
        y = data[target_column]

        # Balance the dataset using SMOTE
        st.write("Balancing the dataset...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        st.write("Balanced Sharp Change Distribution:", pd.Series(y_resampled).value_counts())

        # Convert to NumPy arrays for compatibility with StratifiedKFold
        X_resampled_np = X_resampled.to_numpy()
        y_resampled_np = y_resampled

        # Train the model using k-fold cross-validation
        st.write("Training the model with k-fold cross-validation...")
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = XGBClassifier()
        accuracies = []

        for train_idx, test_idx in kfold.split(X_resampled_np, y_resampled_np):
            X_train, X_test = X_resampled_np[train_idx], X_resampled_np[test_idx]
            y_train, y_test = y_resampled_np[train_idx], y_resampled_np[test_idx]
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies)
        st.write(f"Average Model Accuracy: {avg_accuracy:.2f}")

        # Predict on the entire dataset for visualization
        data["predicted_sharp_change"] = model.predict(X.to_numpy())

        # Plot actual vs predicted sharp changes
        st.subheader("Sharp Change Predictions")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data["sharp_change"],
                mode="lines+markers",
                name="Actual Sharp Changes",
                marker=dict(color="red"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data["predicted_sharp_change"],
                mode="lines+markers",
                name="Predicted Sharp Changes",
                marker=dict(color="blue"),
            )
        )
        fig.update_layout(
            title="Actual vs Predicted Sharp Changes",
            xaxis_title="Timestamp",
            yaxis_title="Sharp Change (1=Yes, 0=No)",
            legend_title="Legend",
            template="plotly_white",
        )
        st.plotly_chart(fig)
    else:
        st.warning("No valid data found for the selected stocks.")
else:
    st.error("TrainingData directory does not exist. Please check the path.")
