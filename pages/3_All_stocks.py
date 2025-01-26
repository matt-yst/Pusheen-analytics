import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import re


def natural_sort(lst):
    """
    Sorts a list using natural sorting (e.g., Period10 comes after Period9).
    """
    return sorted(lst, key=lambda x: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', x)])


def load_data_for_all_stocks(directory, stocks, period):
    """
    Load data for all stocks in a selected period.
    """
    combined_data = {}
    column_names = ['bidVolume', 'bidPrice', 'askVolume', 'askPrice', 'timestamp']
    for stock in stocks:
        stock_data = pd.DataFrame()
        period_path = os.path.join(directory, period, stock)
        if os.path.exists(period_path):
            files = natural_sort(os.listdir(period_path))  # Natural sorting for files
            for file in files:
                if file.startswith("market_data"):
                    data = pd.read_csv(
                        os.path.join(period_path, file),
                        header=None if "market_data_A_1.csv" in file else "infer",
                    )
                    if "market_data_A_1.csv" in file:
                        data.columns = column_names
                    if "timestamp" in data.columns:
                        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                        data = data.dropna(subset=["timestamp"])
                        stock_data = pd.concat([stock_data, data], ignore_index=True)
        if not stock_data.empty:
            combined_data[stock] = stock_data
    return combined_data


def resample_and_aggregate(data, interval="1T"):
    """
    Resample and aggregate data to a specified interval (e.g., 1T for 1 minute).
    """
    return data.resample(interval, on="timestamp").mean().dropna()


st.title("All Stocks for a Selected Period")

# Directory setup
training_data_dir = "./TestData"
stocks = ["A", "B", "C", "D", "E"]
periods = natural_sort(os.listdir(training_data_dir))  # Natural sorting for periods

if os.path.exists(training_data_dir):
    selected_period = st.selectbox("Select a period to analyze:", periods)

    if selected_period:
        # Load data for all stocks in the selected period
        data = load_data_for_all_stocks(training_data_dir, stocks, selected_period)

        # Plot bid prices for all stocks
        fig, ax = plt.subplots(figsize=(12, 6))
        for stock, stock_data in data.items():
            if not stock_data.empty:
                # Resample data for smoother trends
                stock_data = resample_and_aggregate(stock_data)
                ax.plot(stock_data.index, stock_data["bidPrice"], label=f"Stock {stock}")

        ax.set_title(f"Bid Prices for All Stocks in {selected_period}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Bid Price")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Please select a valid period.")
else:
    st.error("TrainingData directory does not exist. Please check the path.")
