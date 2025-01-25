import streamlit as st
import pandas as pd
import os
import plotly.express as px

def load_and_combine_data(directory, stock, period):
    
    combined_data = pd.DataFrame()
    period_path = os.path.join(directory, period, stock)
    column_names = ['bidVolume', 'bidPrice', 'askVolume', 'askPrice', 'timestamp']
    
    if os.path.exists(period_path):
        for file in sorted(os.listdir(period_path)):
            if file.startswith("market_data"):
                data = pd.read_csv(
                    os.path.join(period_path, file),
                    header=None if "market_data_A_1.csv" in file else "infer",
                )
                # Assign column names if missing
                if "market_data_A_1.csv" in file:
                    data.columns = column_names
                combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data

st.title("Overview of Stock Data")

# Directory setup
training_data_dir = "./TrainingData"
stocks = ["A", "B", "C", "D", "E"]
periods = sorted(os.listdir(training_data_dir))  # List of periods

if os.path.exists(training_data_dir):
    # User selects the stock and period
    selected_period = st.selectbox("Select a period:", periods)
    selected_stock = st.selectbox("Select a stock:", stocks)

    if selected_period and selected_stock:
        # Load and combine data for the selected stock and period
        data = load_and_combine_data(training_data_dir, selected_stock, selected_period)

        if not data.empty:
            # Ensure timestamp column is in datetime format
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            data = data.dropna(subset=["timestamp"])  # Drop invalid timestamps

            # Show a data overview
            st.write(f"Data Overview for Stock {selected_stock} in {selected_period}")
            st.dataframe(data.head(10))  # Display the first 10 rows of the data

            # Generate a summary chart
            st.subheader("Summary Chart")
            if "bidPrice" in data.columns and "askPrice" in data.columns:
                data["averagePrice"] = (data["bidPrice"] + data["askPrice"]) / 2
                fig = px.line(
                    data,
                    x="timestamp",
                    y="averagePrice",
                    title=f"Average Price Trend for Stock {selected_stock} in {selected_period}",
                    labels={"averagePrice": "Average Price", "timestamp": "Timestamp"}
                )
                st.plotly_chart(fig)
            else:
                st.warning("Missing required columns (bidPrice, askPrice) in the data.")
        else:
            st.warning(f"No data found for Stock {selected_stock} in {selected_period}.")
else:
    st.error("TrainingData directory does not exist. Please check the path.")
