import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data_for_stock(directory, stock, period):
    """
    Load data for a single stock in a selected period.
    """
    period_path = os.path.join(directory, period, stock)
    column_names = ['bidVolume', 'bidPrice', 'askVolume', 'askPrice', 'timestamp']
    combined_data = pd.DataFrame()
    if os.path.exists(period_path):
        for file in sorted(os.listdir(period_path)):
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
                    combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data


def detect_sharp_drops(data, relative=False, threshold=0.02):
    """
    Detect sharp drops in bid prices, either absolute or relative.
    """
    data["midPrice"] = (data["bidPrice"] + data["askPrice"]) / 2
    data["momentum"] = data["midPrice"].pct_change()

    if relative:
        # Use rolling z-scores for relative drop detection
        data["z_score"] = (data["momentum"] - data["momentum"].rolling(window=30).mean()) / data["momentum"].rolling(window=30).std()
        data["sharp_drop"] = data["z_score"] < -2  # Drops below 2 standard deviations
    else:
        # Absolute threshold-based detection
        data["sharp_drop"] = data["momentum"] < -threshold

    return data[data["sharp_drop"]]


def create_timeline(drops, stock, period):
    """
    Create a timeline of sharp drops for the given stock and period.
    """
    drops["stock"] = stock
    drops["period"] = period
    drops["time"] = drops["timestamp"].dt.time
    return drops[["stock", "period", "time"]]


st.title("Overview: Bid/Ask Prices, Volumes, and Sharp Drop Analysis")

# Directory setup
training_data_dir = "./TrainingData"
stocks = ["A", "B", "C", "D", "E"]
periods = sorted(os.listdir(training_data_dir))

if os.path.exists(training_data_dir):
    selected_period = st.selectbox("Select a period:", periods)
    selected_stock = st.selectbox("Select a stock:", stocks)

    if selected_period and selected_stock:
        # Load data for the selected stock
        data = load_data_for_stock(training_data_dir, selected_stock, selected_period)

        if not data.empty:
            # Plot bid/ask prices and volumes
            st.subheader("Bid/Ask Prices and Volumes")
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Price plot
            axs[0].plot(data["timestamp"], data["bidPrice"], label="Bid Price", color="blue")
            axs[0].plot(data["timestamp"], data["askPrice"], label="Ask Price", color="red")
            axs[0].set_title("Bid and Ask Prices Over Time")
            axs[0].set_ylabel("Price")
            axs[0].legend()

            # Volume plot
            axs[1].plot(data["timestamp"], data["bidVolume"], label="Bid Volume", color="green")
            axs[1].plot(data["timestamp"], data["askVolume"], label="Ask Volume", color="orange")
            axs[1].set_title("Bid and Ask Volumes Over Time")
            axs[1].set_xlabel("Timestamp")
            axs[1].set_ylabel("Volume")
            axs[1].legend()

            plt.tight_layout()
            st.pyplot(fig)

            # Add momentum plot
            data["midPrice"] = (data["bidPrice"] + data["askPrice"]) / 2
            data["momentum"] = data["midPrice"].pct_change()

            st.subheader("Momentum Over Time")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data["timestamp"], data["momentum"], label="Momentum", color="purple")
            ax.axhline(-0.02, color="red", linestyle="--", label="Default Sharp Drop Threshold")
            ax.set_title("Momentum Over Time")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Momentum")
            ax.legend()
            st.pyplot(fig)

            # Dynamic detection mode
            detection_mode = st.radio("Select detection mode:", ["Absolute", "Relative"])
            relative = detection_mode == "Relative"

            # Sharp drop detection
            threshold = st.slider(
                "Set Sharp Drop Threshold (as %):",
                min_value=0.01,
                max_value=0.10,
                value=0.02,
                step=0.01,
            )
            sharp_drops = detect_sharp_drops(data, relative=relative, threshold=threshold)

            if not sharp_drops.empty:
                # Plot sharp drop chart
                st.subheader("Chart of Sharp Drops")
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.scatterplot(
                    x="timestamp",
                    y="bidPrice",
                    data=sharp_drops,
                    ax=ax,
                    color="red",
                    label="Sharp Drops"
                )
                ax.set_title("Sharp Drops in Bid Prices")
                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Bid Price")
                st.pyplot(fig)

                # Display timeline of sharp drops
                st.subheader("Timeline of Sharp Drops")
                timeline = create_timeline(sharp_drops, stock=selected_stock, period=selected_period)
                st.dataframe(timeline)

                # Downloadable timeline
                csv = timeline.to_csv(index=False)
                st.download_button(
                    label="Download Timeline as CSV",
                    data=csv,
                    file_name="sharp_drop_timeline.csv"
                )
            else:
                st.info("No sharp drops detected for the selected stock and period with the current settings.")
        else:
            st.warning(f"No data found for Stock {selected_stock} in {selected_period}.")
else:
    st.error("TrainingData directory does not exist. Please check the path.")
