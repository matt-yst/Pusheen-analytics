import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st

# Utility functions
def load_combined_data(directory):
    """Load and combine stock data from a directory."""
    all_data = []
    column_names = ['bidVolume', 'bidPrice', 'askVolume', 'askPrice', 'timestamp']

    periods = sorted(os.listdir(directory))
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

# Interactive page for new graphs
st.title("Advanced Stock Visualizations")

# Directory setup
training_data_dir = "./TrainingData"

if os.path.exists(training_data_dir):
    data = load_combined_data(training_data_dir)

    if not data.empty:
        # Ensure midPrice is calculated
        data["midPrice"] = (data["bidPrice"] + data["askPrice"]) / 2

        # Sidebar filters
        stock_options = data["stock"].unique()
        selected_stock = st.sidebar.selectbox("Select Stock", stock_options)
        filtered_data = data[data["stock"] == selected_stock]

        st.header(f"Visualizations for Stock {selected_stock}")

        # Price Heatmap
        st.subheader("Price Heatmap")
        heatmap_data = filtered_data.copy()
        heatmap_data["minute"] = heatmap_data["timestamp"].dt.floor("T")
        heatmap_pivot = heatmap_data.pivot_table(
            index="minute", columns="period", values="midPrice", aggfunc="mean"
        )
        sns.heatmap(heatmap_pivot, cmap="coolwarm", cbar_kws={"label": "Mid Price"})
        st.pyplot(plt.gcf())
        plt.clf()

        # Volume vs. Price Change Correlation
        st.subheader("Volume vs. Price Change Correlation")
        filtered_data["price_change"] = filtered_data["midPrice"].pct_change()
        sns.scatterplot(
            x=filtered_data["bidVolume"],
            y=filtered_data["price_change"],
            alpha=0.5
        )
        plt.xlabel("Bid Volume")
        plt.ylabel("Price Change (%)")
        st.pyplot(plt.gcf())
        plt.clf()

        # Candlestick Chart with Momentum
        st.subheader("Candlestick Chart with Momentum")
        candlestick_fig = go.Figure(
            data=[
                go.Candlestick(
                    x=filtered_data["timestamp"],
                    open=filtered_data["bidPrice"],
                    high=filtered_data["askPrice"],
                    low=filtered_data["bidPrice"],
                    close=filtered_data["askPrice"],
                    name="Candlestick"
                ),
                go.Scatter(
                    x=filtered_data["timestamp"],
                    y=filtered_data["midPrice"].pct_change(),
                    mode="lines",
                    name="Momentum"
                )
            ]
        )
        candlestick_fig.update_layout(title="Candlestick and Momentum", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(candlestick_fig)

        # Cross-Correlation Heatmap
        st.subheader("Cross-Correlation Heatmap")
        correlation_matrix = data.pivot_table(
            index="timestamp", columns="stock", values="midPrice"
        ).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar_kws={"label": "Correlation"})
        st.pyplot(plt.gcf())
        plt.clf()

        # Trade Clustering
        st.subheader("Trade Clustering")
        from sklearn.cluster import KMeans
        
        cluster_data = filtered_data[["midPrice", "bidVolume"]].dropna()
        kmeans = KMeans(n_clusters=3, random_state=42).fit(cluster_data)
        filtered_data["cluster"] = kmeans.labels_

        sns.scatterplot(
            x=filtered_data["midPrice"],
            y=filtered_data["bidVolume"],
            hue=filtered_data["cluster"],
            palette="Set1"
        )
        plt.xlabel("Mid Price")
        plt.ylabel("Bid Volume")
        st.pyplot(plt.gcf())
        plt.clf()

    else:
        st.warning("No data found in the specified directory.")
else:
    st.error("TrainingData directory does not exist. Please check the path.")
