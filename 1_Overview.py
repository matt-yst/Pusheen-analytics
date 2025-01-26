import streamlit as st
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
import os

def load_data_for_stock(directory, stock, period):
    """
    Load data for a single stock in a selected period.
    """
    period_path = os.path.join(directory, period, period, stock)  # Double nested `Period` folder
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

st.title("Interactive Overview: Prices, Volumes, and Analysis")

# Directory setup
test_data_dir = "./TestData"  # Changed directory to TestData
stocks = ["A", "B", "C", "D", "E"]
periods = sorted(os.listdir(test_data_dir)) if os.path.exists(test_data_dir) else []

if os.path.exists(test_data_dir):
    selected_period = st.selectbox("Select a period:", periods)
    selected_stock = st.selectbox("Select a stock:", stocks)

    if selected_period and selected_stock:
        # Load data for the selected stock
        data = load_data_for_stock(test_data_dir, selected_stock, selected_period)

        if not data.empty:
            # Compute additional features
            data["midPrice"] = (data["bidPrice"] + data["askPrice"]) / 2
            data.set_index("timestamp", inplace=True)
            data["std_30s"] = data["midPrice"].rolling("30s").std()
            data["std_60s"] = data["midPrice"].rolling("60s").std()
            data.reset_index(inplace=True)

            # Main graph for prices
            st.subheader("Price Data (Bid, Ask, Mid-Price)")
            bokeh_source_prices = ColumnDataSource(data)
            price_fig = figure(
                x_axis_type="datetime",
                title=f"Price Data for Stock {selected_stock} ({selected_period})",
                plot_width=900, plot_height=400,
                tools="pan,wheel_zoom,box_zoom,reset"
            )
            hover_price = HoverTool(
                tooltips=[
                    ("Timestamp", "@timestamp{%F %T}"),
                    ("Bid Price", "@bidPrice"),
                    ("Ask Price", "@askPrice"),
                    ("Mid Price", "@midPrice")
                ],
                formatters={"@timestamp": "datetime"}
            )
            price_fig.add_tools(hover_price)
            price_fig.line("timestamp", "bidPrice", source=bokeh_source_prices, color="blue", legend_label="Bid Price")
            price_fig.line("timestamp", "askPrice", source=bokeh_source_prices, color="red", legend_label="Ask Price")
            price_fig.line("timestamp", "midPrice", source=bokeh_source_prices, color="green", legend_label="Mid Price")
            price_fig.legend.location = "top_left"
            st.bokeh_chart(price_fig, use_container_width=True)

            # Standard deviation graph
            st.subheader("Standard Deviation (30s and 60s)")
            bokeh_source_std = ColumnDataSource(data)
            std_fig = figure(
                x_axis_type="datetime",
                title=f"Standard Deviation for Stock {selected_stock} ({selected_period})",
                plot_width=900, plot_height=400,
                tools="pan,wheel_zoom,box_zoom,reset"
            )
            hover_std = HoverTool(
                tooltips=[
                    ("Timestamp", "@timestamp{%F %T}"),
                    ("30s Std Dev", "@std_30s"),
                    ("60s Std Dev", "@std_60s")
                ],
                formatters={"@timestamp": "datetime"}
            )
            std_fig.add_tools(hover_std)
            std_fig.line("timestamp", "std_30s", source=bokeh_source_std, color="purple", legend_label="30s Std Dev", line_dash="dotted")
            std_fig.line("timestamp", "std_60s", source=bokeh_source_std, color="orange", legend_label="60s Std Dev", line_dash="dotted")
            std_fig.legend.location = "top_left"
            st.bokeh_chart(std_fig, use_container_width=True)

            # Volume graph
            st.subheader("Volume Data (Bid and Ask)")
            bokeh_source_volumes = ColumnDataSource(data)
            volume_fig = figure(
                x_axis_type="datetime",
                title=f"Volume Data for Stock {selected_stock} ({selected_period})",
                plot_width=900, plot_height=400,
                tools="pan,wheel_zoom,box_zoom,reset"
            )
            hover_volume = HoverTool(
                tooltips=[
                    ("Timestamp", "@timestamp{%F %T}"),
                    ("Bid Volume", "@bidVolume"),
                    ("Ask Volume", "@askVolume")
                ],
                formatters={"@timestamp": "datetime"}
            )
            volume_fig.add_tools(hover_volume)
            volume_fig.line("timestamp", "bidVolume", source=bokeh_source_volumes, color="gray", legend_label="Bid Volume")
            volume_fig.line("timestamp", "askVolume", source=bokeh_source_volumes, color="lightblue", legend_label="Ask Volume")
            volume_fig.legend.location = "top_left"
            st.bokeh_chart(volume_fig, use_container_width=True)

            # Highlight low/high points
            st.subheader("Daily Low and High Highlights")
            low_high_source = ColumnDataSource(data)
            daily_low = data["midPrice"].min()
            daily_high = data["midPrice"].max()
            low_high_fig = figure(
                x_axis_type="datetime",
                title=f"Daily Low and High for Stock {selected_stock} ({selected_period})",
                plot_width=900, plot_height=400,
                tools="pan,wheel_zoom,box_zoom,reset"
            )
            hover_low_high = HoverTool(
                tooltips=[
                    ("Timestamp", "@timestamp{%F %T}"),
                    ("Mid Price", "@midPrice")
                ],
                formatters={"@timestamp": "datetime"}
            )
            low_high_fig.add_tools(hover_low_high)
            low_high_fig.line("timestamp", "midPrice", source=low_high_source, color="green", legend_label="Mid Price")
            low_high_fig.circle(
                x=data[data["midPrice"] == daily_low]["timestamp"],
                y=[daily_low],
                size=10, color="cyan", legend_label="Daily Low"
            )
            low_high_fig.circle(
                x=data[data["midPrice"] == daily_high]["timestamp"],
                y=[daily_high],
                size=10, color="magenta", legend_label="Daily High"
            )
            low_high_fig.legend.location = "top_left"
            st.bokeh_chart(low_high_fig, use_container_width=True)
        else:
            st.warning(f"No data found for Stock {selected_stock} in {selected_period}.")
else:
    st.error("TestData directory does not exist. Please check the path.")
