import os
import pandas as pd
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import HoverTool


def load_test_data(directory, stock, period):
    """
    Load data for a specific stock and period from the TestData directory,
    navigating into the nested Period folder twice.
    """
    period_path = os.path.join(directory, period, period, stock)
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

    if not combined_data.empty:
        combined_data["midPrice"] = (combined_data["bidPrice"] + combined_data["askPrice"]) / 2

    return combined_data


# Streamlit interface
st.title("Stock Data Overview from TestData")

# Directory setup
test_data_dir = "./TestData"
stocks = ["A", "B", "C", "D", "E"]
periods = sorted(os.listdir(test_data_dir)) if os.path.exists(test_data_dir) else []

if os.path.exists(test_data_dir):
    selected_period = st.selectbox("Select a period:", periods)
    selected_stock = st.selectbox("Select a stock:", stocks)

    if selected_period and selected_stock:
        # Load test data for the selected stock and period
        data = load_test_data(test_data_dir, selected_stock, selected_period)

        if not data.empty:
            # Create a Bokeh plot
            source = data.set_index("timestamp")
            p = figure(
                x_axis_type="datetime",
                title=f"Price Data for Stock {selected_stock} (Period {selected_period})",
                width=900,
                height=400,
            )
            p.line(source.index, source["bidPrice"], color="blue", legend_label="Bid Price")
            p.line(source.index, source["askPrice"], color="red", legend_label="Ask Price")
            p.line(source.index, source["midPrice"], color="green", legend_label="Mid Price")

            hover = HoverTool(
                tooltips=[
                    ("Timestamp", "@x{%F %T}"),
                    ("Bid Price", "@y{0.00}"),
                    ("Ask Price", "@y{0.00}"),
                    ("Mid Price", "@y{0.00}")
                ],
                formatters={"@x": "datetime"},
                mode="vline",
            )
            p.add_tools(hover)

            p.legend.location = "top_left"
            p.xaxis.axis_label = "Time"
            p.yaxis.axis_label = "Price"

            st.bokeh_chart(p, use_container_width=True)
        else:
            st.warning(f"No data found for Stock {selected_stock} in {selected_period}.")
else:
    st.error("TestData directory does not exist. Please check the path.")
