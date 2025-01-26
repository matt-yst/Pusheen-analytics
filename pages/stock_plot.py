import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os


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
                if "market_data_A_1.csv" in file:
                    data.columns = column_names
                combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data


st.title("Interactive Stock Data Visualization")

# Directory setup
training_data_dir = "./TestData"
stocks = ["A", "B", "C", "D", "E"]
periods = sorted(os.listdir(training_data_dir))

if os.path.exists(training_data_dir):
    selected_period = st.selectbox("Select a period:", periods)
    selected_stock = st.selectbox("Select a stock:", stocks)

    if selected_period and selected_stock:
        # Load and combine data
        data = load_and_combine_data(training_data_dir, selected_stock, selected_period)

        if not data.empty:
            # Ensure timestamp is in datetime format
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
            data = data.dropna(subset=["timestamp"])  # Drop invalid timestamps

            # Add midPrice column
            data["midPrice"] = (data["bidPrice"] + data["askPrice"]) / 2

            # Rolling standard deviations
            data.set_index("timestamp", inplace=True)
            data["30_sec_std"] = data["bidPrice"].rolling("30s").std()
            data["60_sec_std"] = data["bidPrice"].rolling("60s").std()
            data.reset_index(inplace=True)

            # Time range selection
            min_time = data["timestamp"].min()
            max_time = data["timestamp"].max()

            if pd.isnull(min_time) or pd.isnull(max_time):
                st.warning("No valid timestamps found in the data.")
            else:
                # Convert min_time and max_time to Python datetime
                min_time = min_time.to_pydatetime()
                max_time = max_time.to_pydatetime()

                # Debugging: Print min_time and max_time types
                st.write(f"min_time: {min_time}, type: {type(min_time)}")
                st.write(f"max_time: {max_time}, type: {type(max_time)}")

                selected_time = st.slider(
                    "Select a time range:",
                    min_value=min_time,
                    max_value=max_time,
                    value=(min_time, max_time),
                    format="HH:mm:ss",
                )
                filtered_data = data[
                    (data["timestamp"] >= pd.Timestamp(selected_time[0])) &
                    (data["timestamp"] <= pd.Timestamp(selected_time[1]))
                ]

                # Plot the graph
                st.subheader("Stock Price Visualization")
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=filtered_data["timestamp"],
                        y=filtered_data["bidPrice"],
                        mode="lines",
                        name="Bid Price",
                    )
                )

                fig.update_layout(
                    title=f"Stock Data Visualization for {selected_stock} in {selected_period}",
                    xaxis_title="Timestamp",
                    yaxis_title="Value",
                )
                st.plotly_chart(fig)

                st.write("Filtered Data")
                st.dataframe(filtered_data)
        else:
            st.warning(f"No data found for Stock {selected_stock} in {selected_period}.")
else:
    st.error("TrainingData directory does not exist. Please check the path.")
