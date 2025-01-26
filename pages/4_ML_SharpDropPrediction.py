import streamlit as st
from .utils.predicter_utils import load_all_data, generate_features
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.title("ML Model for Sharp Change Prediction")

# Directory setup
training_data_dir = "./TrainingData"
stocks = ["A", "B", "C", "D", "E"]

if os.path.exists(training_data_dir):
    st.write("Loading and combining data...")
    data = load_all_data(training_data_dir, stocks)

    if not data.empty:
        st.write("Generating features...")
        data = generate_features(data)

        # Feature and target setup
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

        st.write("Balancing the dataset using SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Train using k-fold cross-validation
        st.write("Training model with k-fold cross-validation...")
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = XGBClassifier()
        accuracies = []

        for train_idx, test_idx in kfold.split(X_resampled, y_resampled):
            X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
            y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]
            model.fit(X_train, y_train)
            accuracies.append(model.score(X_test, y_test))

        avg_accuracy = np.mean(accuracies)
        st.write(f"Average Model Accuracy: {avg_accuracy:.2f}")

        # Predict on the data
        data["predicted_sharp_change"] = model.predict(X)

        # Visualize predictions
        st.subheader("Predicted vs Actual Sharp Changes")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["timestamp"], y=data["sharp_change"], name="Actual"))
        fig.add_trace(go.Scatter(x=data["timestamp"], y=data["predicted_sharp_change"], name="Predicted"))
        fig.update_layout(title="Sharp Change Predictions", xaxis_title="Timestamp", yaxis_title="Sharp Change")
        st.plotly_chart(fig)
    else:
        st.warning("No data found for the selected stocks.")
else:
    st.error("TrainingData directory not found.")
