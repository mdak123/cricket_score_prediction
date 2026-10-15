
import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="IPL Cricket Score Prediction",
    page_icon="🏏",
    layout="centered"
)

# App Title
st.title("🏏 IPL Cricket Score Prediction")
st.write("Predict Runs using Innings and Strike Rate.")

# Load Model
@st.cache_resource
def load_model():
    model_path = "cricket_score_prediction.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            "Please upload the trained model."
        )

    return joblib.load(model_path)

# Load the trained model
try:
    model = load_model()
    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Show model feature information
if hasattr(model, "feature_names_in_"):
    st.write(
        "Model features:",
        list(model.feature_names_in_)
    )

if hasattr(model, "n_features_in_"):
    st.write(
        "Expected number of features:",
        model.n_features_in_
    )

st.divider()

# User Inputs
st.subheader("Enter Match Details")

innings = st.number_input(
    "Innings (Inns)",
    min_value=1,
    max_value=2,
    value=1,
    step=1
)

strike_rate = st.number_input(
    "Strike Rate (SR)",
    min_value=0.0,
    max_value=500.0,
    value=120.0,
    step=1.0
)

# Prediction
if st.button("Predict Runs", type="primary"):

    try:
        # Input features must match training columns
        input_data = pd.DataFrame(
            [[innings, strike_rate]],
            columns=["Inns", "SR"]
        )

        # Predict
        prediction = model.predict(input_data)

        predicted_runs = float(prediction[0])

        st.success(
            f"🏏 Predicted Runs: {predicted_runs:.2f}"
        )

        st.caption(
            "This is a model estimate, not a guaranteed match score."
        )

    except Exception as e:
        st.error("Prediction failed!")
        st.code(f"{type(e).__name__}: {e}")

        st.info(
            "Check the model file, feature names, "
            "feature order and training preprocessing."
        )
