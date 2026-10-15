
import streamlit as st
import pickle
import pandas as pd
import os

# Page Configuration
st.set_page_config(
    page_title="Cricket Score Predictor",
    page_icon="🏏",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b132b, #1c2541);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
.hero {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    background: linear-gradient(90deg, #11998e, #38ef7d);
    color: #071b24;
    margin-bottom: 25px;
}
.hero h1, .hero p {
    color: #071b24 !important;
}
div.stButton > button {
    width: 100%;
    background-color: #20c997;
    color: #071b24;
    font-weight: bold;
    border-radius: 10px;
    height: 48px;
}
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    with open("cricket_score_prediction.pkl", "rb") as file:
        return pickle.load(file)

# Header
st.markdown("""
<div class="hero">
    <h1>🏏 Cricket Score Predictor</h1>
    <p>AI Powered Cricket Match Score Prediction</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏏 Match Information")
st.sidebar.write("Enter match details below.")

team1 = st.sidebar.selectbox(
    "Batting Team",
    ["India", "Chennai Super Kings",
     "Mumbai Indians", "Royal Challengers Bengaluru",
     "Kolkata Knight Riders", "Rajasthan Royals",
     "Delhi Capitals", "Punjab Kings",
     "Sunrisers Hyderabad", "Lucknow Super Giants",
     "Gujarat Titans"]
)

team2 = st.sidebar.selectbox(
    "Bowling Team",
    ["Australia", "India", "Mumbai Indians",
     "Chennai Super Kings", "Kolkata Knight Riders",
     "Rajasthan Royals", "Delhi Capitals",
     "Punjab Kings", "Sunrisers Hyderabad",
     "Gujarat Titans", "Lucknow Super Giants"]
)

venue = st.sidebar.text_input("Venue", "Wankhede Stadium")

# Main dashboard
st.subheader("📊 Current Match Situation")

c1, c2, c3 = st.columns(3)

with c1:
    runs = st.number_input("Current Runs", 0, 400, 120)

with c2:
    wickets = st.number_input("Wickets Lost", 0, 10, 3)

with c3:
    overs = st.number_input(
        "Overs Completed",
        min_value=0.0,
        max_value=20.0,
        value=12.0,
        step=0.1
    )

c4, c5 = st.columns(2)

with c4:
    runs_last_5 = st.number_input(
        "Runs in Last 5 Overs", 0, 150, 45
    )

with c5:
    wickets_last_5 = st.number_input(
        "Wickets in Last 5 Overs", 0, 10, 1
    )

st.divider()

# Match summary
st.subheader("🏆 Match Summary")

m1, m2, m3 = st.columns(3)
m1.metric("Current Score", f"{runs}/{wickets}")
m2.metric("Overs", f"{overs}/20")
m3.metric(
    "Run Rate",
    f"{runs / overs:.2f}" if overs > 0 else "0.00"
)

# Prediction
if st.button("🔮 Predict Final Score"):
    if overs <= 0:
        st.warning("Please enter overs greater than zero.")
    elif wickets >= 10:
        st.warning("All wickets have fallen.")
    elif not os.path.exists("cricket_score_prediction.pkl"):
        st.error(
            "Model file not found. Keep the .pkl file "
            "in the same folder as app.py."
        )
    else:
        try:
            model = load_model()

            # Update these features to match your training dataset.
            values = {
                "runs": runs,
                "wickets": wickets,
                "overs": overs,
                "runs_last_5": runs_last_5,
                "wickets_last_5": wickets_last_5,
                "total_overs": 20
            }

            features = getattr(model, "feature_names_in_", None)

            if features is not None:
                missing = [f for f in features if f not in values]
                if missing:
                    st.error(
                        "Model ke required features missing hain: "
                        + ", ".join(map(str, missing))
                    )
                    st.write("Expected features:", list(features))
                else:
                    X = pd.DataFrame(
                        [[values[f] for f in features]],
                        columns=features
                    )
                    prediction = model.predict(X)[0]
                    st.success(
                        f"🏏 Predicted Final Score: "
                        f"{round(float(prediction))} runs"
                    )
            else:
                st.error(
                    "Model ke feature names available nahi hain. "
                    "Training code ke input feature order ke "
                    "according prediction input set karein."
                )

        except Exception as e:
            st.error("Prediction failed. Check model and features.")
            st.code(str(e))

st.caption(
    "Cricket Score Predictor | Machine Learning Project"
)

