import numpy as np
import joblib
import streamlit as st
from datetime import datetime, date
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

roi_model = load_model('roi_model.h5', compile=False)
risk_model = load_model('risk_model.h5', compile=False)
scaler = joblib.load('scaler.pkl')

# Risk labels
risk_labels = {
    0: "Very Low Risk",
    1: "Low Risk",
    2: "Moderate Risk",
    3: "High Risk",
    4: "Very High Risk"
}

# Stock mapping
stock_options = {
    1: "NVDA (NVIDIA)",
    2: "AAPL (Apple)",
    3: "MSFT (Microsoft)",
    4: "GOOGL (Alphabet / Google)",
    5: "TSLA (Tesla)",
    6: "AMZN (Amazon)",
    7: "META (Meta / Facebook)",
    8: "JPM (JPMorgan Chase)",
    9: "XOM (ExxonMobil)",
    10: "NFLX (Netflix)",
    11: "BA (Boeing)",
    12: "DIS (Disney)",
    13: "GE (General Electric)",
    14: "WMT (Walmart)",
    15: "PEP (PepsiCo)",
    16: "KO (Coca-Cola)",
    17: "MCD (McDonald's)",
    18: "IBM (IBM)",
    19: "V (Visa)",
    20: "JNJ (Johnson & Johnson)"
}

# Reverse map for dropdown
stock_name_to_id = {v: k for k, v in stock_options.items()}

# Streamlit UI setup
st.set_page_config(page_title="Stock ROI & Risk Predictor", layout="centered")
st.title("📈 Stock ROI & Risk Predictor")
st.markdown("Use this tool to predict *Return on Investment (ROI)*, **Risk Level**, and your **Net Profit**.")

# Dropdown with full stock names
stock_choice = st.selectbox("Select a Stock", options=list(stock_name_to_id.keys()))
stock_id = stock_name_to_id[stock_choice]

# Define default dates safely
default_invest_date = date(2025, 1, 1)
default_takeout_date = date(2025, 2, 1)

# Safe date inputs
invest_date = st.date_input("Investment Date", min_value=date(2020, 1, 1), value=default_invest_date)
safe_takeout_default = max(invest_date, default_takeout_date)
takeout_date = st.date_input("Take-Out Date", min_value=invest_date, value=safe_takeout_default)

# Amount input
amount = st.number_input("Investment Amount ($)", min_value=1.0, value=1000.0, step=10.0)

# Convert dates to the number of days between them
def calculate_days(start_date, end_date):
    days = (end_date - start_date).days
    return max(days, 0)

# Predict button
if st.button("Predict"):
    days = calculate_days(invest_date, takeout_date)

    if days == 0:
        st.error("Please ensure the take-out date is after the investment date.")
    else:
        # Prepare input
        user_input = np.array([[stock_id, days, amount]])
        user_input_scaled = scaler.transform(user_input)

        # Predict using separated models
        log_roi_pred = roi_model.predict(user_input_scaled)
        risk_pred = risk_model.predict(user_input_scaled)

        # Convert predictions
        roi_percent = np.expm1(log_roi_pred[0][0])  # ROI was trained in log1p space
        risk_class = np.argmax(risk_pred)
        risk_label = risk_labels[risk_class]

        # Calculate profit and total value
        profit = (roi_percent / 100) * amount
        new_total = amount + profit

        # Display results
        st.success(f"**Predicted ROI:** {roi_percent:.2f}%")
        st.info(f"**Risk Category:** {risk_label} (Class {risk_class})")
        st.markdown(f"**Net Profit:** ${profit:.2f}")
        st.markdown(f"**Total After Profit:** ${new_total:.2f}")

        # Risk distribution chart
        st.subheader("Risk Probability Distribution")

        # Plotting risk distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=np.arange(len(risk_pred[0])), y=risk_pred[0], ax=ax)
        ax.set_title('Risk Prediction Distribution')
        ax.set_xlabel('Risk Class')
        ax.set_ylabel('Probability')
        ax.set_xticks(np.arange(len(risk_pred[0])))
        ax.set_xticklabels([f"Class {i}" for i in range(len(risk_pred[0]))])
        st.pyplot(fig)
