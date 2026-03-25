```python
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Marketing Optimization System", layout="wide")

st.title("AI-Powered Marketing Optimization System")
st.markdown("Predict customer subscription intent and forecast conversions.")

# =========================
# LOAD MODELS (NO API NEEDED)
# =========================
xgb_model = joblib.load("models/xgb_model.joblib")
scaler = joblib.load("models/scaler.joblib")
kmeans = joblib.load("models/kmeans.joblib")

with open("models/features.txt", "r") as f:
    features = f.read().strip().split(',')

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Scenario Simulator")
age = st.sidebar.slider("Age", 18, 80, 40)
income = st.sidebar.number_input("Income", 20000, 200000, 60000)
has_active_loan = st.sidebar.selectbox("Has Active Loan", [0, 1])
campaign_intensity = st.sidebar.slider("Campaign Intensity (0.0 - 1.0)", 0.0, 1.0, 0.5)
call_duration = st.sidebar.slider("Call Duration (seconds)", 0, 1200, 180)
inflation_rate = st.sidebar.slider("Current Inflation Rate (%)", 1.0, 10.0, 3.5)
unemployment_rate = st.sidebar.slider("Current Unemployment Rate (%)", 1.0, 10.0, 4.0)

# =========================
# PREDICTION LOGIC
# =========================
if st.sidebar.button("Simulate Conversion"):
    try:
        input_data = {
            "age": age,
            "income": income,
            "has_active_loan": has_active_loan,
            "campaign_intensity": campaign_intensity,
            "call_duration": call_duration,
            "inflation_rate": inflation_rate,
            "unemployment_rate": unemployment_rate
        }

        df = pd.DataFrame([input_data])

        # -------- Segmentation --------
        seg_features = df[['age', 'income', 'call_duration', 'has_active_loan']]
        df['segment'] = kmeans.predict(seg_features)

        # -------- Scaling --------
        num_cols = ['age', 'income', 'campaign_intensity', 'call_duration', 'inflation_rate', 'unemployment_rate']
        df_scaled = df.copy()
        df_scaled[num_cols] = scaler.transform(df[num_cols])

        # -------- Prediction --------
        X = df_scaled[features]
        prob = xgb_model.predict_proba(X)[0][1]

        # -------- SHAP --------
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X)
        shap_dict = {feat: float(val) for feat, val in zip(features, shap_values[0])}

        # =========================
        # DISPLAY RESULTS
        # =========================
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Conversion Probability", value=f"{prob:.2%}")
        with col2:
            st.metric(label="Customer Segment", value=f"Segment {int(df['segment'][0])}")

        st.subheader("Feature Influence (SHAP Values)")
        st.write("Positive values increase conversion probability.")

        shap_df = pd.DataFrame(list(shap_dict.items()), columns=['Feature', 'SHAP Value'])
        shap_df['Positive Impact'] = shap_df['SHAP Value'] > 0

        fig, ax = plt.subplots(figsize=(10, 5))
        shap_df.sort_values('SHAP Value', inplace=True)
        colors = shap_df['Positive Impact'].map({True: 'green', False: 'red'})
        ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
        st.pyplot(fig)

        # =========================
        # RECOMMENDATIONS
        # =========================
        st.subheader("Decision Support Recommendations")

        if shap_dict.get('call_duration', 0) < 0:
            st.info("Increase call duration to improve conversion probability.")

        if shap_dict.get('campaign_intensity', 0) < 0:
            st.info("Adjust campaign intensity for better engagement.")

        if prob > 0.7:
            st.success("High potential customer. Prioritize outreach.")
        elif prob < 0.3:
            st.warning("Low probability customer. Consider alternative strategies.")

    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.subheader("Time-Series Forecast")

# =========================
# TIME SERIES FORECAST
# =========================
try:
    ts_model = joblib.load("models/prophet_model.pkl")

    future = ts_model.make_future_dataframe(periods=12, freq='M')
    future['inflation_rate'] = 4.0
    future['unemployment_rate'] = 4.5

    forecast = ts_model.predict(future)

    fig2 = ts_model.plot(forecast)
    st.pyplot(fig2)

except Exception as e:
    st.write("Could not load time-series forecast.")
    st.write(str(e))
```
