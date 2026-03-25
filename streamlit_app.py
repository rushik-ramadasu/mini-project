import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Marketing Optimization", layout="wide")

st.title("Marketing Optimization Prediction API")
st.markdown("This application predicts the likelihood of a customer converting based on marketing features and their profile.")

# Load models
@st.cache_resource
def load_models():
    # Make sure we are in the correct directory, or use absolute paths relative to the script if needed.
    # We'll assume the script is run from the project root where 'models/' exists.
    try:
        xgb_model = joblib.load('models/xgb_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        kmeans = joblib.load('models/kmeans.joblib')
        
        with open('models/features.txt', 'r') as f:
            features = f.read().strip().split(',')
        return xgb_model, scaler, kmeans, features
    except Exception as e:
        st.error(f"Error loading models. Please make sure you are running the app from the project root directory. Details: {e}")
        st.stop()

xgb_model, scaler, kmeans, features = load_models()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Details")
    age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0, step=1.0)
    income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
    has_active_loan = st.selectbox("Has Active Loan", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
with col2:
    st.subheader("Marketing & Macro factors")
    campaign_intensity = st.number_input("Campaign Intensity", min_value=1.0, value=1.0, step=1.0)
    call_duration = st.number_input("Call Duration (seconds)", min_value=0.0, value=120.0, step=10.0)
    inflation_rate = st.number_input("Inflation Rate (%)", value=2.5, step=0.1)
    unemployment_rate = st.number_input("Unemployment Rate (%)", value=5.0, step=0.1)

if st.button("Predict Conversion", type="primary", use_container_width=True):
    with st.spinner("Calculating prediction..."):
        try:
            # Prepare input data
            input_data = {
                'age': age,
                'income': income,
                'has_active_loan': has_active_loan,
                'campaign_intensity': campaign_intensity,
                'call_duration': call_duration,
                'inflation_rate': inflation_rate,
                'unemployment_rate': unemployment_rate
            }
            
            df = pd.DataFrame([input_data])
            
            # Segment customer
            seg_features = df[['age', 'income', 'call_duration', 'has_active_loan']]
            df['segment'] = kmeans.predict(seg_features)
            
            # We need to scale num_cols for ML
            num_cols = ['age', 'income', 'campaign_intensity', 'call_duration', 'inflation_rate', 'unemployment_rate']
            
            df_scaled = df.copy()
            df_scaled[num_cols] = scaler.transform(df[num_cols])
            
            # Reorder to match training
            X = df_scaled[features]
            
            # Prediction
            prob = xgb_model.predict_proba(X)[0][1]
            segment_assigned = int(df['segment'][0])
            
            # Results display
            st.divider()
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Conversion Probability", value=f"{prob:.1%}")
            with res_col2:
                st.metric(label="Assigned Customer Segment", value=f"Segment {segment_assigned}")
                
            # SHAP calc
            st.subheader("Decision Reasoning (SHAP Values)")
            st.markdown("This chart explains how each feature influenced the model's prediction. Red bars push the probability higher, green bars push it lower.")
            
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X)
            
            # Extract values for the single instance
            # Some versions of SHAP return a list for binary classification, some return a 2D array
            if isinstance(shap_values, list):
                s_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                # If it's a 2D array, we just take the first row
                s_vals = shap_values[0]
                if len(s_vals.shape) > 0 and s_vals.shape[0] == 2:
                    # If it's still somehow (2, num_features)
                    s_vals = shap_values[1] if len(shap_values.shape) > 2 else shap_values[0]
            
            # The API assumes shap_values[0] so we use the same logic that works in the fastAPI application:
            # Format SHAP explanation (shap_values[0] works in api_app.py)
            shap_dict = {feat: float(val) for feat, val in zip(features, shap_values[0])}
            
            # Create a dataframe for plotting
            shap_df = pd.DataFrame(list(shap_dict.items()), columns=['Feature', 'SHAP Value'])
            shap_df = shap_df.sort_values(by='SHAP Value', key=abs, ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, max(5, len(features)*0.5)))
            
            # Colors: red for positive impact, blue for negative impact
            colors = ['#ff0051' if val > 0 else '#008bfb' for val in shap_df['SHAP Value']]
            
            ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
            ax.set_xlabel('SHAP Value (Impact on Prediction)')
            ax.set_title('Feature Importance for This Customer')
            
            # Formatting the plot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
