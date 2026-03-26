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
    age = st.slider("Age", min_value=18.0, max_value=100.0, value=35.0, step=1.0)
    income = st.slider("Income", min_value=0.0, max_value=500000.0, value=50000.0, step=1000.0)
    has_active_loan = st.selectbox("Has Active Loan", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
with col2:
    st.subheader("Marketing & Macro factors")
    campaign_intensity = st.slider("Campaign Intensity", min_value=1.0, max_value=10.0, value=1.0, step=1.0)
    call_duration = st.slider("Call Duration (seconds)", min_value=0.0, max_value=1800.0, value=120.0, step=10.0)
    inflation_rate = st.slider("Inflation Rate (%)", min_value=-5.0, max_value=20.0, value=2.5, step=0.1)
    unemployment_rate = st.slider("Unemployment Rate (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)

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
            
            # --- PRESCRIPTIVE RECOMMENDATIONS ---
            st.subheader("Actionable Insights 💡")
            
            best_prob = prob
            best_action = None
            
            # Run a mini simulation to find the "Next Best Action"
            for ci_bump in [0, 1, 2, 3]:
                for cd_bump in [0, 30, 60, 120, 240, 360]:
                    if ci_bump == 0 and cd_bump == 0:
                        continue
                        
                    test_ci = min(10.0, campaign_intensity + ci_bump)
                    test_cd = min(1800.0, call_duration + cd_bump)
                    
                    test_df = df.copy()
                    test_df['campaign_intensity'] = test_ci
                    test_df['call_duration'] = test_cd
                    
                    test_df_scaled = test_df.copy()
                    test_df_scaled[num_cols] = scaler.transform(test_df[num_cols])
                    
                    test_X = test_df_scaled[features]
                    test_prob = xgb_model.predict_proba(test_X)[0][1]
                    
                    # Store if it improves conversion significantly
                    if test_prob > best_prob + 0.01:
                        best_prob = test_prob
                        best_action = (test_ci, test_cd, ci_bump, cd_bump)
            
            if best_action:
                test_ci, test_cd, ci_bump, cd_bump = best_action
                improvement = best_prob - prob
                
                action_text = []
                if ci_bump > 0:
                    action_text.append(f"increasing **Campaign Intensity** to {int(test_ci)} (+{ci_bump} level{'s' if ci_bump > 1 else ''})")
                if cd_bump > 0:
                    action_text.append(f"lengthening **Call Duration** to {int(test_cd)}s (+{cd_bump}s)")
                    
                action_str = " and ".join(action_text)
                
                if best_prob > 0.6:
                    st.success(f"📈 **Optimization Found:** By {action_str}, you could boost the probability by **{improvement:.1%}** (to **{best_prob:.1%}**).")
                else:
                    st.warning(f"⚠️ **Tough Lead:** You can improve odds slightly. By {action_str}, you could boost the probability by **{improvement:.1%}** (to **{best_prob:.1%}**).")
            else:
                if prob > 0.7:
                    st.success("✅ **Optimal Strategy:** This customer is already highly optimized based on current inputs. Proceed with the campaign!")
                else:
                    st.info("ℹ️ **Insight:** Adjusting call duration or campaign intensity won't significantly improve the conversion probability for this specific profile. Consider alternative segments or offers.")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            '''
            vibecoded the forntend and deployment part mainly this streamliy_app.py :)
            '''
