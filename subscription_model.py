import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from prophet import Prophet
import shap

class SubscriptionPredictor:
    def __init__(self, data_path='data/synthetic_marketing_data.csv'):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.ml_model = XGBClassifier(eval_metric='logloss', random_state=42)
        self.ts_model = None # Prophet
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        
    def load_and_preprocess(self):
        self.df = pd.read_csv(self.data_path)
        self.df['interaction_date'] = pd.to_datetime(self.df['interaction_date'])
        self.df['year_month'] = pd.to_datetime(self.df['year_month'])
        
        # Feature Engineering: Lag features for TS later
        self.df = self.df.sort_values(by=['customer_id', 'interaction_date'])
        
    def segment_customers(self):
        # Base segmentation on age, income, and avg call duration
        seg_features = self.df.groupby('customer_id').agg({
            'age': 'first',
            'income': 'first',
            'call_duration': 'mean',
            'has_active_loan': 'first'
        }).reset_index()
        
        X_seg = self.scaler.fit_transform(seg_features[['age', 'income', 'call_duration', 'has_active_loan']])
        seg_features['segment'] = self.kmeans.fit_predict(X_seg)
        
        self.df = self.df.merge(seg_features[['customer_id', 'segment']], on='customer_id', how='left')
        print("Customer Segmentation Complete.")
        
    def train_ml_model(self):
        features = ['age', 'income', 'has_active_loan', 'campaign_intensity', 
                    'call_duration', 'inflation_rate', 'unemployment_rate', 'segment']
        target = 'conversion'
        
        X = self.df[features]
        y = self.df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling numericals
        num_cols = ['age', 'income', 'campaign_intensity', 'call_duration', 'inflation_rate', 'unemployment_rate']
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[num_cols] = self.scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = self.scaler.transform(X_test[num_cols])
        
        self.ml_model.fit(X_train_scaled, y_train)
        
        preds = self.ml_model.predict(X_test_scaled)
        print(f"ML Model Accuracy: {accuracy_score(y_test, preds):.2f}")
        print(classification_report(y_test, preds))
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.ml_model, 'models/xgb_model.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        joblib.dump(self.kmeans, 'models/kmeans.joblib')
        features_path = 'models/features.txt'
        with open(features_path, 'w') as f:
            f.write(','.join(features))

    def train_ts_model(self):
        # Aggregate conversion rate by month
        monthly = self.df.groupby('year_month').agg({
            'conversion': 'mean',
            'inflation_rate': 'mean',
            'unemployment_rate': 'mean'
        }).reset_index()
        
        # Prepare for Prophet
        prophet_df = monthly.rename(columns={'year_month': 'ds', 'conversion': 'y'})
        
        self.ts_model = Prophet()
        self.ts_model.add_regressor('inflation_rate')
        self.ts_model.add_regressor('unemployment_rate')
        
        self.ts_model.fit(prophet_df)
        print("Time-Series Model Trained.")
        
        # Save model config (Prophet isn't trivial to pickle sometimes, but we will try, or save separately if needed) # We'll just train on the fly or keep in memory. For now pickling it.
        import pickle
        with open('models/prophet_model.pkl', 'wb') as f:
            pickle.dump(self.ts_model, f)
            
    def run_all(self):
        self.load_and_preprocess()
        self.segment_customers()
        self.train_ml_model()
        self.train_ts_model()
        print("All models trained and saved to /models")

if __name__ == "__main__":
    predictor = SubscriptionPredictor()
    predictor.run_all()
