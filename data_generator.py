import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_customers=5000, start_date='2020-01-01', end_date='2023-12-31'):
    np.random.seed(42)
    
    # Generate dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = pd.date_range(start, end, freq='D')
    
    # Customer Base
    customer_ids = np.arange(1, num_customers + 1)
    
    # Generate macro indicators (monthly base, interpolated to daily)
    months = pd.date_range(start, end, freq='MS')
    inflation_base = np.linspace(2.0, 7.5, len(months)) + np.sin(np.linspace(0, 10, len(months))) # Simulated inflation trend
    unemployment_base = np.linspace(5.0, 3.5, len(months)) + np.cos(np.linspace(0, 8, len(months))) # Simulated unemployment trend
    
    macro_df = pd.DataFrame({
        'year_month': months.strftime('%Y-%m'),
        'inflation_rate': inflation_base,
        'unemployment_rate': unemployment_base
    })
    
    # Generate customer records
    # We simulate multiple interactions per customer over time
    num_interactions = num_customers * 3 # Average 3 interactions per customer
    
    interaction_dates = np.random.choice(date_range, num_interactions)
    interaction_customers = np.random.choice(customer_ids, num_interactions)
    
    # Features
    ages = np.random.normal(40, 12, num_customers).clip(18, 80).astype(int)
    incomes = np.random.normal(60000, 20000, num_customers).clip(20000, 200000)
    has_active_loan = np.random.choice([0, 1], num_customers, p=[0.7, 0.3])
    
    customer_df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'income': incomes,
        'has_active_loan': has_active_loan
    })
    
    df = pd.DataFrame({
        'customer_id': interaction_customers,
        'interaction_date': interaction_dates,
    })
    
    df = df.merge(customer_df, on='customer_id', how='left')
    
    # Campaign features
    df['campaign_intensity'] = np.random.uniform(0.1, 1.0, len(df))
    df['call_duration'] = np.random.exponential(scale=180, size=len(df)) # in seconds
    
    df['year_month'] = df['interaction_date'].dt.strftime('%Y-%m')
    df = df.merge(macro_df, on='year_month', how='left')
    
    # Conversion logic (Target)
    # Higher income, no loan, higher call duration, optimal campaign intensity -> higher conversion
    # Macro effects: high inflation/unemployment -> lower conversion
    # Seasonal effect: higher in Q4
    
    base_prob = 0.1
    
    # Normalize features for probability calculation
    inc_score = (df['income'] - df['income'].min()) / (df['income'].max() - df['income'].min())
    loan_penalty = df['has_active_loan'] * 0.1
    call_score = np.clip(df['call_duration'] / 600, 0, 0.3) # up to 10 mins adds 30% prob
    camp_score = df['campaign_intensity'] * 0.1
    macro_penalty = (df['inflation_rate']/10 * 0.05) + (df['unemployment_rate']/10 * 0.05)
    seasonality = np.where(df['interaction_date'].dt.month >= 10, 0.05, 0)
    
    prob = np.clip(base_prob + inc_score*0.1 - loan_penalty + call_score + camp_score - macro_penalty + seasonality, 0.01, 0.99)
    
    df['conversion'] = np.random.binomial(1, prob)
    
    df = df.sort_values('interaction_date').reset_index(drop=True)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_marketing_data.csv', index=False)
    print(f"Generated {len(df)} records for {num_customers} customers.")
    print(f"Overall conversion rate: {df['conversion'].mean():.2%}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
