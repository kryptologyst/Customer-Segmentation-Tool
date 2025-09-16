"""
Mock database module for customer segmentation tool.
Generates realistic customer data for testing and demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class CustomerDatabase:
    def __init__(self, num_customers=1000):
        self.num_customers = num_customers
        self.data_file = 'customers.csv'
        np.random.seed(42)  # For reproducible results
        
    def generate_customer_data(self):
        """Generate realistic customer data with multiple segments."""
        
        # Define customer segments with different characteristics
        segments = {
            'Young Professionals': {
                'age_range': (25, 35),
                'income_range': (40000, 80000),
                'spending_range': (60, 90),
                'frequency_range': (15, 30),
                'size': 0.25
            },
            'Middle-aged Families': {
                'age_range': (35, 50),
                'income_range': (50000, 120000),
                'spending_range': (40, 70),
                'frequency_range': (10, 25),
                'size': 0.35
            },
            'Seniors': {
                'age_range': (55, 75),
                'income_range': (30000, 70000),
                'spending_range': (20, 50),
                'frequency_range': (5, 15),
                'size': 0.20
            },
            'High Earners': {
                'age_range': (30, 60),
                'income_range': (100000, 200000),
                'spending_range': (80, 100),
                'frequency_range': (20, 40),
                'size': 0.20
            }
        }
        
        customers = []
        customer_id = 1
        
        for segment_name, params in segments.items():
            segment_size = int(self.num_customers * params['size'])
            
            for _ in range(segment_size):
                # Generate customer data based on segment characteristics
                age = np.random.randint(params['age_range'][0], params['age_range'][1])
                annual_income = np.random.randint(params['income_range'][0], params['income_range'][1])
                spending_score = np.random.randint(params['spending_range'][0], params['spending_range'][1])
                purchase_frequency = np.random.randint(params['frequency_range'][0], params['frequency_range'][1])
                
                # Add some realistic variations
                gender = np.random.choice(['Male', 'Female'])
                location = np.random.choice(['Urban', 'Suburban', 'Rural'])
                membership_years = np.random.randint(1, 10)
                
                # Calculate derived metrics
                avg_order_value = (annual_income * 0.001) + np.random.normal(0, 20)
                avg_order_value = max(10, avg_order_value)  # Minimum order value
                
                total_spent = avg_order_value * purchase_frequency
                
                customer = {
                    'CustomerID': customer_id,
                    'Age': age,
                    'Gender': gender,
                    'AnnualIncome': annual_income,
                    'SpendingScore': spending_score,
                    'PurchaseFrequency': purchase_frequency,
                    'AvgOrderValue': round(avg_order_value, 2),
                    'TotalSpent': round(total_spent, 2),
                    'Location': location,
                    'MembershipYears': membership_years,
                    'LastPurchaseDate': self._generate_last_purchase_date(),
                    'TrueSegment': segment_name
                }
                
                customers.append(customer)
                customer_id += 1
        
        return pd.DataFrame(customers)
    
    def _generate_last_purchase_date(self):
        """Generate a realistic last purchase date."""
        days_ago = np.random.randint(1, 365)
        last_purchase = datetime.now() - timedelta(days=days_ago)
        return last_purchase.strftime('%Y-%m-%d')
    
    def save_data(self, df=None):
        """Save customer data to CSV file."""
        if df is None:
            df = self.generate_customer_data()
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        filepath = os.path.join('data', self.data_file)
        df.to_csv(filepath, index=False)
        print(f"Customer data saved to {filepath}")
        return df
    
    def load_data(self):
        """Load customer data from CSV file."""
        filepath = os.path.join('data', self.data_file)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} customers from {filepath}")
            return df
        else:
            print("No existing data found. Generating new customer data...")
            return self.save_data()
    
    def get_sample_data(self, n=100):
        """Get a sample of customer data for testing."""
        df = self.load_data()
        return df.sample(n=min(n, len(df)))
    
    def get_segment_summary(self):
        """Get summary statistics by true segments."""
        df = self.load_data()
        summary = df.groupby('TrueSegment').agg({
            'Age': ['mean', 'std'],
            'AnnualIncome': ['mean', 'std'],
            'SpendingScore': ['mean', 'std'],
            'PurchaseFrequency': ['mean', 'std'],
            'CustomerID': 'count'
        }).round(2)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary = summary.rename(columns={'CustomerID_count': 'Count'})
        
        return summary

if __name__ == "__main__":
    # Generate and save customer data
    db = CustomerDatabase(num_customers=1000)
    customer_data = db.save_data()
    
    print("\nDataset Overview:")
    print(f"Total customers: {len(customer_data)}")
    print(f"Features: {list(customer_data.columns)}")
    
    print("\nSample data:")
    print(customer_data.head())
    
    print("\nSegment Summary:")
    print(db.get_segment_summary())
