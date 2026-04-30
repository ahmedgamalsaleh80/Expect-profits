"""
data_generator.py
-----------------
Generates a synthetic (fake but realistic) financial dataset.
We use NumPy to create random numbers that represent company monthly data.
"""

import numpy as np
import pandas as pd

def generate_data(n_samples=120, random_seed=42):
    """
    Creates fake monthly company financial data.
    n_samples = number of months (rows)
    random_seed = fixes randomness so we always get the same data
    """

    # Fix the random seed so results are reproducible
    np.random.seed(random_seed)

    # --- FEATURE 1: Month number (1 to n_samples) ---
    month = np.arange(1, n_samples + 1)

    # --- FEATURE 2: Revenue (money coming IN) ---
    # Base revenue between 50,000 and 200,000
    # We add a small upward trend over time (month * 500)
    revenue = np.random.uniform(50000, 200000, n_samples) + month * 500

    # --- FEATURE 3: Expenses (money going OUT) ---
    # Expenses are roughly 40% to 70% of revenue
    expenses = revenue * np.random.uniform(0.4, 0.7, n_samples)

    # --- FEATURE 4: Marketing Cost ---
    # Between 2000 and 15000 per month
    marketing_cost = np.random.uniform(2000, 15000, n_samples)

    # --- FEATURE 5: Number of Customers ---
    # Between 100 and 1000 customers per month
    num_customers = np.random.randint(100, 1000, n_samples).astype(float)

    # --- TARGET: Profit = Revenue - Expenses - Marketing Cost ---
    profit = revenue - expenses - marketing_cost

    # --- FEATURE 6: Previous Month's Profit ---
    # For month 1, previous profit = 0 (no history)
    previous_profit = np.concatenate([[0], profit[:-1]])

    # --- Add some noise (random ups and downs) to profit ---
    noise = np.random.normal(0, 2000, n_samples)
    profit = profit + noise

    # --- Create the DataFrame (table) ---
    df = pd.DataFrame({
        'Month':           month,
        'Revenue':         np.round(revenue, 2),
        'Expenses':        np.round(expenses, 2),
        'Marketing_Cost':  np.round(marketing_cost, 2),
        'Num_Customers':   num_customers,
        'Previous_Profit': np.round(previous_profit, 2),
        'Profit':          np.round(profit, 2)
    })

    # --- Create binary label: 1 = Profit, 0 = Loss ---
    df['Profit_Label'] = (df['Profit'] > 0).astype(int)

    return df


# --- Run this file directly to test it ---
if __name__ == "__main__":
    df = generate_data()
    print("Dataset Shape:", df.shape)       # Should be (120, 8)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nProfit vs Loss counts:")
    print(df['Profit_Label'].value_counts())
    df.to_csv("company_data.csv", index=False)
    print("\nSaved to company_data.csv")
