# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Generate Dummy Data for Financial Statements
np.random.seed(42)
years = range(2020, 2031)
scenarios = ['Baseline', 'Adverse', 'Severe Adverse']

data = {
    'Year': [year for year in years for _ in scenarios],
    'Scenario': scenarios * len(years),
    'Revenue': np.random.randint(500, 1000, len(years) * len(scenarios)),
    'Expenses': np.random.randint(300, 700, len(years) * len(scenarios)),
    'Loan Defaults': np.random.randint(50, 150, len(years) * len(scenarios)),
    'Macroeconomic Factor': np.random.rand(len(years) * len(scenarios)) * 5,
}

financial_df = pd.DataFrame(data)
financial_df['Net Profit'] = financial_df['Revenue'] - financial_df['Expenses'] - financial_df['Loan Defaults']

print("Dummy Financial Statements Data:")
print(financial_df.head())

# 2. Automate Regression Analysis
X = financial_df[['Macroeconomic Factor']]
y = financial_df['Net Profit']

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Regression
reg_model = LinearRegression()
reg_model.fit(X_scaled, y)
predictions = reg_model.predict(X_scaled)

print("\nRegression Analysis:")
print(f"Coefficients: {reg_model.coef_}")
print(f"Intercept: {reg_model.intercept_}")
print(f"MSE: {mean_squared_error(y, predictions)}")

# 3. Fee Income Modeling
fee_data = {
    'Year': years,
    'Scenario': ['Baseline'] * len(years),
    'Private Banking Revenue': np.random.randint(100, 200, len(years)),
    'Investment Banking Revenue': np.random.randint(50, 150, len(years)),
}

fee_df = pd.DataFrame(fee_data)
fee_df['Fee Income'] = fee_df['Private Banking Revenue'] * 0.02 + fee_df['Investment Banking Revenue'] * 0.01

print("\nFee Income Modeling:")
print(fee_df.head())

# 4. Climate Change Scenario Analysis
climate_scenarios = ['Scenario A', 'Scenario B', 'Scenario C']
climate_data = {
    'Scenario': climate_scenarios,
    'Temperature Rise': [1.5, 2.0, 3.0],
    'Sea Level Rise (cm)': [20, 30, 50],
    'Economic Impact (%)': [-0.5, -1.0, -2.0],
}

climate_df = pd.DataFrame(climate_data)

print("\nClimate Change Scenario Analysis:")
print(climate_df)

# 5. Early Warning Indicators and Monitoring
ewi_data = {
    'Indicator': ['Liquidity Ratio', 'Capital Adequacy', 'NPA Ratio'],
    'Threshold': [1.2, 10.0, 5.0],
    'Current Value': [np.random.uniform(1.0, 2.0), np.random.uniform(9.0, 12.0), np.random.uniform(4.0, 6.0)],
}

ewi_df = pd.DataFrame(ewi_data)
ewi_df['Alert'] = ewi_df['Current Value'] < ewi_df['Threshold']

print("\nEarly Warning Indicators:")
print(ewi_df)

# 6. Management Information (MI) for Climate Change Exercise
mi_df = climate_df.copy()
mi_df['Top Management View'] = ['Action Required' if x < 0 else 'Stable' for x in mi_df['Economic Impact (%)']]

print("\nManagement Information for Climate Change Exercise:")
print(mi_df)


# Save all DataFrames to CSV files
financial_df.to_csv("financial_statements.csv", index=False)
print("Saved financial statements to financial_statements.csv")

fee_df.to_csv("fee_income_modeling.csv", index=False)
print("Saved fee income modeling to fee_income_modeling.csv")

climate_df.to_csv("climate_change_scenarios.csv", index=False)
print("Saved climate change scenarios to climate_change_scenarios.csv")

ewi_df.to_csv("early_warning_indicators.csv", index=False)
print("Saved early warning indicators to early_warning_indicators.csv")

mi_df.to_csv("management_information_climate.csv", index=False)
print("Saved management information for climate change to management_information_climate.csv")
