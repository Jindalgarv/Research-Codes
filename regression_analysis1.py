import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# Get current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")

# Load the data
file_path = '/Users/mansimittal/Desktop/efficiency_electrical_data.csv'

try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully")
    print(df.head())
except FileNotFoundError:
    print(f"File not found at {file_path}")
    exit()

# Print available columns for reference
print("Available columns:", df.columns.tolist())

# Rename columns for easier access (you can adjust this as needed)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Define your explanatory (X) and response (Y) variables
X = df[['total_assets', 'pbit', 'debt_to_equity_ratio_(times)', 'research_&_development_expenses']]
Y = df['efficiencyscore']
  # adjust this as well

# Combine X and Y, handle NaNs and Infs
data = pd.concat([X, Y], axis=1)
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

X_clean = data[X.columns]
Y_clean = data[Y.name]

# Add constant to X for intercept
X_clean = sm.add_constant(X_clean)

# Run the regression model
model = sm.OLS(Y_clean, X_clean).fit()

# Print the summary
print(model.summary())

