import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Load datasets
try:
    train_data = pd.read_csv('iris_train.csv')
    inference_data = pd.read_csv('iris_inference.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure iris_train.csv and iris_inference.csv exist.")
    exit(1)

# Perform KS test for each feature
results = []
for column in train_data.columns:
    train_feature = train_data[column]
    inference_feature = inference_data[column]
    d_stat, p_value = ks_2samp(train_feature, inference_feature)
    results.append({
        'Feature': column,
        'D-statistic': d_stat,
        'p-value': p_value,
        'Skewness': 'Significant' if p_value < 0.05 else 'Not Significant'
    })

# Print and save KS test results
results_df = pd.DataFrame(results)
print("\nKS Test Results:")
print(results_df)
results_df.to_csv('ks_test_results.csv', index=False)
print("KS test results saved to ks_test_results.csv")

# Create output directory for plots
os.makedirs('cdf_plots', exist_ok=True)

# Plot CDF for each feature
for column in train_data.columns:
    train_feature = train_data[column]
    inference_feature = inference_data[column]
    
    # Compute CDFs
    train_sorted = np.sort(train_feature)
    inference_sorted = np.sort(inference_feature)
    train_cdf = np.arange(1, len(train_sorted) + 1) / len(train_sorted)
    inference_cdf = np.arange(1, len(inference_sorted) + 1) / len(inference_sorted)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(train_sorted, train_cdf, label='Training', color='blue')
    plt.plot(inference_sorted, inference_cdf, label='Inference', color='orange')
    plt.title(f'CDF of {column}')
    plt.xlabel(column)
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    safe_column = column.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'cdf_plots/cdf_{safe_column}.png')
    plt.close()

print("CDF plots saved in 'cdf_plots' folder.")