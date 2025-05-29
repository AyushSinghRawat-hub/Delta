import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import subprocess

# Load Iris dataset (training data)
iris = load_iris()
train_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Generate synthetic inference dataset with slight distribution shifts
np.random.seed(42)  # For reproducibility
inference_data = iris.data.copy()

# Add noise to simulate drift
inference_data[:, 0] += np.random.normal(0, 0.5, size=inference_data.shape[0])  # Sepal length
inference_data[:, 1] += np.random.normal(0, 0.3, size=inference_data.shape[0])  # Sepal width
inference_data[:, 2] += np.random.normal(0, 0.4, size=inference_data.shape[0])  # Petal length
inference_data[:, 3] += np.random.normal(0, 0.2, size=inference_data.shape[0])  # Petal width


inference_data = pd.DataFrame(data=inference_data, columns=iris.feature_names)

# Save datasets to CSV for reference
train_data.to_csv('iris_train.csv', index=False)
inference_data.to_csv('iris_inference.csv', index=False)

print("Training and inference datasets prepared and saved.")

# Automatically run the test and visualization script
print("Running KS test and visualization...")
try:
    subprocess.run(['python', 'test_and_visualize.py'], check=True)
    print("KS test and visualization completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error running test_and_visualize.py: {e}")
except FileNotFoundError:
    print("Error: test_and_visualize.py not found. Ensure it is in the same directory.") 