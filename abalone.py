import sys
import traceback
import argparse
import json
import hashlib
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import joblib

# Force flush stdout
sys.stdout.flush()

def parse_args():
    parser = argparse.ArgumentParser(description="Run Abalone pipeline with versioning.")
    parser.add_argument("--data-path", required=True, help="Path to Abalone dataset CSV file")
    return parser.parse_args()

def compute_data_hash(data_path):
    try:
        with open(data_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        print(f"Error computing hash for {data_path}: {e}")
        sys.exit(1)

def get_latest_version():
    version_folders = [d for d in os.listdir() if os.path.isdir(d) and re.match(r'^v\d+$', d)]
    if not version_folders:
        return 0
    version_numbers = [int(d.replace('v', '')) for d in version_folders]
    return max(version_numbers)

def load_metadata(version_folder):
    metadata_path = os.path.join(version_folder, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def bin_rings(rings):
    """Bin Rings into classes: 0 (Young), 1 (Middle-aged), 2 (Old)."""
    if rings <= 8:
        return 0
    elif rings <= 11:
        return 1
    else:
        return 2

def main():
    # Parse arguments
    args = parse_args()
    data_path = args.data_path

    print("Starting Abalone pipeline...")

    # Compute dataset hash
    data_hash = compute_data_hash(data_path)

    # Check existing versions
    latest_version = get_latest_version()
    version = latest_version + 1
    for v in range(1, latest_version + 1):
        version_folder = f'v{v}'
        metadata = load_metadata(version_folder)
        if metadata and metadata.get('data_hash') == data_hash:
            print(f"Dataset unchanged, reusing version {version_folder}")
            return
        else:
            print(f"Dataset changed, creating new version v{version}")

    # Create version folder
    version_folder = f'v{version}'
    os.makedirs(version_folder, exist_ok=True)
    print(f"Outputs will be saved in {version_folder}/")

    # Define configuration
    config = {
        "noise_std": [0.012, 0.01, 0.004, 0.049, 0.022, 0.013, 0.014],  # Proportional to feature std
        "model_type": "RandomForest"
    }
    # 1. Load and Preprocess Data
    print("Loading and preprocessing Abalone dataset...")
    try:
        data = pd.read_csv(data_path)
        expected_columns = [
            'Sex', 'Length', 'Diameter', 'Height',
            'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
        ]
        if not all(col in data.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in data.columns]
            raise ValueError(f"Missing columns in CSV: {missing}")

        # One-hot encode Sex
        data = pd.get_dummies(data, columns=['Sex'], prefix='Sex')
        feature_columns = [
            'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
            'Viscera weight', 'Shell weight', 'Sex_F', 'Sex_I', 'Sex_M'
        ]
        X = data[feature_columns]
        # Bin Rings into classes
        y = data['Rings'].apply(bin_rings)
    except Exception as e:
        print(f"Error loading {data_path}: {str(e)}")
        sys.exit(1)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

    # Save preprocessed dataset
    preprocessed_data = X_scaled_df.copy()
    preprocessed_data['Rings_class'] = y
    preprocessed_data.to_csv(os.path.join(version_folder, 'preprocessed.csv'), index=False)
    print(f"Preprocessed dataset saved as {version_folder}/preprocessed.csv")

    # 2. Extract Featurization
    print(f"Extracting featurization...")
    features = X_scaled_df.copy()
    features.to_csv(os.path.join(version_folder, 'features.csv'), index=False)
    print(f"Featurization saved as {version_folder}/features.csv")

    # 3. Prepare Training Dataset
    print(f"Preparing training dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )
    train_data = X_train.copy()
    train_data['Rings_class'] = y_train
    val_data = X_val.copy()
    val_data['Rings_class'] = y_val
    train_data.to_csv(os.path.join(version_folder, 'train.csv'), index=False)
    val_data.to_csv(os.path.join(version_folder, 'val.csv'), index=False)
    print(f"Training and validation datasets saved as {version_folder}/train.csv and {version_folder}/val.csv")

    # 4. Train Model
    print(f"Training RandomForestClassifier...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, os.path.join(version_folder, 'model.pkl'))
    print(f"Model saved as {version_folder}/model.pkl")

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Training accuracy: {train_accuracy:.2f}")
    print(f"Validation accuracy: {val_accuracy:.2f}")

    # 5. Generate Inference Data with Skew
    print(f"Generating inference data with skew...")
    inference_data = X.values.copy()
    # Apply noise only to numeric features (first 7 columns)
    for i, std in enumerate(config['noise_std']):
        inference_data[:, i] += np.random.normal(0, std, size=inference_data.shape[0])
    
    # Scale inference data
    inference_data_scaled = scaler.transform(inference_data)
    inference_data_df = pd.DataFrame(inference_data_scaled, columns=feature_columns)
    inference_data_df.to_csv(os.path.join(version_folder, 'inference.csv'), index=False)
    print(f"Inference dataset saved as {version_folder}/inference.csv")

    # 6. Make Predictions
    print(f"Making predictions...")
    inference_pred = model.predict(inference_data_df)
    inference_accuracy = accuracy_score(y, inference_pred)
    print(f"Inference accuracy: {inference_accuracy:.2f}")

    # 7. Visualization and KS Test
    print(f"Performing KS test and visualizing distributions...")

    # KS Test
    results = []
    for column in X_train.columns:
        train_feature = X_train[column]
        inference_feature = inference_data_df[column]
        d_stat, p_value = ks_2samp(train_feature, inference_feature)
        results.append({
            'Feature': column,
            'D-statistic': d_stat,
            'p-value': p_value,
            'Skewness': 'Significant' if p_value < 0.05 else 'Not Significant'
        })
    results_df = pd.DataFrame(results)
    print(f"\nKS Test Results:")
    print(results_df)
    results_df.to_csv(os.path.join(version_folder, 'ks_test_results.csv'), index=False)
    print(f"KS test results saved to {version_folder}/ks_test_results.csv")

    # Histogram Plots
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(feature_columns[:7]):  # Only numeric features
        plt.subplot(3, 3, i + 1)
        plt.hist(X_train[feature], bins=20, alpha=0.5, label='Train', color='blue')
        plt.hist(inference_data_df[feature], bins=20, alpha=0.5, label='Inference', color='orange')
        plt.title(f"{feature}")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(version_folder, 'histograms.png'))
    plt.close()
    print(f"Histogram plots saved as {version_folder}/histograms.png")

    # CDF Plots
    cdf_folder = os.path.join(version_folder, 'cdf_plots')
    os.makedirs(cdf_folder, exist_ok=True)
    for column in feature_columns:
        train_feature = X_train[column]
        inference_feature = inference_data_df[column]
        train_sorted = np.sort(train_feature)
        inference_sorted = np.sort(inference_feature)
        train_cdf = np.arange(1, len(train_sorted) + 1) / len(train_sorted)
        inference_cdf = np.arange(1, len(inference_sorted) + 1) / len(inference_sorted)
        
        plt.figure(figsize=(8, 6))
        plt.plot(train_sorted, train_cdf, label='Training', color='blue')
        plt.plot(inference_sorted, inference_cdf, label='Inference', color='orange')
        plt.title(f'CDF of {column}')
        plt.xlabel(column)
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.grid(True)
        
        safe_column = column.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(cdf_folder, f'cdf_{safe_column}.png'))
        plt.close()
    print(f"CDF plots saved in {version_folder}/cdf_plots/")

    # Save metadata
    metadata = {
        "version": f"v{version}",
        "data_hash": data_hash,
        "data_path": data_path,
        "noise_std": config["noise_std"],
        "model_type": config["model_type"],
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "inference_accuracy": inference_accuracy
    }
    with open(os.path.join(version_folder, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved as {version_folder}/metadata.json")

    print(f"Pipeline completed successfully for v{version}.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        sys.exit(1)