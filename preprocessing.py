import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def run(config):
    # Load dataset
    data_path = config['data']['raw_data_path']
    data = pd.read_csv(data_path)

    # Replace '?' with NaN
    data.replace('?', np.nan, inplace=True)

    # Handle missing values (example: drop rows with missing values)
    data.dropna(inplace=True)

    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # The last column

    # Convert data types if necessary (e.g., to float)
    X = X.astype(float)

    # Normalize/scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Save processed data
    joblib.dump((X_train, X_test, y_train, y_test), "data/processed/splits.joblib")
    joblib.dump(scaler, "models/preprocessor.joblib")  # Save the scaler for later use
