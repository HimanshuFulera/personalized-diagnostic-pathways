#SAE PATIENT ANALYZER


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import joblib  # This is how we save scikit-learn models

# --- Configuration ---
CSV_FILE_PATH = 'heart_diagnostic_10k_balanced_clean.csv'
INITIAL_FEATURES = [
    'age', 'gender', 'smoking', 'chest_pain', 'dyspnea', 'fatigue',
    'systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi', 'temperature'
]
ENCODING_DIM = 4
print("--- Scikit-learn Autoencoder Builder ---")

# --- 1. Load Data ---
print(f"Loading data from '{CSV_FILE_PATH}'...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: Could not find the file '{CSV_FILE_PATH}'.")
    print("Please make sure the CSV file is in the same directory as this script.")
    exit()

df_features = df[INITIAL_FEATURES].astype(float)

# --- 2. Preprocess Data ---
print("Scaling data...")
# We must scale the data for neural networks
scaler = MinMaxScaler()
x_train = scaler.fit_transform(df_features)
print(f"Data loaded. Shape of training data: {x_train.shape}")

# --- 3. Build and Train the Autoencoder ---
print(f"Building Autoencoder (Input: {x_train.shape[1]} -> Encoded: {ENCODING_DIM})...")

# We use a Multi-Layer Perceptron (MLP) Regressor as an autoencoder.
# We tell it to have one hidden layer with ENCODING_DIM neurons.
# The 'alpha' parameter provides the L1 sparsity regularization.
autoencoder = MLPRegressor(
    hidden_layer_sizes=(ENCODING_DIM,),
    activation='relu',
    solver='adam',
    alpha=10e-5,  # This is the L1 regularization (Sparsity)
    max_iter=500, # scikit-learn often needs more iterations
    shuffle=True,
    random_state=1,
    verbose=True  # This will show you the training progress
)

print("Training autoencoder (using Scikit-learn)...")
# We train the model to predict its own input
autoencoder.fit(x_train, x_train)

# --- 4. Save the Model and Scaler ---
# We save the *full* autoencoder and the *scaler*
# The MAB (Part 2) will need both.
# The .joblib file is the scikit-learn equivalent of .h5
print("\n--- Training Complete! ---")
joblib.dump(autoencoder, 'sae_autoencoder.joblib')
joblib.dump(scaler, 'sae_scaler.joblib')

print("The 'Patient Analyzer' (SAE) has been saved as 'sae_autoencoder.joblib'")
print("The 'Scaler' has been saved as 'sae_scaler.joblib'")
print("These models are now ready to be used by the MAB (Part 2).")

# --- 5. Example of Use (How to get the fingerprint) ---
print("\nTesting the encoder with the first patient:")
# Get the first patient and scale them
first_patient = scaler.transform(df_features.iloc[0:1])
print(f"Original 11 features (scaled):\n{first_patient[0]}")

# To get the "fingerprint" from a scikit-learn MLP,
# you apply the first layer's weights and biases manually.
patient_fingerprint = np.dot(first_patient, autoencoder.coefs_[0]) + autoencoder.intercepts_[0]
# Apply the 'relu' activation
patient_fingerprint[patient_fingerprint < 0] = 0

print(f"\nLearned 4-feature 'fingerprint':\n{patient_fingerprint[0]}")
