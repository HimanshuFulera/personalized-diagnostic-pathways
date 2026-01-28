import joblib
import pandas as pd
import numpy as np
import warnings
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
# from sklearn.exceptions import UserWarning <-- This line is removed
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
print("--- t-SNE Latent Space Visualizer ---")

# --- 1. Define Features ---
INITIAL_FEATURES = [
    'age', 'gender', 'smoking', 'chest_pain', 'dyspnea', 'fatigue',
    'systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi', 'temperature'
]
LABEL_COLUMN = 'label'

# --- 2. Load Data and Models ---
print("Loading data and trained SAE models...")
try:
    df = pd.read_csv('heart_diagnostic_10k_balanced_clean.csv')
    scaler = joblib.load('sae_scaler.joblib')
    autoencoder = joblib.load('sae_autoencoder.joblib')
except FileNotFoundError as e:
    print(f"Error: Could not find a required file: {e.filename}")
    print("Make sure all .csv and .joblib files are in this directory.")
    exit()

print("All files loaded successfully.")

# --- 3. Generate Fingerprints for ALL Patients ---
print(f"Generating 4D fingerprints for all {len(df)} patients...")

# Get the 11 initial features
patient_features = df[INITIAL_FEATURES].astype(float)
# Get the true disease labels
patient_labels = df[LABEL_COLUMN]

# Step 3a: Scale the features
scaled_features = scaler.transform(patient_features)

# Step 3b: Get the 4D fingerprints from the SAE "bottleneck"
# This is the "Encoder" part of our model
latent_space_4d = np.dot(scaled_features, autoencoder.coefs_[0]) + autoencoder.intercepts_[0]
# Apply the ReLU activation (same as in our main app)
latent_space_4d[latent_space_4d < 0] = 0

print(f"Generated {latent_space_4d.shape[0]} fingerprints, each with {latent_space_4d.shape[1]} features.")

# --- 4. Run t-SNE to "Squash" 4D down to 2D ---
print("Running t-SNE to compress 4D fingerprints into 2D plot coordinates...")
# n_components=2 means we want a 2D (x, y) plot
# perplexity is a key parameter; 30-50 is a good default
#
# *** THIS IS THE FIX: 'n_iter' is renamed to 'max_iter' ***
#
tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42, verbose=1)
latent_space_2d = tsne.fit_transform(latent_space_4d)

print("t-SNE complete.")

# --- 5. Create and Save the Plot ---
print("Generating plot with Matplotlib and Seaborn...")

# Create a new DataFrame for plotting
plot_df = pd.DataFrame(data=latent_space_2d, columns=['t-SNE Component 1', 't-SNE Component 2'])
plot_df['Diagnosis'] = patient_labels

# Set the style
sns.set(style="whitegrid", palette="muted")
sns.set_context("notebook", font_scale=1.2)

# Define the colors for our 4 diseases
# This ensures "Healthy" is green, "CAD" is red, etc.
palette = {
    "Healthy": "#10b981", # Green
    "CAD": "#f87171",     # Red
    "HF": "#facc15",      # Yellow
    "STR": "#818cf8"      # Indigo/Purple
}

# Create the plot
plt.figure(figsize=(14, 10))
ax = sns.scatterplot(
    x="t-SNE Component 1", y="t-SNE Component 2",
    hue="Diagnosis",
    palette=palette,
    data=plot_df,
    legend="full",
    alpha=0.7, # Make dots slightly transparent
    s=20       # Set dot size
)

# Set title and labels
plt.title('t-SNE Visualization of SAE Latent Space (Patient Fingerprints)', fontsize=20, weight='bold', pad=20)
plt.xlabel('t-SNE Component 1', fontsize=14)
plt.ylabel('t-SNE Component 2', fontsize=14)
ax.legend(loc='upper right', title='Diagnosis', title_fontsize='14', fontsize='12', frameon=True, shadow=True)

# Save the figure
output_filename = 'sae_latent_space_tsne.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"\n--- SUCCESS! ---")
print(f"Plot has been saved as: {output_filename}")