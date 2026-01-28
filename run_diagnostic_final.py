import numpy as np
import joblib
import torch
import torch.nn as nn
import warnings
from sklearn.exceptions import ConvergenceWarning
import warnings

# --- 0. Suppress Warnings ---
# These are harmless warnings from scikit-learn and we can ignore them
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

# --- 1. Define the DQN "Brain" Architecture ---
# We MUST define the *exact same* class here so PyTorch knows
# how to read the saved model file.
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# --- 2. Define Features and Labels (Constants) ---
INITIAL_FEATURES = [
    'age', 'gender (0=F, 1=M)', 'smoking (0=No, 1=Yes)', 
    'chest_pain (0=No, 1=Yes)', 'dyspnea (0=No, 1=Yes)', 'fatigue (0=No, 1=Yes)',
    'systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi', 'temperature'
]
TEST_FEATURES = [
    'troponin', 'cholesterol_total', 'HDL', 'LDL', 'BNP',
    'NT_proBNP', 'echo_ef', 'stress_test_result (0=Neg, 1=Pos)', 'c_reactive_protein'
]
LABELS = ['Healthy', 'CAD', 'HF', 'STR']
ENCODING_DIM = 4 # Must match our SAE

# --- 3. Set Device and Load All Models ---
print("--- Adaptive Diagnostic Model (PyTorch) ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading all trained models (using {device})...")

try:
    # Load the scikit-learn models (SAE)
    scaler = joblib.load('sae_scaler.joblib')
    autoencoder = joblib.load('sae_autoencoder.joblib')
    
    # Load the PyTorch agent (MAB)
    n_actions = len(TEST_FEATURES) + len(LABELS)
    n_observations = ENCODING_DIM + len(TEST_FEATURES)
    
    agent = DQN(n_observations, n_actions).to(device)
    # Load the trained "brain" we just made
    agent.load_state_dict(torch.load('mab_agent_pytorch.pth', map_location=device))
    agent.eval() # Set agent to evaluation mode (no learning)
    
except FileNotFoundError as e:
    print(f"Error: Could not find model file: {e.filename}")
    print("Please make sure 'sae_scaler.joblib', 'sae_autoencoder.joblib', and 'mab_agent_pytorch.pth' are present.")
    print("You must run 'train_mab_agent_pytorch.py' first.")
    exit()

print("All models loaded successfully.")

# --- 4. Helper Function to Get Patient Fingerprint ---
def get_patient_fingerprint(patient_data, scaler, autoencoder):
    """Uses our saved SAE to get the 4-feature 'fingerprint'."""
    try:
        scaled_features = scaler.transform(patient_data.reshape(1, -1))
        fingerprint = np.dot(scaled_features, autoencoder.coefs_[0]) + autoencoder.intercepts_[0]
        fingerprint[fingerprint < 0] = 0  # Apply ReLU
        return fingerprint[0]
    except Exception as e:
        print(f"Error processing patient data: {e}")
        return None

# --- 5. Main Application Loop ---
def start_new_diagnosis():
    print("\n--- Starting New Patient Diagnosis ---")
    print("Please enter the patient's initial 11 features:")
    
    patient_initial_data = []
    for feature_name in INITIAL_FEATURES:
        while True:
            try:
                val = float(input(f"  Enter {feature_name}: "))
                patient_initial_data.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    patient_initial_data = np.array(patient_initial_data)
    
    fingerprint = get_patient_fingerprint(patient_initial_data, scaler, autoencoder)
    if fingerprint is None:
        return 

    test_results = np.full(len(TEST_FEATURES), -1.0)
    current_state = np.concatenate([fingerprint, test_results]) # 1D numpy array
    
    print("\n--- Starting Adaptive Pathway ---")
    print(f"Patient Fingerprint (SAE Output): {fingerprint}")
    
    done = False
    pathway = []
    
    while not done:
        # 1. Convert state to PyTorch tensor
        state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 2. Ask the MAB Agent for its "opinions" (Q-values)
        with torch.no_grad():
            q_values = agent(state_tensor).cpu().numpy()[0]
        
        # 3. Mask out (set to -inf) any tests that are ALREADY DONE.
        for i in range(len(TEST_FEATURES)):
            if current_state[ENCODING_DIM + i] != -1:
                q_values[i] = -np.inf  
        
        # 4. Now, find the best *available* action
        action = np.argmax(q_values)
        
        # 5. Interpret the action
        
        # --- Case 1: Action is a TEST (0-8) ---
        if 0 <= action < len(TEST_FEATURES):
            test_name = TEST_FEATURES[action]
            pathway.append(f"Test: {test_name}")
            print(f"\n[AGENT DECISION]: Run test '{test_name}'")
            
            while True:
                try:
                    result = float(input(f"  Please enter the result for {test_name}: "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            current_state[ENCODING_DIM + action] = result # Update the 1D state
            print(f"  (State updated with {test_name} = {result})")

        # --- Case 2: Action is a DIAGNOSIS (9-12) ---
        else:
            diagnosis_index = action - len(TEST_FEATURES)
            diagnosis = LABELS[diagnosis_index]
            
            pathway.append(f"Diagnose: {diagnosis}")
            print("\n========================================")
            print(f"[AGENT DECISION]: FINAL DIAGNOSIS")
            print(f"DIAGNOSIS: {diagnosis}")
            print("========================================")
            print(f"Pathway Taken: {' -> '.join(pathway)}")
            done = True

# --- Run the application ---
while True:
    start_new_diagnosis()
    
    if input("\nRun another diagnosis? (y/n): ").lower() != 'y':
        break
        
print("Exiting diagnostic tool.")
