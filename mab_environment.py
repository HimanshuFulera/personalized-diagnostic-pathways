#"Rulebook" for the training "game."

import numpy as np
import pandas as pd
import joblib

class DiagnosticEnvironment:
    """
    This is the "World Simulator" or "Playground" for our MAB agent.
    It uses the CSV data as the ground truth.
    The agent will "play" in this environment to learn the optimal pathway.
    """
    
    def __init__(self, csv_path, scaler_path, autoencoder_path):
        print(f"Loading environment from {csv_path}...")
        
        # --- 1. Load Ground Truth Data ---
        self.df = pd.read_csv(csv_path)
        
        # --- 2. Load our Trained SAE (Part 1) ---
        self.scaler = joblib.load(scaler_path)
        self.autoencoder = joblib.load(autoencoder_path)
        
        # --- THIS IS THE FIX ---
        # The autoencoder's *hidden layer* (our fingerprint) is 4 features.
        # The old, buggy code was using self.autoencoder.n_outputs_ (which was 11).
        self.encoding_dim = 4 
        
        # --- 3. Define Features, Tests, and Labels ---
        self.initial_features = [
            'age', 'gender', 'smoking', 'chest_pain', 'dyspnea', 'fatigue',
            'systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi', 'temperature'
        ]
        
        # These are the "arms" of the bandit
        self.test_features = [
            'troponin', 'cholesterol_total', 'HDL', 'LDL', 'BNP',
            'NT_proBNP', 'echo_ef', 'stress_test_result', 'c_reactive_protein'
        ]
        
        # These are the possible final diagnoses
        self.labels = ['Healthy', 'CAD', 'HF', 'STR']
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        
        # --- 4. Define State and Action Space ---
        # The MAB's "state" is the (4-feature fingerprint + 9 test results)
        self.state_size = self.encoding_dim + len(self.test_features) # Now 4 + 9 = 13
        
        # The MAB's "actions" are (9 tests + 4 diagnoses)
        self.action_size = len(self.test_features) + len(self.labels) # 9 + 4 = 13
        
        self.current_patient_index = 0
        self.current_state = None
        self.current_patient_data = None
        self.current_patient_label = None

    def _get_patient_fingerprint(self, patient_data):
        """Uses our saved SAE to get the 4-feature 'fingerprint'."""
        # Scale the 11 initial features
        scaled_features = self.scaler.transform(patient_data[self.initial_features].values.reshape(1, -1))
        
        # Get the "fingerprint" from the autoencoder's hidden layer
        fingerprint = np.dot(scaled_features, self.autoencoder.coefs_[0]) + self.autoencoder.intercepts_[0]
        fingerprint[fingerprint < 0] = 0  # Apply ReLU
        return fingerprint[0]

    def reset(self):
        """
        Resets the environment to a new random patient.
        Returns the initial state for the MAB.
        """
        # Get a random patient
        self.current_patient_index = np.random.randint(0, len(self.df))
        self.current_patient_data = self.df.iloc[self.current_patient_index]
        self.current_patient_label = self.label_map[self.current_patient_data['label']]
        
        # Get the 4-feature "fingerprint" from our SAE
        fingerprint = self._get_patient_fingerprint(self.current_patient_data)
        
        # Initialize the state: (fingerprint + 9 "unknown" tests)
        # We use -1 to represent an "unknown" or "not yet performed" test
        test_results = np.full(len(self.test_features), -1.0)
        
        self.current_state = np.concatenate([fingerprint, test_results])
        return self.current_state

    def step(self, action):
        """
        The MAB takes an 'action'. The environment responds.
        Returns: (next_state, reward, done)
        """
        # --- Define Rewards ---
        COST_PER_TEST = -1  # Small penalty for each test
        REWARD_CORRECT_DIAGNOSIS = +50
        PENALTY_WRONG_DIAGNOSIS = -50
        PENALTY_ALREADY_TESTED = -10 # Big penalty for re-testing
        
        # The episode is over
        done = False
        reward = 0
        
        # ---- Case 1: Action is a TEST (actions 0-8) ----
        if 0 <= action < len(self.test_features):
            test_name = self.test_features[action]
            
            # Check if we already ran this test
            if self.current_state[self.encoding_dim + action] != -1:
                reward = PENALTY_ALREADY_TESTED
                # State doesn't change, but we give a penalty
            else:
                # "Perform" the test by looking up the real value
                test_result = self.current_patient_data[test_name]
                
                # Update the state vector
                self.current_state[self.encoding_dim + action] = test_result
                reward = COST_PER_TEST
            
            done = False
        
        # ---- Case 2: Action is a DIAGNOSIS (actions 9-12) ----
        elif action >= len(self.test_features):
            diagnosis_index = action - len(self.test_features)
            
            # Check if the MAB's diagnosis matches the patient's true label
            if diagnosis_index == self.current_patient_label:
                reward = REWARD_CORRECT_DIAGNOSIS
            else:
                reward = PENALTY_WRONG_DIAGNOSIS
            
            done = True # The episode ends on a diagnosis
            
        return self.current_state.copy(), reward, done

if __name__ == '__main__':
    # This just tests the environment to make sure it works.
    print("--- Testing MAB Environment ---")
    
    env = DiagnosticEnvironment(
        csv_path='heart_diagnostic_10k_balanced_clean.csv',
        scaler_path='sae_scaler.joblib',
        autoencoder_path='sae_autoencoder.joblib'
    )
    
    state = env.reset()
    print(f"Initial State (SAE fingerprint + -1s): {state}")
    print(f"State vector size: {len(state)}")
    
    # Test 1: Perform "troponin" test (action 0)
    print("\nTaking action 0 (test 'troponin')...")
    next_state, reward, done = env.step(0)
    print(f"New State: {next_state}")
    print(f"Reward: {reward}, Done: {done}")

    # Test 2: Perform "cholesterol" test (action 1)
    print("\nTaking action 1 (test 'cholesterol_total')...")
    next_state, reward, done = env.step(1)
    print(f"New State: {next_state}")
    print(f"Reward: {reward}, Done: {done}")
    
    # Test 3: Make a (wrong) diagnosis of "Healthy" (action 9)
    print(f"\nTaking action 9 (diagnose 'Healthy')...")
    final_state, reward, done = env.step(9)
    print(f"Final State: {final_state}")
    print(f"Reward: {reward}, Done: {done}")
    print(f"(True label was: {env.labels[env.current_patient_label]})")
    
    print("\nEnvironment test complete. Ready for MAB agent.")