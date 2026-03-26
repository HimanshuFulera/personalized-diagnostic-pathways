from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
import torch
import torch.nn as nn
import warnings
from sklearn.exceptions import ConvergenceWarning
import logging
import os

# --- 0. Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- BASE DIR (IMPORTANT FIX) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Define Model ---
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

# --- 2. Constants ---
INITIAL_FEATURES = [
    'age', 'gender', 'smoking', 'chest_pain', 'dyspnea', 'fatigue',
    'systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi', 'temperature'
]

TEST_FEATURES = [
    'troponin', 'cholesterol_total', 'HDL', 'LDL', 'BNP',
    'NT_pro_BNP', 'echo_ef', 'stress_test_result', 'c_reactive_protein'
]

LABELS = ['Healthy', 'CAD', 'HF', 'STR']
ENCODING_DIM = 4
TOTAL_TESTS = len(TEST_FEATURES)

# --- 3. Load Models ---
print("Loading models...")

device = torch.device("cpu")  # safer for Render

try:
    scaler = joblib.load(os.path.join(BASE_DIR, 'sae_scaler.joblib'))
    autoencoder = joblib.load(os.path.join(BASE_DIR, 'sae_autoencoder.joblib'))

    n_actions = len(TEST_FEATURES) + len(LABELS)
    n_observations = ENCODING_DIM + len(TEST_FEATURES)

    agent = DQN(n_observations, n_actions).to(device)
    agent.load_state_dict(
        torch.load(os.path.join(BASE_DIR, 'mab_agent_pytorch.pth'), map_location=device)
    )
    agent.eval()

except FileNotFoundError as e:
    raise RuntimeError(f"Model file missing: {e.filename}")

print("Models loaded successfully.")

# --- 4. Helper ---
def get_patient_fingerprint(patient_data):
    scaled = scaler.transform(patient_data.reshape(1, -1))
    fingerprint = np.dot(scaled, autoencoder.coefs_[0]) + autoencoder.intercepts_[0]
    fingerprint[fingerprint < 0] = 0
    return fingerprint[0]

def get_agent_decision(state_np):
    state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = agent(state_tensor).numpy()[0]

    diagnosis_q = q_values[TOTAL_TESTS:]
    exp_q = np.exp(diagnosis_q - np.max(diagnosis_q))
    probs = exp_q / np.sum(exp_q)

    best_idx = np.argmax(probs)

    # mask done tests
    for i in range(TOTAL_TESTS):
        if state_np[ENCODING_DIM + i] != -1:
            q_values[i] = -np.inf

    action_idx = np.argmax(q_values)

    response = {
        "current_prediction": LABELS[best_idx],
        "current_confidence": float(probs[best_idx])
    }

    if action_idx < TOTAL_TESTS:
        response.update({
            "status": "needs_test",
            "next_test": TEST_FEATURES[action_idx]
        })
    else:
        response.update({
            "status": "diagnosis_complete",
            "diagnosis": LABELS[action_idx - TOTAL_TESTS]
        })

    return response

# --- 5. Flask ---
app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/start', methods=['POST'])
def start():
    data = request.json

    initial = np.array(data['initial_data'], dtype=float)
    fingerprint = get_patient_fingerprint(initial)

    test_results = np.full(TOTAL_TESTS, -1.0)
    state = np.concatenate([fingerprint, test_results])

    res = get_agent_decision(state)
    res['current_state'] = state.tolist()
    res['fingerprint'] = fingerprint.tolist()

    return jsonify(res)

@app.route('/next_step', methods=['POST'])
def next_step():
    data = request.json

    state = np.array(data['current_state'], dtype=float)
    test_name = data['test_name']
    test_value = data['test_value']

    idx = TEST_FEATURES.index(test_name)
    state[ENCODING_DIM + idx] = test_value

    res = get_agent_decision(state)
    res['current_state'] = state.tolist()

    return jsonify(res)

# --- 6. HTML (keep your existing one) ---
HTML_TEMPLATE = """PASTE YOUR SAME HTML HERE"""

# --- 7. Run (FIXED) ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
