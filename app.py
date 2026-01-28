import flask
from flask import Flask, render_template_string, request, jsonify
import numpy as np
import joblib
import torch
import torch.nn as nn
import warnings
from sklearn.exceptions import ConvergenceWarning
import logging

# --- 0. Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- 1. Define the DQN "Brain" Architecture ---
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
# This order is CRITICAL and must be matched by the JavaScript
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

# --- 3. Set Device and Load All Models ---
print("--- Adaptive Diagnostic Model (PyTorch) ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading all trained models (using {device})...")

try:
    scaler = joblib.load('sae_scaler.joblib')
    autoencoder = joblib.load('sae_autoencoder.joblib')
    
    n_actions = len(TEST_FEATURES) + len(LABELS)
    n_observations = ENCODING_DIM + len(TEST_FEATURES)
    
    agent = DQN(n_observations, n_actions).to(device)
    agent.load_state_dict(torch.load('mab_agent_pytorch.pth', map_location=device))
    agent.eval()
    
except FileNotFoundError as e:
    print(f"[FATAL ERROR] Could not find model file: {e.filename}")
    print("Please make sure all .joblib and .pth files are present.")
    print("You may need to run 'build_autoencoder.py' and 'train_mab_agent_pytorch.py' first.")
    exit()

print("All models loaded successfully. Starting web server...")

# --- 4. Helper Function to Get Patient Fingerprint ---
def get_patient_fingerprint(patient_data, scaler, autoencoder):
    try:
        scaled_features = scaler.transform(patient_data.reshape(1, -1))
        fingerprint = np.dot(scaled_features, autoencoder.coefs_[0]) + autoencoder.intercepts_[0]
        fingerprint[fingerprint < 0] = 0  # Apply ReLU
        return fingerprint[0]
    except Exception as e:
        print(f"Error processing patient data: {e}")
        return None

# --- 5. The Main Agent Logic Function ---
def get_agent_decision(current_state_np):
    """
    Receives the current state and asks the agent
    for its next best action *and* its current confidence.
    """
    # 1. Convert state to PyTorch tensor
    state_tensor = torch.tensor(current_state_np, dtype=torch.float32, device=device).unsqueeze(0)
    
    # 2. Ask the MAB Agent for its "opinions" (Q-values)
    with torch.no_grad():
        q_values = agent(state_tensor).cpu().numpy()[0]
    
    # 3. Find current best diagnosis and confidence
    diagnosis_q_values = q_values[TOTAL_TESTS:]
    # Use softmax for stable probabilities
    exp_q = np.exp(diagnosis_q_values - np.max(diagnosis_q_values)) 
    diagnosis_probs = exp_q / np.sum(exp_q)
    
    best_diagnosis_index = np.argmax(diagnosis_probs)
    current_prediction = LABELS[best_diagnosis_index]
    current_confidence = diagnosis_probs[best_diagnosis_index]

    # 4. Mask out (set to -inf) any tests that are ALREADY DONE.
    for i in range(TOTAL_TESTS):
        if current_state_np[ENCODING_DIM + i] != -1:
            q_values[i] = -np.inf  
    
    # 5. Now, find the best *available* action
    action_index = np.argmax(q_values)
    
    # 6. Interpret and return the action
    response = {
        "current_prediction": current_prediction,
        "current_confidence": float(current_confidence)
    }

    if 0 <= action_index < TOTAL_TESTS:
        # Action is a test
        response.update({
            "status": "needs_test",
            "next_test": TEST_FEATURES[action_index]
        })
    else:
        # Action is a diagnosis
        diagnosis_index = action_index - TOTAL_TESTS
        diagnosis = LABELS[diagnosis_index]
        response.update({
            "status": "diagnosis_complete",
            "diagnosis": diagnosis
        })
    return response

# --- 6. The Flask API (Backend) ---
app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/start', methods=['POST'])
def start_diagnosis():
    data = request.json
    
    # The JS side now prepares the 'initial_data' array in the correct order
    initial_data = np.array(data['initial_data'], dtype=float)
    
    fingerprint = get_patient_fingerprint(initial_data, scaler, autoencoder)
    if fingerprint is None:
        return jsonify({"error": "Error processing patient data"}), 500
        
    test_results = np.full(TOTAL_TESTS, -1.0)
    current_state = np.concatenate([fingerprint, test_results])
    
    action_response = get_agent_decision(current_state)
    
    action_response['current_state'] = current_state.tolist()
    action_response['fingerprint'] = fingerprint.tolist()
    action_response['initial_data_map'] = data['initial_data_map'] # Pass map for logging
    return jsonify(action_response)

@app.route('/next_step', methods=['POST'])
def next_step():
    data = request.json
    current_state = np.array(data['current_state'], dtype=float)
    
    test_name = data['test_name']
    test_value = data['test_value']
    
    try:
        test_index = TEST_FEATURES.index(test_name)
    except ValueError:
        # Handle the 'NT_proBNP' vs 'NT_pro_BNP' mismatch
        if test_name == 'NT_proBNP' and 'NT_pro_BNP' in TEST_FEATURES:
             test_index = TEST_FEATURES.index('NT_pro_BNP')
        elif test_name == 'NT_pro_BNP' and 'NT_proBNP' in TEST_FEATURES:
             test_index = TEST_FEATURES.index('NT_proBNP')
        else:
            print(f"Error: Test name '{test_name}' not in TEST_FEATURES list.")
            return jsonify({"error": "Invalid test name"}), 400

    current_state[ENCODING_DIM + test_index] = test_value
    
    action_response = get_agent_decision(current_state)
    
    action_response['current_state'] = current_state.tolist()
    return jsonify(action_response)

# --- 7. The HTML/CSS/JS (Frontend) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en"> <!-- Theme set by JS -->
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Adaptive Diagnostic Model</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <!-- START: CSS (Merged from your files) -->
  <style>
    :root { 
      --bg: #fafafa; 
      --card: #ffffff;
      --card-alt: #f8faff;
      --text: #111827; 
      --muted: #6b7280; 
      --primary: #1d4ed8; 
      --primary-hsl: 221, 83%, 53%;
      --success: #059669; 
      --danger: #dc2626;
      --warning: #f59e0b;
      --info: #6366f1;
      --border: #e5e7eb; 
      --shadow: 0 4px 20px rgba(0,0,0,0.06); 
      --radius: 14px; 
    }
    [data-theme="dark"] { 
      --bg: #162036; 
      --card: #1e293b;
      --card-alt: #131c2c;
      --text: #f8fafc; 
      --muted: #94a3b8; 
      --primary: #3b82f6; 
      --primary-hsl: 221, 91%, 61%;
      --success: #10b981;
      --danger: #f87171;
      --warning: #facc15;
      --info: #818cf8;
      --border: #334155; 
      --shadow: 0 4px 20px rgba(0,0,0,0.3); 
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: 'Inter', sans-serif; 
      color: var(--text); 
      background-color: var(--bg);
      line-height: 1.7; 
      min-height: 100vh; 
      padding: 32px 16px; 
      transition: background-color 0.3s ease, color 0.3s ease; 
    }
    .container { 
      max-width: 960px; /* Widened layout */
      margin: 0 auto; 
    }
    .card { 
      background-color: var(--card); 
      border-radius: var(--radius); 
      box-shadow: var(--shadow); 
      border: 1px solid var(--border); 
      padding: 32px 40px; 
      transition: background-color 0.3s ease, border-color 0.3s ease; 
    }
    
    /* --- HEADER --- */
    .header-container {
      position: relative;
      padding: 24px 0; 
      margin-bottom: 32px; 
    }
    .header-title {
      text-align: center;
    }
    .header-title h1 {
      font-size: 32px; /* Increased */
      font-weight: 700;
      color: var(--text);
    }
    .header-title p {
      font-size: 20px; /* Increased */
      font-weight: 500;
      color: var(--muted);
      margin-top: 8px;
    }
    .theme-toggle { 
      background: var(--card-alt); 
      border: 1px solid var(--border); 
      border-radius: 99px; 
      padding: 12px;
      cursor: pointer; 
      display: flex;
      font-size: 24px;
      line-height: 1;
      transition: all 0.3s ease;
      position: absolute;
      right: 16px;
      top: 50%; /* Perfect vertical align */
      transform: translateY(-50%); /* Perfect vertical align */
    }
    .theme-toggle:hover {
      border-color: var(--primary);
    }
    
    h2 { 
      font-size: 18px; 
      font-weight: 600; 
      color: var(--muted);
      margin-top: 24px;
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }
    h2:first-of-type { margin-top: 0; }
    h3 {
      font-size: 15px; 
      color: var(--muted); 
      margin: 28px 0 14px; 
      font-weight: 600;
    }
    .grid { 
      display: grid; 
      grid-template-columns: repeat(4, 1fr); 
      gap: 20px; 
    }
    .grid-cols-2 { grid-template-columns: repeat(2, 1fr); }
    .grid-cols-3 { grid-template-columns: repeat(3, 1fr); }

    .form-group { 
      display: flex; 
      flex-direction: column; 
    }
    label { 
      font-size: 14px; 
      font-weight: 500; 
      color: var(--muted); 
      margin-bottom: 6px; 
    }
    input[type="number"], input[type="text"], select { 
      font-family: 'Inter', sans-serif;
      font-size: 15px;
      background: var(--bg); 
      border: 1px solid var(--border); 
      border-radius: 8px; 
      padding: 12px 14px; 
      color: var(--text); 
      transition: all 0.3s ease;
      width: 100%;
      -moz-appearance: textfield; /* Firefox */
    }
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }
    select {
      appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3E%3Cpath stroke='%236B7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='m6 8 4 4 4-4'/%3E%3C/svg%3E");
      background-position: right 0.75rem center;
      background-repeat: no-repeat;
      background-size: 1.25em;
      padding-right: 2.5rem;
    }
    [data-theme="dark"] select {
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3E%3Cpath stroke='%2394a3b8' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='m6 8 4 4 4-4'/%3E%3C/svg%3E");
    }

    input[type="number"]:focus, input[type="text"]:focus, select:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px hsla(var(--primary-hsl), 0.2);
    }
    
    /* --- NEW CHECKBOX STYLES (from start.html) --- */
    .checkbox-group {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 20px;
      margin-top: 6px;
    }
    .checkbox-label {
      display: flex;
      flex-direction: row;
      align-items: center;
      gap: 10px;
      cursor: pointer;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 14px;
      font-weight: 600;
      color: var(--muted);
      transition: all 0.3s ease;
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: var(--primary);
    }
    input[type="checkbox"]:checked + span {
      color: var(--text);
    }
    .checkbox-label:hover {
      border-color: var(--primary);
    }
    /* --- END CHECKBOX STYLES --- */
    
    button { 
      font-family: 'Inter', sans-serif;
      font-size: 15px;
      font-weight: 600;
      background-color: var(--primary); 
      color: #fff; 
      border: none; 
      border-radius: 8px; 
      padding: 12px 24px; 
      cursor: pointer; 
      transition: all 0.3s ease; 
      display: inline-flex;
      justify-content: center;
      align-items: center;
      width: 100%;
    }
    button.theme-toggle { /* Override for theme button */
        width: auto;
        padding: 12px; /* Must match parent */
    }
    button:hover { 
      opacity: 0.85; 
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    button.secondary {
      background-color: var(--card-alt);
      color: var(--text);
      border: 1px solid var(--border);
    }
    button.secondary:hover {
      background-color: var(--bg);
      opacity: 1;
    }
    .spinner {
        border: 2px solid rgba(255,255,255,0.3);
        border-top-color: #ffffff;
        animation: spin 1s linear infinite;
        border-radius: 50%;
        width: 18px;
        height: 18px;
        margin-left: 10px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .hidden { display: none; }
    .pathway-log {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      max-height: 400px;
      overflow-y: auto;
    }
    .log-item {
      display: flex;
      align-items: flex-start;
      gap: 12px;
      padding: 12px 0;
    }
    .log-item + .log-item {
      border-top: 1px solid var(--border);
    }
    .log-icon { flex-shrink: 0; width: 32px; height: 32px; border-radius: 50%; background: var(--card-alt); display: flex; align-items: center; justify-content: center; border: 1px solid var(--border); }
    .log-content { flex-grow: 1; padding-top: 4px; }
    .log-title { font-weight: 600; color: var(--text); }
    
    .diagnosis-card {
      border-radius: var(--radius);
      padding: 32px;
      text-align: center;
      margin-bottom: 24px;
    }
    .diagnosis-card.Healthy { background-color: var(--success); color: #fff; }
    .diagnosis-card.CAD { background-color: var(--danger); color: #fff; }
    .diagnosis-card.HF { background-color: var(--warning); color: var(--text); }
    .diagnosis-card.STR { background-color: var(--info); color: #fff; }
    
    .diagnosis-label { font-size: 16px; font-weight: 600; opacity: 0.8; }
    .diagnosis-value { font-size: 48px; font-weight: 700; line-height: 1.2; }
    .confidence-value { font-size: 20px; font-weight: 600; margin-top: 8px; opacity: 0.9; }

    .stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-top: 24px; }
    .stat-card { background: var(--bg); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid var(--border); }
    .stat-value { font-size: 28px; font-weight: 700; color: var(--primary); }
    .stat-label { font-size: 14px; color: var(--muted); font-weight: 500; }
    .stat-value.success { color: var(--success); }

    .error-box { padding: 16px; background: var(--danger); color: #fff; border-radius: 8px; margin-top: 20px; text-align: center; }
  </style>
  <!-- END: CSS -->
</head>
<body>

  <div class="container">
    
    <!-- Header - FIXED -->
    <div class="header-container">
      <div class="header-title">
        <h1>Adaptive Diagnostic Test Recommendation Model</h1>
        <p>Cardiac Risk Stratification</p>
      </div>
      <button class="theme-toggle" id="theme-toggle" title="Toggle theme">
        <span id="theme-icon">🌙</span>
      </button>
    </div>

    <!-- Main Card -->
    <div class="card">

        <!-- Step 1: Initial Patient Data -->
        <div id="step-1-initial-data">
            <form id="initial-data-form">
                
                <!-- Layout adapted from start.html -->
                <h2>Patient Profile</h2>
                <div class="grid grid-cols-3">
                    <div class="form-group">
                        <label for="age">Age</label>
                        <input type="number" id="age" name="age" required placeholder="58">
                    </div>
                    <div class="form-group">
                        <label for="gender">Biological Sex</label>
                        <select id="gender" name="gender" required>
                            <option value="">Select</option>
                            <option value="1">Male</option>
                            <option value="0">Female</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="bmi">BMI</label>
                        <input type="number" step="any" id="bmi" name="bmi" required placeholder="28.4">
                    </div>
                </div>

                <h2>Vitals & Clinical Indicators</h2>
                <div class="grid grid-cols-2">
                    <!-- Vitals -->
                    <div class="form-group"><label for="systolic_bp">Systolic BP</label><input type="number" step="any" id="systolic_bp" name="systolic_bp" required placeholder="148"></div>
                    <div class="form-group"><label for="diastolic_bp">Diastolic BP</label><input type="number" step="any" id="diastolic_bp" name="diastolic_bp" required placeholder="92"></div>
                    <div class="form-group"><label for="heart_rate">Heart Rate</label><input type="number" step="any" id="heart_rate" name="heart_rate" required placeholder="92"></div>
                    <div class="form-group"><label for="temperature">Temp (°C)</label><input type="number" step="any" id="temperature" name="temperature" required placeholder="36.8"></div>
                </div>
                
                <h2 style="margin-top: 20px; margin-bottom: 16px;">Symptoms & History</h2>
                <div class="checkbox-group">
                    <label class="checkbox-label" for="smoking">
                        <input type="checkbox" id="smoking" name="smoking" value="1">
                        <span>Smoking</span>
                    </label>
                    <label class="checkbox-label" for="chest_pain">
                        <input type="checkbox" id="chest_pain" name="chest_pain" value="1">
                        <span>Chest Pain</span>
                    </label>
                    <label class="checkbox-label" for="dyspnea">
                        <input type="checkbox" id="dyspnea" name="dyspnea" value="1">
                        <span>Dyspnea</span>
                    </label>
                    <label class="checkbox-label" for="fatigue">
                        <input type="checkbox" id="fatigue" name="fatigue" value="1">
                        <span>Fatigue</span>
                    </label>
                </div>
                
                <button id="start-button" type="submit" style="margin-top: 24px;">
                    <span>Initiate Diagnostic Pathway</span>
                    <div class="spinner hidden"></div>
                </button>
            </form>
        </div>

        <!-- Step 2: Adaptive Pathway -->
        <div id="step-2-adaptive-pathway" class="hidden">
            <div id="confidence-display" class="stat-card" style="margin-bottom: 24px; background-color: var(--bg);">
                <div class="stat-label">Agent's Current Prediction</div>
                <div id="current-prediction-text" class="stat-value" style="font-size: 24px;">...</div>
                <div class="stat-label" style="margin-top: 8px;">Confidence: <span id="current-confidence-text" style="font-weight: 700; color: var(--text);">...</span></div>
            </div>

            <form id="next-step-form" class="form-group">
                <label id="next-step-label" for="next-step-input" style="font-weight: 600; color: var(--text);">Next test required:</label>
                <input type="number" step="any" id="next-step-input" class="mt-2" required autofocus>
                <button id="next-step-button" type="submit" style="margin-top: 16px;">
                    <span>Submit Result</span>
                    <div class="spinner hidden"></div>
                </button>
            </form>

            <h3 style="text-align: center;">Pathway Log</h3>
            <div id="live-pathway-log" class="pathway-log">
                <!-- Log entries will be generated by JS -->
            </div>
        </div>
        
        <!-- Step 3: Final Diagnosis -->
        <div id="step-3-final-diagnosis" class="hidden">
            <div id="diagnosis-card" class="diagnosis-card">
                <p class="diagnosis-label">Final Diagnosis</p>
                <p id="diagnosis-text" class="diagnosis-value">...</p>
                <p id="confidence-text" class="confidence-value">...% Confidence</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div id="tests-performed-text" class="stat-value">0</div>
                    <div class="stat-label">Tests Performed</div>
                </div>
                <div class="stat-card">
                    <div id="efficiency-gain-text" class="stat-value success">0%</div>
                    <div class="stat-label">Efficiency Gain</div>
                </div>
            </div>

            <h3>Complete Diagnostic Pathway</h3>
            <div id="final-pathway-log" class="pathway-log">
                <!-- Final log entries will be generated by JS -->
            </div>

            <button id="reset-button" class="secondary" style="margin-top: 24px;">
                <span>Diagnose New Patient</span>
            </button>
        </div>

        <!-- Error Message Box -->
        <div id="error-box" class="error-box hidden">
            <p id="error-text"></p>
        </div>

    </div> <!-- End .card -->
  </div> <!-- End .container -->

  <script>
    // --- 1. Get References to all HTML elements ---
    const step1Div = document.getElementById('step-1-initial-data');
    const step2Div = document.getElementById('step-2-adaptive-pathway');
    const step3Div = document.getElementById('step-3-final-diagnosis');
    
    const initialDataForm = document.getElementById('initial-data-form');
    const startButton = document.getElementById('start-button');
    
    const livePathwayLog = document.getElementById('live-pathway-log');
    const finalPathwayLog = document.getElementById('final-pathway-log');
    
    const nextStepForm = document.getElementById('next-step-form');
    const nextStepLabel = document.getElementById('next-step-label');
    const nextStepInput = document.getElementById('next-step-input');
    const nextStepButton = document.getElementById('next-step-button');

    const confidenceDisplay = document.getElementById('confidence-display');
    const currentPredictionText = document.getElementById('current-prediction-text');
    const currentConfidenceText = document.getElementById('current-confidence-text');

    const diagnosisCard = document.getElementById('diagnosis-card');
    const diagnosisText = document.getElementById('diagnosis-text');
    const confidenceText = document.getElementById('confidence-text');
    const testsPerformedText = document.getElementById('tests-performed-text');
    const efficiencyGainText = document.getElementById('efficiency-gain-text');
    const resetButton = document.getElementById('reset-button');
    
    const errorBox = document.getElementById('error-box');
    const errorText = document.getElementById('error-text');

    // --- 2. Define "Constants" (from Python) ---
    // CRITICAL: This order MUST match the Python list
    const INITIAL_FEATURES = [
        'age', 'gender', 'smoking', 'chest_pain', 'dyspnea', 'fatigue',
        'systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi', 'temperature'
    ];
    const TOTAL_TESTS = 9; // from TEST_FEATURES
    const ICONS = {
        USER: `<svg style="width: 24px; height: 24px; color: var(--muted);" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" /></svg>`,
        STETHOSCOPE: `<svg style="width: 24px; height: 24px; color: var(--primary);" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 9.75h16.5m-16.5 0a2.25 2.25 0 01-2.25-2.25V5.25A2.25 2.25 0 015.25 3h13.5A2.25 2.25 0 0121 5.25v2.25c0 1.242-.998 2.25-2.25 2.25m-16.5 0V21a2.25 2.25 0 002.25 2.25h1.5a2.25 2.25 0 002.25-2.25v-8.25m3 1.125V21a2.25 2.25 0 002.25 2.25h1.5a2.25 2.25 0 002.25-2.25v-8.25m3-1.125V21a2.25 2.25 0 002.25 2.25h1.5A2.25 2.25 0 0021 18.75v-8.25m-16.5 0c0-1.657 3.358-3 7.5-3s7.5 1.343 7.5 3m-7.5 0V5.25m0 3.75v.008m0 0A.75.75 0 0112 9.75h.008a.75.75 0 01.75.75v.008a.75.75 0 01-.75.75h-.008A.75.75 0 0112 10.5v-.008z" /></svg>`,
        BEAKER: `<svg style="width: 24px; height: 24px; color: var(--info);" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.5 1.58-2.25 2.25 0 00-.5 1.58V18.75m0 0A2.25 2.25 0 0012 21a2.25 2.25 0 002.25-2.25v-6.772a2.25 2.25 0 00-.5-1.582 2.25 2.25 0 01-.5-1.58V3.104m6.364 5.241A2.25 2.25 0 0116.5 10.5v2.25a2.25 2.25 0 01-2.25 2.25H12a2.25 2.25 0 01-2.25-2.25V10.5a2.25 2.25 0 012.25-2.25h.75m-6.75 0V3.104m6.364 5.241A2.25 2.25 0 0116.5 10.5v2.25a2.25 2.25 0 01-2.25 2.25H12a2.25 2.25 0 01-2.25-2.25V10.5a2.25 2.25 0 012.25-2.25h.75m-6.75 0h1.5" /></svg>`,
        CLIPBOARD: `<svg style="width: 24px; height: 24px; color: var(--success);" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>`
    };
    
    // State variables
    let state = null;
    let testCount = 0;
    
    // --- 3. Attach Event Handlers ---
    document.addEventListener('DOMContentLoaded', () => {
      const toggleButton = document.getElementById('theme-toggle');
      const icon = document.getElementById('theme-icon');
      const html = document.documentElement;

      // Set initial theme (default to dark)
      if (!localStorage.getItem('theme')) {
          localStorage.setItem('theme', 'dark');
      }
      
      if (localStorage.getItem('theme') === 'dark') {
          html.setAttribute('data-theme', 'dark');
          icon.textContent = '☀️';
      } else {
          html.removeAttribute('data-theme');
          icon.textContent = '🌙';
      }

      // Add click listener
      toggleButton.addEventListener('click', () => {
        if (html.hasAttribute('data-theme')) {
          html.removeAttribute('data-theme');
          icon.textContent = '🌙';
          localStorage.setItem('theme', 'light');
        } else {
          html.setAttribute('data-theme', 'dark');
          icon.textContent = '☀️';
          localStorage.setItem('theme', 'dark');
        }
      });

      // Attach form listeners
      initialDataForm.addEventListener('submit', onStartDiagnosis);
      nextStepForm.addEventListener('submit', onSubmitTestResult);
      resetButton.addEventListener('click', onReset);
      
      // Set placeholder checkboxes
      document.getElementById('smoking').checked = true;
      document.getElementById('chest_pain').checked = true;
      document.getElementById('dyspnea').checked = true;
      document.getElementById('fatigue').checked = true;
    });

    // --- 4. Event Handler Logic ---
    
    async function onStartDiagnosis(e) {
        e.preventDefault();
        const form = new FormData(initialDataForm);
        const initialDataMap = {};
        const initialDataArray = [];
        let isValid = true;
        
        // --- NEW: Read data in the correct order from the form ---
        // This order MUST match the Python INITIAL_FEATURES list
        try {
            const numberFields = ['age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi', 'temperature'];
            
            // 1. Get all number values
            numberFields.forEach(key => {
                const val = parseFloat(form.get(key));
                if (isNaN(val)) isValid = false;
                initialDataMap[key] = val;
            });
            
            // 2. Get Gender from select
            const genderVal = form.get('gender');
            if (genderVal === "") {
                isValid = false;
            }
            initialDataMap['gender'] = parseFloat(genderVal);
            
            // 3. Get Checkboxes
            const checkboxFields = ['smoking', 'chest_pain', 'dyspnea', 'fatigue'];
            checkboxFields.forEach(key => {
                const val = document.getElementById(key).checked ? 1 : 0;
                initialDataMap[key] = val;
            });
            
            // 4. Build the final array IN ORDER
            const orderedData = INITIAL_FEATURES.map(key => {
                // This check ensures if a key is missing in the map, we catch it
                if (initialDataMap[key] === undefined) {
                    console.error("Key mismatch:", key);
                    isValid = false;
                }
                return initialDataMap[key];
            });
            
            if (!isValid) {
                showError("Please fill out all patient profile fields with valid numbers.");
                return;
            }
        
            showLoading(startButton, true);
            hideError();
            clearLogs();

            const response = await fetch('/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    initial_data: orderedData, // Send the correctly ordered array
                    initial_data_map: initialDataMap // Send the map for logging
                })
            });
            
            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
            const result = await response.json();
            
            if (result.error) throw new Error(result.error);
            
            state = result.current_state; // Store the state
            testCount = 0; // Reset test count
            
            // --- Update UI ---
            step1Div.classList.add('hidden');
            step2Div.classList.remove('hidden');
            
            // Log the initial data
            let profileHTML = '<strong>Patient Profile (Input):</strong><ul class="list-disc list-inside mt-1" style="font-size: 13px;">';
            for(const key of INITIAL_FEATURES) { // Log in the correct order
                profileHTML += `<li><strong>${key}:</strong> ${initialDataMap[key]}</li>`;
            }
            profileHTML += '</ul>';
            addLog(ICONS.USER, profileHTML);

            // Log the fingerprint
            const fingerprint = result.fingerprint.map(n => n.toFixed(4)).join(', ');
            addLog(ICONS.STETHOSCOPE, `<strong>Patient Fingerprint (SAE Output):</strong><br>[${fingerprint}]`);
            
            processResponse(result);

        } catch (err) {
            showError(`Error: ${err.message}`);
        } finally {
            showLoading(startButton, false);
        }
    }

    async function onSubmitTestResult(e) {
        e.preventDefault();
        const testName = nextStepInput.name;
        const testValue = parseFloat(nextStepInput.value);

        if (isNaN(testValue)) {
            showError("Please enter a valid number for the test result.");
            return;
        }
        
        showLoading(nextStepButton, true);
        hideError();
        step2Div.classList.add('hidden'); // Hide prompt while thinking

        try {
            const response = await fetch('/next_step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    current_state: state,
                    test_name: testName,
                    test_value: testValue
                })
            });

            if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
            const result = await response.json();
            
            if (result.error) throw new Error(result.error);

            state = result.current_state; // Store the *updated* state
            testCount++; // Increment test count
            
            addLog(ICONS.BEAKER, `<strong>Test Result:</strong> ${testName} = <strong>${testValue}</strong>`);
            
            processResponse(result);

        } catch (err) {
            showError(`Network Error: ${err.message}`);
            step2Div.classList.remove('hidden'); // Show prompt again on error
        } finally {
            showLoading(nextStepButton, false);
        }
    }

    function onReset() {
        state = null;
        testCount = 0;
        step3Div.classList.add('hidden');
        step1Div.classList.remove('hidden');
        hideError();
        clearLogs();
        
        // Clear form values, but placeholders remain
        initialDataForm.reset(); 
        
        // Reset diagnosis card style
        diagnosisCard.className = 'diagnosis-card';
        diagnosisText.className = "diagnosis-value";
        
        // Reset placeholder checkboxes to checked
        document.getElementById('smoking').checked = true;
        document.getElementById('chest_pain').checked = true;
        document.getElementById('dyspnea').checked = true;
        document.getElementById('fatigue').checked = true;
    }

    // --- 5. Helper Functions ---

    function processResponse(response) {
        // Always update the live confidence display
        updateConfidence(response.current_prediction, response.current_confidence);

        if (response.status === 'needs_test') {
            const testName = response.next_test;
            addLog(ICONS.STETHOSCOPE, `[AGENT]: Requesting test: <strong>${testName}</strong>`);
            
            // Show the prompt for the next test
            nextStepLabel.textContent = `Agent requests result for ${testName}:`;
            nextStepInput.name = testName;
            nextStepInput.value = '';
            step2Div.classList.remove('hidden'); // Show this section
            nextStepInput.focus();
            
        } else if (response.status === 'diagnosis_complete') {
            const diagnosis = response.diagnosis;
            addLog(ICONS.CLIPBOARD, `[AGENT]: Final Diagnosis: <strong>${diagnosis}</strong>`);
            
            // Show the final diagnosis screen
            step2Div.classList.add('hidden');
            step3Div.classList.remove('hidden');
            
            // Populate the final card
            diagnosisText.textContent = diagnosis;
            confidenceText.textContent = `${(response.current_confidence * 100).toFixed(1)}% Confidence`;
            diagnosisCard.classList.add(diagnosis); // Adds 'Healthy', 'CAD', etc. as a class
            
            // Populate the stats
            const saved = TOTAL_TESTS - testCount;
            const pct = (saved / TOTAL_TESTS) * 100;
            testsPerformedText.textContent = testCount;
            efficiencyGainText.textContent = `${pct.toFixed(0)}%`;
            
            // Copy the log from the live view to the final view
            finalPathwayLog.innerHTML = livePathwayLog.innerHTML;
        }
    }
    
    function updateConfidence(prediction, confidence) {
        currentPredictionText.textContent = prediction;
        currentConfidenceText.textContent = `${(confidence * 100).toFixed(1)}%`;
        
        // Set color based on prediction
        currentPredictionText.style.color = 'var(--text)';
        if (prediction === 'Healthy') currentPredictionText.style.color = 'var(--success)';
        if (prediction === 'CAD') currentPredictionText.style.color = 'var(--danger)';
        if (prediction === 'HF') currentPredictionText.style.color = 'var(--warning)';
        if (prediction === 'STR') currentPredictionText.style.color = 'var(--info)';
    }

    function addLog(icon, message) {
        const div = document.createElement('div');
        div.className = 'log-item';
        div.innerHTML = `
            <div class="log-icon">${icon}</div>
            <div class="log-content">
                <div class="log-title">${message}</div>
            </div>
        `;
        livePathwayLog.appendChild(div);
        livePathwayLog.scrollTop = livePathwayLog.scrollHeight;
    }
    
    function clearLogs() {
        livePathwayLog.innerHTML = '';
        finalPathwayLog.innerHTML = '';
    }

    function showLoading(button, isLoading) {
        const span = button.querySelector('span');
        const spinner = button.querySelector('.spinner');
        if (isLoading) {
            button.disabled = true;
            span.style.display = 'none';
            spinner.classList.remove('hidden');
        } else {
            button.disabled = false;
            span.style.display = 'inline';
            spinner.classList.add('hidden');
        }
    }

    function showError(message) {
        errorText.textContent = `Error: ${message}`;
        errorBox.classList.remove('hidden');
    }

    function hideError() {
        errorBox.classList.add('hidden');
    }
  </script>
</body>
</html>
"""

# --- 8. Run the Flask App ---
if __name__ == '__main__':
    if 'agent' in locals() and 'scaler' in locals() and 'autoencoder' in locals():
        print("\n--- Web Server is READY ---")
        print("Open your browser and go to: http://127.0.0.1:5000")
        app.run(debug=False, port=5000)
    else:
        print("\n[FATAL ERROR] Models did not load. Cannot start server.")