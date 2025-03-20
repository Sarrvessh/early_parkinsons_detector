import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import librosa
import io
from scipy.stats import skew, kurtosis
import tempfile

# Load trained models and scaler
xgb_model = joblib.load("parkinsons_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names from dataset
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# Define Neural Network Architecture
class ShapNN(nn.Module):
    def __init__(self, input_dim):
        super(ShapNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load SHAP-based Neural Network model
input_dim = len(feature_names)  # Ensure input dimensions match
shap_nn = ShapNN(input_dim)
shap_nn.load_state_dict(torch.load("parkinsons_shap_nn.pth"))
shap_nn.eval()

# Streamlit UI
st.title("Parkinson's Disease Prediction")

st.markdown(
    "**There are two Input Methods:**\n"
    "- **Manual Entry:** Paste comma-separated values.\n"
    "- **Voice Recording:** Upload a `.wav` file or use microphone."
)

# Display feature order for reference
st.subheader("Feature Order")
st.write(", ".join(feature_names))

# Accept User Input
user_input_str = st.text_area("Paste values (comma-separated)", "")

st.subheader("Or you can upload a `.wav` file to predict:")
# Audio File Upload or Recording
audio_file = st.file_uploader("Upload a voice recording (`.wav`)", type=["wav"])

# Function to Extract Voice Features
def extract_voice_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)

        # Extract features
        features = [
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.max(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.min(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(librosa.feature.rms(y=y)),
            np.std(librosa.feature.rms(y=y)),
            np.mean(librosa.feature.zero_crossing_rate(y)),
            skew(librosa.feature.zero_crossing_rate(y)[0]),
            kurtosis(librosa.feature.zero_crossing_rate(y)[0]),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=6)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=9)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=11)),
            np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12))
        ]
        return np.array(features)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# If file is uploaded, extract features
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_wav.write(audio_file.read())
        extracted_features = extract_voice_features(temp_wav.name)
        if extracted_features is not None:
            user_input_str = ", ".join(map(str, extracted_features))

# Prediction Processing
if st.button("Predict"):
    try:
        # Convert input string to array
        user_input = np.array([float(x.strip()) for x in user_input_str.split(",")])

        # Ensure correct number of inputs
        if len(user_input) != len(feature_names):
            st.error(f"Expected {len(feature_names)} values, but got {len(user_input)}. Please check your input.")
        else:
            # Reshape and scale input
            input_array = user_input.reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # Predict with XGBoost
            xgb_prediction = xgb_model.predict(input_array)[0]
            xgb_proba = xgb_model.predict_proba(input_array)[0][1]  

            # Predict with SHAP-based Neural Network
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            with torch.no_grad():
                shap_nn_proba = shap_nn(input_tensor).item()
            shap_nn_prediction = int(shap_nn_proba > 0.5)  

            # Ensemble Prediction (Weighted Average)
            ensemble_proba = (0.6 * xgb_proba) + (0.4 * shap_nn_proba)
            threshold = 0.5
            ensemble_prediction = int(ensemble_proba > threshold)

            # Display results
            st.subheader("Prediction Results")
            st.write(f"**Prediction based on various metrics:** {'Healthy, no early signs of Parkinsons Disease found' if ensemble_prediction == 0 
                                                                 else 'Parkinson’s early signs are detected. Contact doctor immediately.'} (Probability: {ensemble_proba:.2f})")

            # Provide insights
            if ensemble_prediction == 1:
                st.warning("The model predicts a high probability of Parkinson’s. Consider consulting a healthcare professional.")
            else:
                st.success("The model predicts no signs of Parkinson’s based on the input values.")

    except ValueError:
        st.error("Invalid input! Ensure all values are numerical and separated by commas.")

# Reset button
if st.button("Reset"):
    st.rerun()
