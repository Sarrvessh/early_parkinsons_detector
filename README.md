# Parkinson's Disease Prediction using Machine Learning and Deep Learning

## Overview
This project aims to predict early signs of Parkinson’s Disease using a combination of **XGBoost (XGB)** and a **SHAP-based Neural Network (SHAP-NN)**. The application is built using **Streamlit** for a user-friendly interface and allows users to input data manually or upload a voice recording for prediction.

## Features
- **Manual Entry:** Users can enter comma-separated values of 22 key vocal biomarkers.
- **Voice-Based Prediction:** Users can upload a `.wav` file, and the system will extract relevant features automatically.
- **Ensemble Model Prediction:** The project uses an ensemble of **XGBoost** and **SHAP-NN** to provide accurate predictions.
- **Real-time Feedback:** Results are displayed with probability scores and actionable insights.

## Dataset and Features
The model uses key vocal biomarkers for prediction:
- Fundamental frequencies (MDVP:Fo, MDVP:Fhi, MDVP:Flo)
- Jitter and shimmer measures
- Noise-to-Harmonics Ratio (NHR), Harmonics-to-Noise Ratio (HNR)
- Nonlinear dynamic measures (RPDE, DFA, Spread1, Spread2, D2, PPE)

## Model Architecture
1. **XGBoost Model:** 
   - Trained on Parkinson’s Disease dataset
   - Outputs probability of disease

2. **SHAP-Based Neural Network (SHAP-NN):**
   - 3-layer neural network with ReLU activations
   - Uses SHAP-based features for better interpretability
   - Outputs probability of disease

3. **Ensemble Prediction:**
   - Weighted averaging: **60% XGBoost + 40% SHAP-NN**
   - Final probability determines Parkinson’s risk

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed along with the following dependencies:

```bash
pip install streamlit pandas numpy joblib torch torchvision torchaudio librosa scikit-learn scipy xgboost
```

### Running the Application
```bash
streamlit run app.py
```

## How to Use
1. **Enter Data Manually:** Copy and paste 22 feature values as comma-separated input.
2. **Upload a `.wav` File:** The model will extract vocal features and predict.
3. **Click "Predict":** Get an instant prediction along with a probability score.
4. **Reset:** Clears inputs and allows fresh predictions.

## File Structure
```
├── parkinsons_xgb_model.pkl       # Pre-trained XGBoost Model
├── parkinsons_shap_nn.pth         # Pre-trained SHAP-Based Neural Network
├── scaler.pkl                     # Scaler for data normalization
├── app.py                         # Streamlit application
├── README.md                      # Documentation
```

## Example Prediction
```
Input: 228.34, 251.16, 195.89, 0.0028, 0.00002, ..., 0.211

Output: "Parkinson’s early signs are detected. Contact doctor immediately."
Probability: 0.78
```

## Authors
- **Sarvesh PV**
