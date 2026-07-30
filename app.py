import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------
# Custom CSS: cards, shadows, hover zoom, gradient header
# ---------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', -apple-system, sans-serif;
    }

    /* Soft animated pink gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fff5f7 0%, #ffe4ec 25%, #fff0f3 50%, #ffe9ef 75%, #fff5f7 100%);
        background-size: 300% 300%;
        animation: gentleDrift 22s ease infinite;
    }
    @media (prefers-reduced-motion: reduce) {
        [data-testid="stAppViewContainer"] {
            animation: none;
        }
    }
    @keyframes gentleDrift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Subtle ribbon watermark, fixed in the corner, non-interactive */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: -40px;
        right: -40px;
        width: 340px;
        height: 340px;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'%3E%3Cpath d='M100 90 C70 60, 40 70, 45 100 C50 130, 85 130, 100 105 C115 130, 150 130, 155 100 C160 70, 130 60, 100 90 Z' fill='none' stroke='%23e8879f' stroke-width='7'/%3E%3Cpath d='M100 105 L75 175 L100 160 L125 175 Z' fill='none' stroke='%23e8879f' stroke-width='7' stroke-linejoin='round'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-size: contain;
        opacity: 0.12;
        pointer-events: none;
        z-index: 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

    .hero-banner {
        background: linear-gradient(135deg, #d1477a 0%, #f0729a 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(209, 71, 122, 0.2);
        position: relative;
        z-index: 1;
    }
    .hero-banner h1 {
        color: white;
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-size: 1.02rem;
    }

    .accuracy-badge {
        display: inline-block;
        background: linear-gradient(135deg, #c2185b 0%, #e91e8c 100%);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 14px rgba(233, 30, 140, 0.3);
    }

    div[data-testid="stSlider"] {
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    }
    div[data-testid="stSlider"]:hover {
        transform: scale(1.015);
        background: rgba(233, 30, 140, 0.05);
        box-shadow: 0 4px 14px rgba(233, 30, 140, 0.12);
        z-index: 2;
        position: relative;
    }

    div[data-testid="stSlider"] [role="slider"] {
        box-shadow: 0 2px 8px rgba(233, 30, 140, 0.4);
    }

    div.stButton > button {
        background: linear-gradient(135deg, #c2185b 0%, #e91e8c 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05rem;
        box-shadow: 0 4px 16px rgba(233, 30, 140, 0.3);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(233, 30, 140, 0.4);
    }

    .result-benign {
        background: linear-gradient(135deg, #e6fff5 0%, #d0fbe8 100%);
        border-left: 6px solid #11998e;
        border-radius: 12px;
        padding: 1.4rem 1.7rem;
        box-shadow: 0 6px 24px rgba(17, 153, 142, 0.18);
        color: #0d3d33;
    }
    .result-benign .result-title {
        color: #0d3d33;
    }
    .result-benign strong {
        color: #0d3d33;
    }
    .result-malignant {
        background: linear-gradient(135deg, #fff0f0 0%, #ffe1e1 100%);
        border-left: 6px solid #e63946;
        border-radius: 12px;
        padding: 1.4rem 1.7rem;
        box-shadow: 0 6px 24px rgba(230, 57, 70, 0.18);
        color: #5c1015;
    }
    .result-malignant .result-title {
        color: #5c1015;
    }
    .result-malignant strong {
        color: #5c1015;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .footer-note {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }

    /* Respect reduced-motion preference */
    @media (prefers-reduced-motion: reduce) {
        * {
            transition: none !important;
            animation: none !important;
        }
    }

    /* Visible focus states for keyboard navigation */
    div[data-testid="stSlider"] [role="slider"]:focus-visible,
    div.stButton > button:focus-visible {
        outline: 3px solid #e91e8c;
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Hero header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🎗️ Breast Cancer Diagnosis Predictor</h1>
    <p>A logistic regression model trained on 569 clinical cases, predicting
    tumor diagnosis from 10 key measurements — with 97.4% test accuracy.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "⚠️ Educational project only — not a medical diagnostic tool. "
    "Do not use for actual clinical decisions."
)

# ---------------------------------------------------------
# Load data + train model (cached so it only runs once)
# ---------------------------------------------------------
TOP_FEATURES = [
    'worst texture', 'radius error', 'worst symmetry', 'mean concave points',
    'worst concavity', 'area error', 'worst radius', 'worst area',
    'mean concavity', 'worst concave points'
]

@st.cache_resource
def load_model():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    X = df[TOP_FEATURES]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    stats = df[TOP_FEATURES].describe().loc[['min', 'mean', 'max']].T

    return model, scaler, acc, stats

model, scaler, accuracy, stats = load_model()

st.markdown(f"""
<div style="margin-bottom: 1.2rem;">
    <span class="accuracy-badge">✓ Model Test Accuracy: {accuracy:.1%}</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# User input sliders
# ---------------------------------------------------------
st.subheader("📊 Enter Tumor Measurements")
st.caption("Adjust the sliders below — hover to focus on a measurement.")

col1, col2 = st.columns(2)
user_input = {}

for i, feature in enumerate(TOP_FEATURES):
    min_val = float(stats.loc[feature, 'min'])
    max_val = float(stats.loc[feature, 'max'])
    mean_val = float(stats.loc[feature, 'mean'])

    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        user_input[feature] = st.slider(
            feature.title(),
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val) / 100,
        )

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
predict_clicked = st.button("🔍 Predict Diagnosis", type="primary")

if predict_clicked:
    input_df = pd.DataFrame([user_input])[TOP_FEATURES]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.write("")

    if prediction == 1:
        st.markdown(f"""
        <div class="result-benign">
            <div class="result-title">✅ Prediction: Benign</div>
            <div>Confidence: <strong>{probability[1]:.1%}</strong></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-malignant">
            <div class="result-title">⚠️ Prediction: Malignant</div>
            <div>Confidence: <strong>{probability[0]:.1%}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Remember: this is a demonstration of a machine learning model, "
        "not a medical diagnosis. Always consult a physician."
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="footer-note">
    Built as an independent project exploring machine learning applications in healthcare.<br>
    <a href="https://github.com/ribhav01/breast-cancer-diagnosis-ml" target="_blank">View source code on GitHub</a>
</div>
""", unsafe_allow_html=True)
