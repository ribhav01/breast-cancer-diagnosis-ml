import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


st.set_page_config(page_title="Breast Cancer Diagnosis Predictor")

st.title("🔬 Breast Cancer Diagnosis Predictor")
st.write(
    "This app uses a logistic regression model trained on the Wisconsin Breast "
    "Cancer dataset (569 patient samples) to predict whether a tumor is "
    "**malignant** or **benign** based on 10 key clinical measurements."
)
st.warning(
    "Educational project only — not a medical diagnostic tool. "
    "Do not use for actual clinical decisions."
)


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

    # Stats used to set slider ranges/defaults
    stats = df[TOP_FEATURES].describe().loc[['min', 'mean', 'max']].T

    return model, scaler, acc, stats

model, scaler, accuracy, stats = load_model()

st.caption(f"Model test accuracy: **{accuracy:.1%}**")


st.header("Enter tumor measurements")
st.write("Adjust the sliders below, or leave at default (average) values.")

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


if st.button("Predict Diagnosis", type="primary"):
    input_df = pd.DataFrame([user_input])[TOP_FEATURES]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.success(f"### Prediction: Benign")
        st.write(f"Confidence: {probability[1]:.1%}")
    else:
        st.error(f"### Prediction: Malignant")
        st.write(f"Confidence: {probability[0]:.1%}")

    st.caption(
        "Remember: this is a demonstration of a machine learning model, "
        "not a medical diagnosis. Always consult a physician."
    )

st.divider()
st.markdown(
    "Built as an independent project exploring machine learning applications "
    "in healthcare. "
    "[View source code on GitHub](https://github.com/ribhav01/breast-cancer-diagnosis-ml)"
)
