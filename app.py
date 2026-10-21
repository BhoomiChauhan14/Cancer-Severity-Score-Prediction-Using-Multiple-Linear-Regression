
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Cancer Severity Predictor",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("cancer_severity_model.pkl")

model = load_model()

st.title("🩺 Cancer Severity Score Prediction")

st.markdown(
    """
    **Model:** Multiple Linear Regression
    """
)

st.divider()

st.subheader("Enter Patient Risk Factors")

col1, col2 = st.columns(2)

with col1:
    smoking = st.slider(
        "Smoking Risk",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

    genetic_risk = st.slider(
        "Genetic Risk",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

    air_pollution = st.slider(
        "Air Pollution Risk",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

with col2:
    alcohol_use = st.slider(
        "Alcohol Use Risk",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

    obesity_level = st.slider(
        "Obesity Level",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )


st.divider()


if st.button("🔮 Predict Severity Score", use_container_width=True):

    input_data = pd.DataFrame({
        "Smoking": [smoking],
        "Genetic_Risk": [genetic_risk],
        "Air_Pollution": [air_pollution],
        "Alcohol_Use": [alcohol_use],
        "Obesity_Level": [obesity_level]
    })

    prediction = model.predict(input_data)[0]

    st.metric(
        label="Predicted Cancer Severity Score",
        value=f"{prediction:.2f}"
    )

    st.info(
        """
        The predicted value is a continuous severity score generated
        by the Multiple Linear Regression model. It should not be
        interpreted as a medical diagnosis.
        """
    )

st.divider()

