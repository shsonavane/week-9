import streamlit as st
import pandas as pd
from apputil import GroupEstimate

st.title("Week-9 Exercise — GroupEstimate Model")

st.write("""
This exercise builds a simple estimator that predicts the **mean** or **median**
numeric value (y) for each combination of categorical features (X).
""")

# ------------------------------------------
# Initialize session state variables
# ------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "categorical_cols" not in st.session_state:
    st.session_state.categorical_cols = []
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ------------------------------------------
# Upload dataset
# ------------------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # Model inputs
    st.subheader("Select Model Inputs")
    categorical_cols = st.multiselect("Select categorical columns (X)", df.columns.tolist())
    target_col = st.selectbox("Select numeric target column (y)", df.columns.tolist())
    estimate_type = st.radio("Choose estimate type", ["mean", "median"])

    # Train the model
    if st.button("Train Model"):
        st.session_state.model = GroupEstimate(estimate=estimate_type)
        st.session_state.model.fit(df[categorical_cols], df[target_col])
        st.session_state.categorical_cols = categorical_cols
        st.session_state.target_col = target_col
        st.success("Model trained successfully!")

    # Predict section (only if model exists)
    if st.session_state.model is not None:
        st.subheader("Predict on New Data")
        X_new = []
        for col in st.session_state.categorical_cols:
            val = st.text_input(f"Enter value for {col}")
            X_new.append(val)

        if st.button("Predict"):
            prediction = st.session_state.model.predict([X_new])[0]
            st.session_state.prediction = prediction
            if pd.isna(prediction):
                st.warning("Combination not seen in training data.")
            else:
                st.success(f"Predicted {st.session_state.target_col}: {prediction:.2f}")

        # Keep last prediction visible
        if st.session_state.prediction is not None:
            if pd.isna(st.session_state.prediction):
                st.warning("Combination not seen in training data.")
            else:
                st.info(f"Last prediction: {st.session_state.prediction:.2f}")

else:
    st.info("Upload a dataset to begin.")
