import streamlit as st
import pandas as pd
from apputil import GroupEstimate

st.title("Week-9 Exercise — GroupEstimate Model")

st.write("""
This exercise builds a simple estimator that predicts the **mean** or **median**
numeric value (y) for each combination of categorical features (X).
""")

# Upload dataset (any CSV you want to test)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    st.subheader("Select Model Inputs")
    categorical_cols = st.multiselect("Select categorical columns (X)", df.columns.tolist())
    target_col = st.selectbox("Select numeric target column (y)", df.columns.tolist())
    estimate_type = st.radio("Choose estimate type", ["mean", "median"])

    if st.button("Train Model"):
        model = GroupEstimate(estimate=estimate_type)
        model.fit(df[categorical_cols], df[target_col])
        st.success("Model trained successfully!")

        st.subheader("Predict on New Data")
        X_new = []
        for col in categorical_cols:
            val = st.text_input(f"Enter value for {col}")
            X_new.append(val)

        if st.button("Predict"):
            prediction = model.predict([X_new])[0]
            if pd.isna(prediction):
                st.warning("⚠️ Combination not seen in training data.")
            else:
                st.success(f"Predicted {target_col}: {prediction:.2f}")
else:
    st.info("Upload a dataset to begin.")
