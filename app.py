import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="centered"
)


# =========================================================
# LOAD MODEL AND PREPROCESSING ARTIFACTS
# =========================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("random_forest.pkl")
    gender_encoder = joblib.load("gender_encoder.pkl")
    education_mapping = joblib.load("education_mapping.pkl")
    job_title_columns = joblib.load("job_title_columns.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    all_job_titles = joblib.load("all_job_titles.pkl")

    return (
        model,
        gender_encoder,
        education_mapping,
        job_title_columns,
        feature_columns,
        all_job_titles
    )


(
    model,
    gender_encoder,
    education_mapping,
    job_title_columns,
    feature_columns,
    all_job_titles
) = load_artifacts()


# =========================================================
# TITLE AND DESCRIPTION
# =========================================================

st.title("💰 Salary Prediction")

st.write(
    "This machine learning application predicts an individual's "
    "expected salary based on personal, educational, and professional factors."
)

st.divider()


# =========================================================
# USER INPUTS
# =========================================================

st.subheader("Enter Employee Details")


# ---------------------------------------------------------
# Age
# ---------------------------------------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)


# ---------------------------------------------------------
# Gender
# ---------------------------------------------------------

gender = st.selectbox(
    "Gender",
    options=list(gender_encoder.classes_)
)


# ---------------------------------------------------------
# Education Level
# ---------------------------------------------------------

education = st.selectbox(
    "Education Level",
    options=list(education_mapping.keys())
)


# ---------------------------------------------------------
# Years of Experience
# ---------------------------------------------------------

years_of_experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.5
)


# ---------------------------------------------------------
# Job Title
# ---------------------------------------------------------

job_title = st.selectbox(
    "Job Title",
    options=all_job_titles
)


# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Salary", type="primary"):

    try:

        # -------------------------------------------------
        # Encode Gender
        # -------------------------------------------------

        gender_encoded = gender_encoder.transform([gender])[0]


        # -------------------------------------------------
        # Encode Education Level
        # -------------------------------------------------

        education_encoded = education_mapping[education]


        # -------------------------------------------------
        # Create basic input DataFrame
        # -------------------------------------------------

        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender_encoded],
            "Education Level": [education_encoded],
            "Years of Experience": [years_of_experience]
        })


        # -------------------------------------------------
        # Create all job-title dummy columns
        # -------------------------------------------------

        for column in job_title_columns:
            input_data[column] = False


        # -------------------------------------------------
        # Set selected job title to True
        #
        # If the selected job title is the category that
        # was dropped by drop_first=True, all dummy columns
        # remain False.
        # -------------------------------------------------

        if job_title in job_title_columns:
            input_data[job_title] = True


        # -------------------------------------------------
        # Arrange columns in exactly the same order used
        # during model training
        # -------------------------------------------------

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=False
        )


        # -------------------------------------------------
        # Make prediction
        # -------------------------------------------------

        prediction = model.predict(input_data)[0]


        # -------------------------------------------------
        # Display prediction
        # -------------------------------------------------

        st.success("Salary prediction generated successfully!")

        st.metric(
            label="Predicted Salary",
            value=f"{prediction:,.2f}"
        )


        # -------------------------------------------------
        # Display entered information
        # -------------------------------------------------

        st.divider()

        st.subheader("Prediction Details")

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Age:**", age)
            st.write("**Gender:**", gender)
            st.write("**Education:**", education)

        with col2:

            st.write(
                "**Years of Experience:**",
                years_of_experience
            )

            st.write(
                "**Job Title:**",
                job_title
            )


    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Salary Prediction App | Random Forest Regression"
)
