# **Employee Salary Prediction 💰**

A machine learning web app that predicts employee salaries based on factors such as age, gender, education, experience, and job title.

# **🚀 Tech Stack**

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

# **📂 Project Structure**
```
employee-salary/
├── app.py
├── Salary Prediction.ipynb
├── Salary_Data.xls
├── random_forest.pkl
├── gender_encoder.pkl
├── education_mapping.pkl
├── job_title_columns.pkl
├── feature_columns.pkl
└── all_job_titles.pkl
```

# **⚙️ Installation**
```
git clone https://github.com/m4h3k/employee-salary.git
cd employee-salary
```
```
pip install streamlit pandas numpy scikit-learn joblib
```
```
▶️ Run the App
streamlit run app.py
```
Then open the link in your browser.

# **🧠 How It Works**

The app takes employee details as input, preprocesses the data using saved encoders, and uses a trained Random Forest Regression model to predict the expected salary.

# **📌 Inputs**

- Age
- Gender
- Education Level
- Years of Experience
- Job Title

# **📄 Disclaimer**

This project is for educational purposes. Salary predictions are estimates and should not be considered guaranteed compensation.
