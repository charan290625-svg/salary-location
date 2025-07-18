
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data
df = pd.read_csv("Salary_Data.csv")
df.dropna(inplace=True)

# Encode categorical data
df['Education Level'].replace(["Bachelor's Degree", "Master's Degree", "phD"], ["Bachelor's", "Master's", "PhD"], inplace=True)
df['Education Level'] = df['Education Level'].map({"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3})

# Remove 'Gender' and 'Age' columns if present
df = df.drop(columns=[col for col in ['Gender', 'Age'] if col in df.columns])

# Add dummy data if 'Location' and 'Industry' are not present
if 'Location' not in df.columns:
    df['Location'] = np.random.choice(['Urban', 'Rural', 'Suburban'], size=len(df))
if 'Industry' not in df.columns:
    df['Industry'] = np.random.choice(['Tech', 'Finance', 'Healthcare', 'Education'], size=len(df))

df = pd.get_dummies(df, columns=['Location', 'Industry'], drop_first=True)

# Reduce Job Titles with count < 25
job_title_count = df['Job Title'].value_counts()
job_title_edited = job_title_count[job_title_count <= 25]
df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in job_title_edited else x)
df = pd.get_dummies(df, columns=['Job Title'], drop_first=True)

# Split data
X = df.drop('Salary', axis=1)
y = df['Salary']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Random Forest Regressor (Best model)
model = RandomForestRegressor(n_estimators=20)
model.fit(x_train, y_train)

# Streamlit App
st.title("💼 Salary Prediction App")

education = st.selectbox("Education Level", ['High School', "Bachelor's", "Master's", "PhD"])
experience = st.slider("Years of Experience", 0, 40, 1)
location = st.selectbox("Location", ['Urban', 'Rural', 'Suburban'])
industry = st.selectbox("Industry", ['Tech', 'Finance', 'Healthcare', 'Education'])
job_title = st.selectbox("Job Title", sorted([col.replace("Job Title_", "") for col in df.columns if "Job Title_" in col] + ['Others']))

# Preprocess input
education_encoded = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}[education]

input_data = {
    'Education Level': education_encoded,
    'Years of Experience': experience,
}

# One-hot encode Location and Industry
for col in X.columns:
    if "Location_" in col:
        input_data[col] = 1 if col == f'Location_{location}' else 0
    if "Industry_" in col:
        input_data[col] = 1 if col == f'Industry_{industry}' else 0
    if "Job Title_" in col:
        input_data[col] = 1 if col == f'Job Title_{job_title}' else 0

input_df = pd.DataFrame([input_data])

# Prediction
prediction = model.predict(input_df)[0]

# Output
st.subheader("📈 Predicted Salary:")
st.success(f"${prediction:,.2f}")

# Optional: Display feature importance
if st.checkbox("Show Feature Importances"):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    st.bar_chart(importance_df.set_index('Feature'))
