import streamlit as st
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")
st.title("📊 Loan Default Risk Prediction")

# Load dataset
df = pd.read_csv("/home/yasser/Desktop/machine_learning/data/application_train.csv")

# Fit encoders for categorical features
encoders = {}
categorical_fields = [
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR",
    "NAME_CONTRACT_TYPE",
]

for col in categorical_fields:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    le.fit(df[col])
    encoders[col] = le

# User Input
sk_id = st.number_input("Enter SK_ID_CURR", step=1)

# Check if SK_ID_CURR exists in the dataset
if sk_id in df["SK_ID_CURR"].values:
    st.success("Profile found!")
    user_data = df[df["SK_ID_CURR"] == sk_id].iloc[0]  # Retrieve the user's data
    age_years = abs(user_data["DAYS_BIRTH"]) // 365  # Convert days to years
    gender = user_data["CODE_GENDER"]
    education = user_data["NAME_EDUCATION_TYPE"]
    housing = user_data["NAME_HOUSING_TYPE"]
    owns_car = user_data["FLAG_OWN_CAR"]
    contract = user_data["NAME_CONTRACT_TYPE"]
    amt_income = user_data["AMT_INCOME_TOTAL"]
    amt_credit = user_data["AMT_CREDIT"]

    # Display user data
    st.write("User's data retrieved:")
    st.write(f"Age: {age_years} years")
    st.write(f"Gender: {gender}")
    st.write(f"Education Level: {education}")
    st.write(f"Housing Type: {housing}")
    st.write(f"Owns Car: {owns_car}")
    st.write(f"Contract Type: {contract}")
    st.write(f"Income: {amt_income}")
    st.write(f"Credit Amount: {amt_credit}")
else:
    st.warning("Profile not found. Please enter manually.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        if sk_id not in df["SK_ID_CURR"].values:
            age_years = st.number_input("Age (years)", min_value=18, max_value=100)
            amt_income = st.number_input("Total Income", value=100000)
            amt_credit = st.number_input("Credit Amount", value=500000)

    with col2:
        if sk_id not in df["SK_ID_CURR"].values:
            gender = st.selectbox("Gender", encoders["CODE_GENDER"].classes_)
            education = st.selectbox("Education Level", encoders["NAME_EDUCATION_TYPE"].classes_)
            housing = st.selectbox("Housing Type", encoders["NAME_HOUSING_TYPE"].classes_)
            owns_car = st.selectbox("Owns Car", encoders["FLAG_OWN_CAR"].classes_)
            contract = st.selectbox("Contract Type", encoders["NAME_CONTRACT_TYPE"].classes_)

    submitted = st.form_submit_button("📈 Predict")

    if submitted:
        # Prepare the profile for prediction (from user input or the retrieved profile)
        profile = {
            "SK_ID_CURR": int(sk_id),
            "DAYS_BIRTH": -int(age_years * 365),  # Convert years to negative days
            "AMT_INCOME_TOTAL": int(amt_income),
            "AMT_CREDIT": int(amt_credit),
            "CODE_GENDER": int(encoders["CODE_GENDER"].transform([gender])[0]),
            "NAME_EDUCATION_TYPE": int(encoders["NAME_EDUCATION_TYPE"].transform([education])[0]),
            "NAME_HOUSING_TYPE": int(encoders["NAME_HOUSING_TYPE"].transform([housing])[0]),
            "FLAG_OWN_CAR": int(encoders["FLAG_OWN_CAR"].transform([owns_car])[0]),
            "NAME_CONTRACT_TYPE": int(encoders["NAME_CONTRACT_TYPE"].transform([contract])[0]),
        }

        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=profile)
            if response.status_code == 200:
                result = response.json()
                prob = float(result["probability"])
                st.success(f"Prediction for SK_ID_CURR {sk_id}:")
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 30px;">
                        <h2 style="color: #333;">Prediction Result</h2>
                        <div style="font-size: 48px; color: {'red' if prob > 0.5 else 'green'};">
                            {round(prob * 100, 2)}%
                        </div>
                        <p style="color: #777;">Probability of loan rejection</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
