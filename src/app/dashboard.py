import streamlit as st
import requests

st.title("Prediction Dashboard")

sk_id = st.number_input("SK_ID_CURR", step=1)
DAYS_BIRTH = st.number_input("DAYS_BIRTH")
REGION_RATING_CLIENT_W_CITY = st.number_input("REGION_RATING_CLIENT_W_CITY")
REGION_RATING_CLIENT = st.number_input("REGION_RATING_CLIENT")
DAYS_LAST_PHONE_CHANGE = st.number_input("DAYS_LAST_PHONE_CHANGE")
NAME_EDUCATION_TYPE = st.number_input("NAME_EDUCATION_TYPE")
CODE_GENDER = st.number_input("CODE_GENDER")
DAYS_ID_PUBLISH = st.number_input("DAYS_ID_PUBLISH")
REG_CITY_NOT_WORK_CITY = st.number_input("REG_CITY_NOT_WORK_CITY")


if st.button("Predict"):
    payload = {
        "SK_ID_CURR": sk_id,
        "DAYS_BIRTH": DAYS_BIRTH,
        "REGION_RATING_CLIENT_W_CITY": REGION_RATING_CLIENT_W_CITY,
        'REGION_RATING_CLIENT': REGION_RATING_CLIENT,
        'DAYS_LAST_PHONE_CHANGE': DAYS_LAST_PHONE_CHANGE,
        'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE,
        'CODE_GENDER': CODE_GENDER,
        'DAYS_ID_PUBLISH': DAYS_ID_PUBLISH,
        'REG_CITY_NOT_WORK_CITY': REG_CITY_NOT_WORK_CITY,
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    if response.status_code == 200:
        result = response.json()
        
        st.write(f"Predicted Probability: {round(result['probability'], 2) * 100}%")
    else:
        st.error("Error in API request")
