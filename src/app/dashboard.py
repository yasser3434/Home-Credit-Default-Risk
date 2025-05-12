import streamlit as st
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Set page configuration for wide layout
st.set_page_config(layout="wide")

# Load the dataset
df = pd.read_csv("/home/yasser/Desktop/machine_learning/data/application_train.csv")

# Function to encode categorical features
def categorical_to_numeric(dataset):
    categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
    le = LabelEncoder()
    for col in categorical_cols:
        dataset[col] = le.fit_transform(dataset[col].astype(str))

# Define the columns for prediction (excluding SK_ID_CURR)
test_columns = [ 
    'DAYS_BIRTH', 
    'REGION_RATING_CLIENT_W_CITY', 
    'REGION_RATING_CLIENT', 
    'DAYS_LAST_PHONE_CHANGE',
    'NAME_EDUCATION_TYPE',
    'CODE_GENDER',
    'DAYS_ID_PUBLISH',
    'REG_CITY_NOT_WORK_CITY',
]

# Make a copy of the dataset to work with
train_df = df.copy()

# Encode categorical columns
categorical_to_numeric(train_df)
train_df = train_df.set_index('SK_ID_CURR')

# Set up the title of the app
st.title("📊 Loan Default Risk Prediction")

# User input for SK_ID_CURR
sk_id = st.number_input("🔍 Enter SK_ID_CURR", step=1)

# If the profile exists in the dataset, use it, otherwise ask the user for inputs
if sk_id in train_df.index:
    st.success("✅ Profile found in training data.")
    profile = train_df[test_columns].loc[sk_id].to_dict()
    profile["SK_ID_CURR"] = int(sk_id)
else:
    st.warning("⚠️ Profile not found. Please enter manually.")
    
    age_in_years = st.number_input("Age (in years)", min_value=18, max_value=120, step=1)
    profile = {
        "SK_ID_CURR": sk_id,
        "DAYS_BIRTH": -age_in_years * 365,
        "CODE_GENDER": st.number_input("Gender (0 for Female, 1 for Male)", help="Please enter '0' for Female and '1' for Male. Refer to the dataset encoding."),
        "REGION_RATING_CLIENT_W_CITY": st.number_input("City Rating", help="City Rating. For example, use values like 1, 2, or 3 based on the client's region."),
        "REGION_RATING_CLIENT": st.number_input("Region Rating", help="Region Rating, similar to city rating. Use values like 1, 2, or 3."),
        "DAYS_LAST_PHONE_CHANGE": st.number_input("Last Phone Change (in days)", help="Number of days since the client's last phone change."),
        "NAME_EDUCATION_TYPE": st.number_input("Education (use encoded values)", help="Use the encoded value for education type. Refer to the dataset encoding."),
        "DAYS_ID_PUBLISH": st.number_input("ID Publishing Date (in days)", help="Number of days since the ID was published."),
        "REG_CITY_NOT_WORK_CITY": st.number_input("Work City", help="Enter '1' for clients who do not work in the city where they reside, otherwise '0'."),
    }

# Layout with columns for displaying the data
col1, col2, col3 = st.columns([1, 2, 1])

# Display demographics information (Age, Gender, Education)
with col1:
    st.subheader("👤 Demographics")
    st.metric("Age", f"{int(abs(profile['DAYS_BIRTH']) // 365)} years")
    st.metric("Gender", df.loc[sk_id, "CODE_GENDER"])
    st.write("📚 Education:", df.loc[sk_id, "NAME_EDUCATION_TYPE"])

with col3:
    st.subheader("🌍 INCOME Info")
    st.write("Client income:", df.loc[sk_id, "AMT_INCOME_TOTAL"])
    st.write("Owns a house:", df.loc[sk_id, "FLAG_OWN_REALTY"])
    st.write("Number of children:", df.loc[sk_id, "CNT_CHILDREN"])

with col2:
    if st.button("📈 Predict"):
        response = requests.post("http://127.0.0.1:8000/predict", json=profile)
        if response.status_code == 200:
            result = response.json()
            prob = float(result["probability"])
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
