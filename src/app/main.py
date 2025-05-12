from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import xgboost as xgb

app = FastAPI()

model = joblib.load("/home/yasser/Desktop/machine_learning/src/app/models/xgboost_model.pkl")

class InputData(BaseModel):
    SK_ID_CURR: int
    DAYS_BIRTH: int
    CODE_GENDER: int
    NAME_EDUCATION_TYPE: int
    AMT_INCOME_TOTAL: float
    NAME_HOUSING_TYPE: int
    AMT_CREDIT: float
    FLAG_OWN_CAR: int
    NAME_CONTRACT_TYPE: int

@app.post("/predict")
def predict(data: InputData):
    features = {
        'DAYS_BIRTH': data.DAYS_BIRTH,
        'CODE_GENDER': data.CODE_GENDER,
        'NAME_EDUCATION_TYPE': data.NAME_EDUCATION_TYPE,
        'AMT_INCOME_TOTAL': data.AMT_INCOME_TOTAL,
        'NAME_HOUSING_TYPE': data.NAME_HOUSING_TYPE,
        'AMT_CREDIT': data.AMT_CREDIT,
        'FLAG_OWN_CAR': data.FLAG_OWN_CAR,
        'NAME_CONTRACT_TYPE': data.NAME_CONTRACT_TYPE,
    }

    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[0][1]
    return {
        "SK_ID_CURR": int(data.SK_ID_CURR),
        "probability": round(float(proba), 2)
    }