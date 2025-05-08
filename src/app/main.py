from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

model = joblib.load('/home/yasser/Desktop/machine_learning/src/app/models/xgboost_model.pkl')

class InputData(BaseModel):
    SK_ID_CURR: int
    DAYS_BIRTH: int
    REGION_RATING_CLIENT_W_CITY: int
    REGION_RATING_CLIENT: int
    DAYS_LAST_PHONE_CHANGE: int
    NAME_EDUCATION_TYPE: int
    CODE_GENDER: int
    DAYS_ID_PUBLISH: int
    REG_CITY_NOT_WORK_CITY: int

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[:, 1][0]
    return {
        "SK_ID_CURR": int(data.SK_ID_CURR),
        "probability": float(round(prob, 2))
    }


