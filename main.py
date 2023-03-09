from fastapi import FastAPI
from pydantic import BaseModel
import pickle 
import pandas as pd
app = FastAPI()

class SoilItem(BaseModel):
    N : int
    P : int
    K : int
    temperature : float
    humidity : float
    ph : float
    rainfall : float

with open("trained_model.pkl","rb") as f:
    model = pickle.load(f)

@app.post('/')
async def scoring_endpoints(item:SoilItem):
    df = pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    yhat = model.predict(df)
    
    return {"prediction":str(yhat)}