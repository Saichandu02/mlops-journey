from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

with open("model.pkl", "rb") as f:
    model=pickle.load(f)
class IrisInput(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float

@app.post("/predict")

def predict(input_data: IrisInput):
    features = np.array([[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]])

    prediction = model.predict(features)

    species =["setosa","versicolor","virginica"]
    result = species[prediction[0]]

    return {"prediction": result}

@app.get("/")
@app.head("/")

def health_check():
    return {"status": "API is running"}