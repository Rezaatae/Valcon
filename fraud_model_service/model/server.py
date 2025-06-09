from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load('nbModel.joblib')

app = FastAPI()

@app.get('/')
def valconModel_root():
    return {'message': 'Valcon model API'}