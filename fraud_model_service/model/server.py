# from fastapi import FastAPI
import joblib
import numpy as np
from paysim_processor_pipeline import PaySimPreprocessor
from tensorflow.keras.preprocessing.text import Tokenizer

# Load preprocessor
preprocessor = joblib.load('paysim_preprocessor.joblib')
model = joblib.load('nbModel.joblib')

# Simulated new transaction (as dict)
new_txn = {
    'type': 'CASH_OUT',
    'amount': 500000,
    'oldbalanceOrg': 1000000,
    'newbalanceOrig': 500000,
    'oldbalanceDest': 0,
    'newbalanceDest': 500000,
    'nameOrig': 'C12345678',
    'nameDest': 'M87654321',
    'isFlaggedFraud': 0
}

# Transform
processed = preprocessor.transform(new_txn)
print(processed)
model.predict(processed)


# app = FastAPI()

# @app.get('/')
# def valconModel_root():
#     return {'message': 'Valcon model API'}