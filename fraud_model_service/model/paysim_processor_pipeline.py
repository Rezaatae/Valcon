import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PaySimPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.tokenizer_org = Tokenizer()
        self.tokenizer_dest = Tokenizer()
        self.numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        self.known_type_dummies = []
        self.expected_columns = []

    def balance_diff(self, data):
        orig_change = (data['newbalanceOrig'] - data['oldbalanceOrg']).astype(int)
        data['orig_txn_diff'] = np.where(orig_change < 0, data['amount'] + orig_change, data['amount'] - orig_change)
        data['orig_diff'] = (data['orig_txn_diff'].astype(int) != 0).astype(int)

        dest_change = (data['newbalanceDest'] - data['oldbalanceDest']).astype(int)
        data['dest_txn_diff'] = np.where(dest_change < 0, data['amount'] + dest_change, data['amount'] - dest_change)
        data['dest_diff'] = (data['dest_txn_diff'].astype(int) != 0).astype(int)

        data.drop(['orig_txn_diff', 'dest_txn_diff'], axis=1, inplace=True)

    def surge_indicator(self, data):
        data['surge'] = (data['amount'] > 450000).astype(int)

    def frequency_receiver(self, data):
        freq = data['nameDest'].value_counts()
        data['freq_dest'] = data['nameDest'].map(freq).gt(20).astype(int)

    def merchant(self, data):
        data['merchant'] = data['nameDest'].str.startswith('M').astype(int)

    def tokenize_names(self, df):
        df['customers_org'] = pad_sequences(
            self.tokenizer_org.texts_to_sequences(df['nameOrig']), maxlen=1
        ).astype(int)
        df['customers_dest'] = pad_sequences(
            self.tokenizer_dest.texts_to_sequences(df['nameDest']), maxlen=1
        ).astype(int)

    def fit(self, df):
        # Feature engineering
        self.balance_diff(df)
        self.surge_indicator(df)
        self.frequency_receiver(df)
        self.merchant(df)

        # One-hot encoding
        df = pd.concat([df, pd.get_dummies(df['type'], prefix='type_')], axis=1)
        df.drop(['type'], axis=1, inplace=True)
        self.known_type_dummies = [col for col in df.columns if col.startswith('type_')]

        # Tokenizers
        self.tokenizer_org.fit_on_texts(df['nameOrig'])
        self.tokenizer_dest.fit_on_texts(df['nameDest'])

        # Create tokenized columns during fit too
        self.tokenize_names(df)

        # Scale numeric features
        self.scaler.fit(df[self.numeric_cols])

        # Drop non-feature columns
        df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud', 'isFraud'], axis=1, errors='ignore')
        df = df.drop(columns=['surge', 'merchant'], errors='ignore')  # if not used

        # Capture training feature structure (including customer tokens)
        self.expected_columns = df.columns.tolist()

    def transform(self, df):
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        elif isinstance(df, pd.Series):
            df = df.to_frame().T

        # Feature engineering
        self.balance_diff(df)
        self.surge_indicator(df)
        self.frequency_receiver(df)
        self.merchant(df)

        # One-hot encode transaction type
        df = pd.concat([df, pd.get_dummies(df['type'], prefix='type_')], axis=1)
        df.drop(['type'], axis=1, inplace=True)
        for col in self.known_type_dummies:
            if col not in df.columns:
                df[col] = 0

        # Tokenized names
        self.tokenize_names(df)

        # Scale numeric
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])

        # Drop unneeded
        df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud', 'isFraud'], axis=1, errors='ignore')
        df = df.drop(columns=['surge', 'merchant'], errors='ignore')  # unless trained with them

        # Reorder columns to match training
        df = df.reindex(columns=self.expected_columns, fill_value=0)

        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

