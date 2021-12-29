import os
import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.core.frame import DataFrame

class MyModel:
    def __init__(self, model_path: str, df: DataFrame):
        self.dropped_cols = ["q_id", "doc_id", "label"]
        if not os.path.exists(model_path):
            self.train(model_path, df)
        self.model = xgb.XGBRanker()
        self.model.load_model(model_path)

    def load_data(self, df: DataFrame):    
        X = df.drop(self.dropped_cols, axis=1)
        y = df["label"]
        qids = df.groupby("q_id")["q_id"].count().to_numpy()
        
        return X, y, qids

    def train(self, model_path: str, df: DataFrame):
        X_train, y_train, qids_train = self.load_data(df)
        model = xgb.XGBRanker(
            tree_method='exact',
            booster='gbtree',
            objective='rank:map',
        )

        model.fit(X_train, y_train, group=qids_train, verbose=True)
        model.save_model(model_path)
    
    def predict(self, df: DataFrame):
        return self.model.predict(df.loc[:, ~df.columns.isin(self.dropped_cols)])