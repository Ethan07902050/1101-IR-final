import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.core.frame import DataFrame

class MyModel:
    def __init__(self, df: DataFrame):
        self.df = df
        self.dropped_cols = ["q_id", "doc_id", "label"]

    def load_data(self, df: DataFrame):    
        X = df.drop(self.dropped_cols, axis=1)
        y = df["label"]
        qids = df.groupby("q_id")["q_id"].count().to_numpy()
        
        return X, y, qids

    def train(self, model_path: str):
        X_train, y_train, qids_train = self.load_data(self.df)
        model = xgb.XGBRanker(  
            tree_method='exact',
            booster='gbtree',
            objective='rank:ndcg',
            random_state=42, 
            learning_rate=0.1,
            colsample_bytree=0.9, 
            eta=0.05, 
            max_depth=6, 
            n_estimators=110, 
            subsample=0.75
        )

        model.fit(X_train, y_train, group=qids_train, verbose=True)
        model.save_model(model_path)
    
    def predict(self, model_path: str, df: str):
        model = xgb.XGBRanker()
        model.load_model(model_path)

        return model.predict(df.loc[:, ~df.columns.isin(self.dropped_cols)])
