import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

import l2r
from utils import map_at_k, precision
from feat_generator import GenFeat

gt_path = './dataset/train_ans.csv'
df_path = './feat.csv'
model_path = './model'

df_type = {'q_id': int, 'doc_id': int, 'label': int, 'feat0': float, 'feat1': float, 'feat2': float, 'feat3': float}
split = {'train': set(map(int, os.listdir('./dataset/train_query'))),
         'test': set(map(int, os.listdir('./dataset/test_query')))}
no2id = [filename for filename in os.listdir('./dataset/doc')]

def load_data():
    df = pd.read_csv(df_path, dtype=df_type)
    train_df = df.loc[df['q_id'].isin(split['train'])]
    test_df = df.loc[df['q_id'].isin(split['test'])]

    return train_df, test_df

def get_result(model, df: DataFrame, save_path: str):
    pred = (df.groupby('q_id').apply(lambda x: model.predict(x))
              .apply(lambda x: np.argsort(x)[::-1][:50])
              .apply(lambda x: " ".join([no2id[i] for i in x]))
              .reset_index()
              .rename(columns={'q_id': 'topic', 0: 'doc'})
    )
    pred.to_csv(save_path, index=None)

def main(new_feat: bool):
    # Generate feature file
    if new_feat:
        gen = GenFeat(gt_path)
        gen.gen_feat(df_path)
    
    # Train model
    train_df, test_df = load_data()
    model = l2r.MyModel(model_path, train_df)
    
    # Get predicted results
    pred_train_path = './pred_train.csv'
    pred_test_path = './pred_test.csv'
    
    get_result(model, train_df, pred_train_path)
    get_result(model, test_df, pred_test_path)
    
    score = map_at_k(pred_train_path, gt_path, 50)
    print(pd.DataFrame(precision(pred_train_path, gt_path)))
    print(f'------\nDone! MAP@50 = {round(score, 3)}\n------')

if __name__ == '__main__':
    main(new_feat=True)
