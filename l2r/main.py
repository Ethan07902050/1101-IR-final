import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

import l2r
import feat_generator
from doc_helper import DocCollection
from utils import map_at_k

model_path = 'model'
gt_path = './dataset/train_ans.csv'
df_type = {'q_id': int, 'doc_id': int, 'label': bool, 'feature': float}
no2id = [filename for filename in os.listdir('./dataset/doc')]

def load_data():
    train_df = pd.read_csv(f'feat_train.csv', dtype=df_type)
    test_df = pd.read_csv(f'feat_test.csv', dtype=df_type)
    model = l2r.MyModel(train_df)
    model.train(model_path)

    return train_df, test_df, model

def gen_feature(mode: str):
    gen = feat_generator.GenFeat(mode, gt_path)
    gen.gen_feat()

def get_result(model, df: DataFrame):
    pred = (df.groupby('q_id').apply(lambda x: model.predict(model_path, x))
              .apply(lambda x: np.argsort(x)[::-1][:50])
              .apply(lambda x: " ".join([no2id[i] for i in x]))
              .reset_index()
              .rename(columns={'q_id': 'topic', 0: 'doc'})
    )
    pred.to_csv(f'pred.csv', index=None)

def main():
    gen_feature('train')
    gen_feature('test')

    train_df, test_df, model = load_data()
    
    get_result(model, train_df)
    score = map_at_k('pred.csv', gt_path, 50)
    get_result(model, test_df)

    print(f'------\nDone! MAP@50 = {round(score, 3)}\n------')

if __name__ == '__main__':
    main()
    