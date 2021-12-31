from locale import normalize
import os
import pandas as pd
import numpy as np
from gensim.models import TfidfModel, LsiModel, LdaModel
from gensim.similarities import Similarity
from alt_rank_bm25 import BM25Okapi
from doc_helper import DocRetrieval
from utils import read_pred

class GenFeat:
    def __init__(self, gt_path: str, mode='all'):
        self.mode = mode
        self.ans = read_pred(gt_path)
        self.corpus = DocRetrieval(mode)
        self.models = [TfidfModel, LsiModel, LdaModel]
    
    def build_model(self, idx, model_cls):
        model_path = f'./cache_mod/feat{idx}'
        if os.path.exists(model_path):
            model = model_cls.load(model_path)
        else:
            model = model_cls(self.corpus.corpus)
            model.save(model_path)
        
        return model

    def build_index(self, idx, transform):
        index_path = f'./cache_sim/feat{idx}.index'
        if os.path.exists(index_path):
            index = Similarity.load(index_path)
        else:
            index = Similarity(index_path, transform[self.corpus.corpus], len(self.corpus.doc2bow.dictionary))
            index.save(index_path)
        
        return index

    def gen_feat(self, file_path: str):
        def fill_init(df, qids, dids):
            df = pd.DataFrame(index=pd.MultiIndex.from_product([qids, dids], names=['q_id', 'doc_id'])).reset_index()
            return df
        
        def fill_label(df):
            df['label'] = df.apply(lambda row: 1 if row.q_id in self.ans.keys() and row.doc_id in self.ans[row.q_id] else 0, axis=1)
            return df
        
        def fill_feat(df, feature):
            df[f'feat{len(df.columns) - 3}'] = feature
            return df

        qids = self.corpus.query.no2id
        dids = self.corpus.documents.no2id
        df = (pd.DataFrame().pipe(fill_init, qids, dids)
                            .pipe(fill_label)
        )

        for i, model in enumerate(self.models):
            model = self.build_model(i, model)
            index = self.build_index(i, model)

            sims = [index[model[q]] for q in self.corpus.query2bow]
            df = df.pipe(fill_feat, np.vstack(sims).ravel())

        bm25_mod = BM25Okapi(self.corpus.corpus)
        sims = [bm25_mod.get_scores(q) for q in self.corpus.query2bow]
        df = df.pipe(fill_feat, np.vstack(sims).ravel())

        df.to_csv(file_path, index=False)

        print('Success: feature generation complete!')

            
            
