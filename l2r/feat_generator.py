import os
import csv
import pandas as pd
from gensim.models import TfidfModel, LsiModel, LdaModel
from gensim.summarization.bm25 import BM25
from gensim.similarities import Similarity
from doc_helper import DocRetrieval

class GenFeat:
    def __init__(self, gt_path: str, mode='all'):
        self.mode = mode
        self.ans = self.parse_ans(gt_path)
        self.corpus = DocRetrieval(mode)
        self.models = [TfidfModel, LsiModel, LdaModel]

    def parse_ans(self, gt_path: str):
        ans = dict()
        with open(gt_path, newline='') as csvfile:
            rows = csv.DictReader(csvfile)
            for row in rows:
                qid = row['topic']
                docs = row['doc'].split()
                ans[qid] = docs
        
        return ans
    
    def build_model(self, idx, model_cls):
        model_path = f'./cache_mod/{self.mode}/feat{idx}'
        if os.path.exists(model_path):
            model = model_cls.load(model_path)
        else:
            model = model_cls(self.corpus.corpus)
            model.save(model_path)
        
        return model

    def build_index(self, idx, transform):
        index_path = f'./cache_sim/{self.mode}/feat{idx}'
        if os.path.exists(index_path):
            index = Similarity.load(index_path)
        else:
            index = Similarity(index_path, transform[self.corpus.corpus], len(self.corpus.doc2bow.dictionary))
            index.save(index_path)
        
        return index

    def gen_feat(self, file_path: str):
        def fill_qid(df, qid):
            df = df.assign(q_id=[qid] * 100000)
            return df
        
        def fill_docid(df, did):
            df = df.assign(doc_id=did)
            return df
        
        def fill_label(df, qid, ans):
            if qid in ans.keys():
                df = df.assign(label=df["doc_id"].map(lambda x: 1 if x in ans[qid] else 0))
            else:
                df = df.assign(label=[0] * 100000)
            return df
        
        def fill_feat(df, features):
            for i, feat in enumerate(features):
                df[f'feat{i}'] = feat
            return df
        
        modeles = [self.build_model(i, model) for i, model in enumerate(self.models)]
        indices = [self.build_index(i, model) for i, model in enumerate(modeles)]
        bm25_mod = BM25(self.corpus.corpus)

        pd_columns = ['q_id', 'doc_id', 'label'] + [f'feat{i}' for i, _ in enumerate(modeles)] + ['feat3']
        pd.DataFrame(columns=pd_columns).to_csv(file_path, index=False)
        for i, q in enumerate(self.corpus.query2bow):
            qid = self.corpus.query.no2id[i]
            sims = [index[model[q]] for index, model in zip(indices, modeles)] + [bm25_mod.get_scores(q)]

            df = (pd.DataFrame().pipe(fill_qid, qid)
                                .pipe(fill_docid, self.corpus.documents.no2id)
                                .pipe(fill_label, qid, self.ans)
                                .pipe(fill_feat, sims))
            
            df.to_csv(file_path, mode='a', header=None, index=False)
