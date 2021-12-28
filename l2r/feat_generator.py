import os
import csv
import pandas as pd
from gensim import models
from gensim.similarities import Similarity
from doc_helper import DocRetrieval

class GenFeat:
    def __init__(self, mode: str, gt_path: str):
        self.mode = mode
        self.path_ans = gt_path
        self.ans = self.parse_ans()
        self.corpus = DocRetrieval(mode)
        self.file_path = f'feat_{mode}.csv'

    def parse_ans(self):
        ans = dict()
        with open(self.path_ans, newline='') as csvfile:
            rows = csv.DictReader(csvfile)
            for row in rows:
                qid = row['topic']
                docs = row['doc'].split()
                ans[qid] = docs
        
        return ans
    
    def build_index(self, transform):
        index_path = f'./cache_sim/{self.mode}'
        if len(os.listdir(index_path)) > 1:
            index = Similarity.load(f'{index_path}/pre')
        else:
            index = Similarity(f'{index_path}/pre', transform[self.corpus.corpus], len(self.corpus.doc2bow.dictionary))
            index.save(f'{index_path}/pre')
        
        return index


    def gen_feat(self):
        def fill_qid(df, qid):
            df = df.assign(q_id=[qid] * 100000)
            return df
        
        def fill_docid(df, did):
            df = df.assign(doc_id=did)
            return df
        
        def fill_label(df, qid, ans):
            if self.mode == 'train':
                df = df.assign(label=df["doc_id"].map(lambda x: 1 if x in ans[qid] else 0))
            else:
                df = df.assign(label=[0] * 100000)
            return df
        
        def fill_feat(df, feat):
            df = df.assign(feature=feat)
            return df
    
        if os.path.exists(self.file_path):
            print(f'Feature of "{self.mode}" already exists!')
            return None

        tfidf = models.TfidfModel(self.corpus.corpus)
        index = self.build_index(tfidf)

        pd.DataFrame(columns=['q_id', 'doc_id', 'label', 'feature']).to_csv(self.file_path, index=False)
        for i, q in enumerate(self.corpus.query2bow):
            qid = self.corpus.query.no2id[i]
            sims = index[tfidf[q]]

            df = (pd.DataFrame().pipe(fill_qid, qid)
                                .pipe(fill_docid, self.corpus.documents.no2id)
                                .pipe(fill_label, qid, self.ans)
                                .pipe(fill_feat, sims))
            
            df.to_csv(self.file_path, mode='a', header=None, index=False)
            