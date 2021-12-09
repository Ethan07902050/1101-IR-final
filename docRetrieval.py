import utils
import os
from docHelper import DocCollection, Doc2Bow


from gensim import similarities
from gensim.corpora.mmcorpus import MmCorpus

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        

class DocRetrieval:
    def __init__(self, split='train'):
        doc_name = 'doc'
        dict_name = 'doc.dict'
        corpus_name = 'doc.mm'

        self.documents = DocCollection(doc_name)
        self.query = DocCollection(f'{split}_query')
        self.doc2bow = Doc2Bow(self.documents, dict_name)
        self.query2bow = Doc2Bow(self.query, dict_name)
        self.corpus = self.build_corpus(corpus_name)      


    # Save BOW of documents 
    def build_corpus(self, corpus_name):
        corpus_path = f'./cache/{corpus_name}'
        if not os.path.exists(corpus_path):
            MmCorpus.serialize(corpus_path, self.doc2bow)

        corpus = MmCorpus(corpus_path)
        return corpus

    
    def retrieve(self, transform):
        index = similarities.MatrixSimilarity(transform[self.corpus], num_features=174289)
        ans = {'topic': [], 'doc': []}

        for i, q in enumerate(self.query2bow):
            # Find documents of highest score in the format [(doc_no, score), ...]
            sims = index[transform[q]]
            top_docs = sorted(enumerate(sims), key=lambda item: -item[1])[:50]

            # Convert doc_no to doc_id
            query_id = self.query.no2id[i]
            doc_ids = ' '.join([self.documents.no2id[doc[0]] for doc in top_docs])
            ans['topic'].append(query_id)
            ans['doc'].append(doc_ids)

        return ans