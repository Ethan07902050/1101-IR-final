import gensim
import os
import utils
from docHelper import DocCollection

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Corpus(DocCollection):
    def __init__(self, dirname, tokens_only=False):
        super().__init__(dirname)
        self.tokens_only = tokens_only
    
    def __iter__(self):
        for i, filename in enumerate(os.listdir(self.dirname)):
            path = os.path.join(self.dirname, filename)

            if 'doc' in self.dirname:
                doc = self.retrieve_doc(path)
            else:
                doc = self.retrieve_query(path)

            if self.tokens_only:
                yield doc
            else:
                yield gensim.models.doc2vec.TaggedDocument(doc, [i])


documents = Corpus('doc')
query = Corpus('train_query', tokens_only=True)

# model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=10)
# model.build_vocab(documents)
# model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
# model.save('cache/doc2vec')
model = gensim.models.doc2vec.Doc2Vec.load('cache/doc2vec')

ans = {'topic': [], 'doc': []}
for i, q in enumerate(query):
    inferred_vector = model.infer_vector(q)
    sims = model.dv.most_similar([inferred_vector], topn=50)
    top_docs = sorted(sims, key=lambda item: -item[1])

    # Convert doc_no to doc_id
    query_id = query.no2id[i]
    doc_ids = ' '.join([documents.no2id[doc[0]] for doc in top_docs])
    ans['topic'].append(query_id)
    ans['doc'].append(doc_ids)

utils.write_file(ans, 'doc2vec.csv')
utils.map_score('train_ans.csv', 'doc2vec.csv')