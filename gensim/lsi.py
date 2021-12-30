from gensim import models
from docRetrieval import DocRetrieval
import utils
import tempfile

rtv = DocRetrieval('test')
tfidf = models.TfidfModel(rtv.corpus)
tfidf_corpus = tfidf[rtv.corpus]
# lsi = models.LsiModel.load('cache/d500.lsi')
lsi = models.LsiModel(tfidf_corpus, id2word=rtv.doc2bow.dictionary, num_topics=500)
# lsi.save('cache/d500.lsi')

ans = rtv.retrieve(lsi)
utils.write_file(ans, 'pred.csv')
utils.map_score('train_ans.csv', 'pred.csv')