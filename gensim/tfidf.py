from gensim import models
from docRetrieval import DocRetrieval
import utils

rtv = DocRetrieval('train')
tfidf = models.TfidfModel(rtv.corpus)
ans = rtv.retrieve(tfidf)
utils.write_file(ans, 'submission.csv')
utils.map_score('train_ans.csv', 'submission.csv')