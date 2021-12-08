import os
import xml.etree.ElementTree as ET
import xml.dom.minidom

from gensim import corpora
from gensim import models
from gensim import similarities
from gensim.corpora.mmcorpus import MmCorpus
from gensim.test.utils import datapath
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import stem_text

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def write_file(ans, path):
    with open(path, 'w') as f:
        for query, docs in zip(ans['topic'], ans['doc']):
            f.write(f'{query},{docs}\n')


def read_file(path):  
    data_dict = {}  
    with open(path, 'r') as f:
        for line in f:
            query, docs = line.split(',')
            data_dict[query] = docs.split(' ')
    return data_dict


def map_score(target_path, pred_path):
    target = read_file(target_path)
    pred = read_file(pred_path)
    precision = []

    for query, docs in pred.items():
        target_set = set(target[query])
        pred_set = set(docs)
        p = len(target_set.intersection(pred_set)) / len(pred_set)
        precision.append(p)

    print(f'map@50: {sum(precision) / len(precision)}')


class DocCollection:
    def __init__(self, dirname):
        self.dirname = dirname
        self.custom_filters = [lambda x: x.lower(), remove_stopwords, strip_numeric, strip_punctuation, stem_text, strip_short]
        self.no2id = [filename for filename in os.listdir(self.dirname)]
    
    def __iter__(self):
        for filename in os.listdir(self.dirname):
            path = os.path.join(self.dirname, filename)

            if 'doc' in self.dirname:
                yield self.retrieve_doc(path)
            else:
                yield self.retrieve_query(path)


    def retrieve_doc(self, path):
        root = ET.parse(path).getroot()
        paragraphs = []

        for p in root.findall('.//p'):
            if p.text is not None:
                words = preprocess_string(p.text, self.custom_filters)
                
                if len(words) > 10:
                    paragraphs += words

        return paragraphs


    def retrieve_query(self, path):
        parser = ET.XMLParser(encoding='utf-8')
        root = ET.parse(path, parser=parser).getroot()
        paragraphs = []
        tags = ['note', 'description', 'summary']

        for tag in tags:
            target = f'.//{tag}'
            text = root.find(target).text
            words = preprocess_string(text, self.custom_filters)
            paragraphs += words

        return paragraphs


class Doc2Bow:
    def __init__(self, docs, dict_filename):
        dirname = 'cache'
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self.dict_path = os.path.join(dirname, dict_filename)
        self.docs = docs
        self.build_dict()


    def build_dict(self):
        # Load dictionary from dict_path if it exists
        # Otherwise, create dictionary and save it to dict_path
        if os.path.exists(self.dict_path):
            dictionary = corpora.Dictionary.load(self.dict_path)

        else:
            dictionary = corpora.Dictionary(self.docs, prune_at=5000000)

            once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
            dictionary.filter_tokens(once_ids)  # remove words that appear only once
            dictionary.compactify()  # remove gaps in id sequence after words that were removed
            dictionary.save(self.dict_path)

        self.dictionary = dictionary


    def __iter__(self):
        for doc in self.docs:
            yield self.dictionary.doc2bow(doc)
        

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
    
    
    def retrieve(self):
        tfidf = models.TfidfModel(self.corpus)
        index = similarities.MatrixSimilarity(tfidf[self.corpus], num_features=174289)
        ans = {'topic': [], 'doc': []}

        for i, q in enumerate(self.query2bow):
            # Find documents of highest score in the format [(doc_no, score), ...]
            sims = index[q]
            top_docs = sorted(enumerate(sims), key=lambda item: -item[1])[:50]

            # Convert doc_no to doc_id
            query_id = self.query.no2id[i]
            doc_ids = ' '.join([self.documents.no2id[doc[0]] for doc in top_docs])
            ans['topic'].append(query_id)
            ans['doc'].append(doc_ids)

        write_file(ans, 'test_pred.csv')


rtv = DocRetrieval('test')
rtv.retrieve()
# map_score('train_ans.csv', 'pred.csv')