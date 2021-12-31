import os
import logging
import xml.etree.ElementTree as ET

from gensim import corpora
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import stem_text
from gensim.corpora.mmcorpus import MmCorpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class DocCollection:
    def __init__(self, dirname):
        self.dirname = dirname
        self.custom_filters = [lambda x: x.lower(), remove_stopwords, strip_punctuation, stem_text, strip_short]
        self.no2id = [filename for filename in os.listdir(self.dirname)]

    def __iter__(self):
        for filename in os.listdir(self.dirname):
            path = os.path.join(self.dirname, filename)

            parser = ET.XMLParser(encoding='utf-8')
            root = ET.parse(path, parser=parser).getroot()

            if 'doc' in self.dirname:
                yield self.retrieve_doc(root)
            else:
                yield self.retrieve_que(root)

    def retrieve_doc(self, root: ET.Element):
        paragraphs = []

        for p in root.findall('.//p'):
            text = ET.tostring(p, method='text').decode("utf-8")
            words = preprocess_string(text, self.custom_filters)
            words = [w for w in words if not w.isnumeric()]

            paragraphs += words

        return paragraphs

    def retrieve_que(self, root: ET.Element):
        paragraphs = []
        tags = ['description', 'summary']

        for tag in tags:  # 12/28: drop note
            p = root.find(f'.//{tag}')
            text = ET.tostring(p, method='text').decode("utf-8")
            words = preprocess_string(text, self.custom_filters)
            words = [w for w in words if not w.isnumeric()]

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
            dictionary = corpora.Dictionary(self.docs)  # 12/28: Remove Prun at

            # once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
            # dictionary.filter_tokens(once_ids)
            dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=None)  # 12/30 add filter
            dictionary.compactify()  # remove gaps in id sequence after words that were removed
            dictionary.save(self.dict_path)

        self.dictionary = dictionary

    def __iter__(self):
        for doc in self.docs:
            yield self.dictionary.doc2bow(doc)


class DocRetrieval:
    def __init__(self, split='train'):
        doc_name = './dataset/doc'
        dict_name = 'doc.dict'
        corpus_name = 'doc.mm'

        self.documents = DocCollection(doc_name)
        self.query = DocCollection(f'./dataset/{split}_query')
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

