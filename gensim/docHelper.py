import os
import xml.etree.ElementTree as ET
import xml.dom.minidom

from gensim import corpora
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import stem_text

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