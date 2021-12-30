from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import stem_text
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom

custom_filters = [lambda x: x.lower(), remove_stopwords, strip_punctuation, strip_numeric, stem_text, strip_short]

def retrieve_doc(root):        
    paragraphs = []
    
    for p in root.findall('.//p'):
        text = ET.tostring(p, method='text').decode("utf-8")
        words = preprocess_string(text, custom_filters)
                
        if len(words) > 10:
            paragraphs += words

    return ' '.join(paragraphs)


def retrieve_query(root):
    paragraphs = []
    tags = ['description', 'summary']

    for tag in tags:
        target = root.find(f'.//{tag}')
        text = ET.tostring(target, method='text').decode("utf-8")
        words = preprocess_string(text, custom_filters)
        paragraphs += words

    return ' '.join(paragraphs)


def doc2line(dirname, output_path, is_doc=True):
    f = open(output_path, 'w')
    filenames = sorted(os.listdir(dirname))

    for filename in tqdm(filenames):    
        path = os.path.join(dirname, filename)
        parser = ET.XMLParser(encoding='utf-8')
        root = ET.parse(path, parser=parser).getroot()
        doc = retrieve_doc(root) if is_doc else retrieve_query(root)
        
        if doc != '':
            f.write(f'{filename}\t{doc}\n')

    f.close()

doc2line('../doc', '../docStem.txt')
doc2line('../test_query', '../testQueryStem.txt', is_doc=False)