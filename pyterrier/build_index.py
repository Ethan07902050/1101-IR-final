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

from pathlib import Path
import json
import argparse
import ssl

import pyterrier as pt

if not pt.started():
    ssl._create_default_https_context = ssl._create_unverified_context
    pt.init()

custom_filters = [
    lambda x: x.lower(),
    remove_stopwords,
    strip_punctuation,
    strip_numeric,
]


def retrieve_doc(root):
    paragraphs = []

    for p in root.findall(".//p"):
        text = ET.tostring(p, method="text").decode("utf-8")
        words = preprocess_string(text, custom_filters)
        if len(words) > 10:
            paragraphs += words

    return " ".join(paragraphs)


def doc2dict(dirname: Path):
    count = len([file for file in dirname.iterdir()])

    for path in tqdm(dirname.iterdir(), total=count):
        filename = path.stem
        parser = ET.XMLParser(encoding="utf-8")
        root = ET.parse(path, parser=parser).getroot()
        doc = retrieve_doc(root)
        if doc != "":
            yield {"docno": filename, "text": doc}


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path)
    parser.add_argument("--index-dir", type=str)
    args = parser.parse_args()

    # Create index directory
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)

    # build index
    iter_indexer = pt.IterDictIndexer(args.index_dir, threads=os.cpu_count())
    doc_iter = doc2dict(args.src_dir)
    indexref = iter_indexer.index(doc_iter)
