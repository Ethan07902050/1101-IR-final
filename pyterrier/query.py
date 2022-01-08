import os
from pathlib import Path
import argparse

import ssl
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET
import xml.dom.minidom
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
import pyterrier as pt

import utils

if not pt.started():
    ssl._create_default_https_context = ssl._create_unverified_context
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])


def save_result(pred_df, split):
    data_path = Path("data")
    pred_dict = {}
    qids = pred_df["qid"].unique()
    for qid in qids:
        top_docs = pred_df.loc[(pred_df["qid"] == qid) & (pred_df["rank"] < 50)]
        pred_dict[qid] = top_docs["docno"].tolist()

    pred_path = data_path / f"{split}_pred.csv"
    utils.write_pred(pred_path, pred_dict)

    if split == "train":
        gt_path = data_path / "train_ans.csv"
        print(f"MAP@50 = {utils.map_at_k(pred_path, gt_path, 50)}")


def get_named_entity(contents, model_name):
    nlp = spacy.load(model_name)
    nlp.add_pipe("abbreviation_detector")
    doc = nlp(contents)
    abbrev = {abrv: abrv._.long_form for abrv in doc._.abbreviations}

    named_entities = []
    for ent in doc.ents:
        named_entities.append(abbrev.get(ent.text) or ent.text)
    return named_entities


def retrieve_query(root):
    paragraphs = []
    tags = ["summary"]
    filters = [strip_punctuation]

    # parse query
    for tag in tags:
        target = root.find(f".//{tag}")
        text = ET.tostring(target, method="text").decode("utf-8")
        words = preprocess_string(text, filters)
        paragraphs += words

    # get named entities
    entities = get_named_entity(" ".join(paragraphs), "en_core_sci_sm")
    entity_str = " ".join([e.lower() for e in entities])
    return entity_str


def query2df(dirname: Path):
    count = len([file for file in dirname.iterdir()])

    query_dict = {"qid": [], "query": []}
    for path in tqdm(dirname.iterdir(), total=count):
        filename = path.stem
        parser = ET.XMLParser(encoding="utf-8")
        root = ET.parse(path, parser=parser).getroot()
        doc = retrieve_query(root)

        query_dict["qid"].append(filename)
        query_dict["query"].append(doc)

    query_df = pd.DataFrame(query_dict)
    return query_df


def retrieve(query_df, index_dir):
    index_path = os.path.join(index_dir, "data.properties")
    index = pt.IndexFactory.of(index_path)

    # query expansion
    bo1 = pt.rewrite.Bo1QueryExpansion(index, fb_terms=5, fb_docs=10)
    kl = pt.rewrite.KLQueryExpansion(index, fb_terms=10, fb_docs=10)
    rm3 = pt.rewrite.RM3(index, fb_terms=5, fb_docs=3)

    # retrieval
    dph = pt.BatchRetrieve(index, wmodel="DPH")
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")

    pipeline = dph >> bo1 >> dph
    pred_df = pipeline.transform(query_df)
    return pred_df


def main(index_dir: Path):
    data_path = Path("data")
    for split in ["train", "test"]:
        query_path = data_path / f"{split}_query"
        query_df = query2df(query_path)
        pred_df = retrieve(query_df, index_dir)
        save_result(pred_df, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=str)
    args = parser.parse_args()

    main(args.index_dir)
