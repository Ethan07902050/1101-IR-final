import argparse
from pathlib import Path

from pyserini.search import SimpleSearcher
import xml.etree.ElementTree as ET
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

from utils import write_pred, map_at_k

# Reference: https://github.com/allenai/scispacy


class QueryParser:
    def __init__(self, nlp="en_core_sci_sm"):
        self.nlp = spacy.load(nlp)
        self.nlp.add_pipe("abbreviation_detector")
        # self.nlp.add_pipe(
        #     "scispacy_linker",
        #     config={"resolve_abbreviations": True, "linker_name": "umls"},
        # )

    def retrieve_query(self, query_path: Path, named_entity=False):
        parser = ET.XMLParser(encoding="utf-8")
        root = ET.parse(query_path, parser=parser).getroot()
        paragraphs = ""
        tags = ["summary"]

        for tag in tags:
            target = f".//{tag}"
            paragraphs += root.find(target).text

        if named_entity:
            named_entities = self._get_named_entity(paragraphs)
            return " ".join(named_entities)

        return paragraphs

    def _get_named_entity(self, contents: str):
        doc = self.nlp(contents)
        abbrev = {abrv: abrv._.long_form for abrv in doc._.abbreviations}
        print(abbrev)
        named_entities = []
        for ent in doc.ents:
            named_entities.append(abbrev.get(ent.text) or ent.text)
        return named_entities


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--named-entity", action="store_true")
    parser.add_argument("--index-dir", type=Path)
    args = parser.parse_args()

    data_path = Path("data")
    searcher = SimpleSearcher(str(args.index_dir))
    query_parser = QueryParser()

    # query on training set and testing set
    for dataset in ["train", "test"]:
        # query
        pred = {}
        for query_path in (data_path / f"{dataset}_query").iterdir():
            hits = searcher.search(
                query_parser.retrieve_query(query_path, args.named_entity), k=50
            )
            pred[query_path.name] = [hit.docid for hit in hits]

        # write pred to csv
        pred_path = data_path / f"{dataset}_pred.csv"
        write_pred(pred_path, pred)

        # compute MAP@50 for training set
        if dataset == "train":
            gt_path = data_path / "train_ans.csv"
            print(f"MAP@50 = {map_at_k(pred_path, gt_path, 50)}")
