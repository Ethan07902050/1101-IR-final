import argparse
from pathlib import Path

from pyserini.search import SimpleSearcher
import xml.etree.ElementTree as ET

from utils import write_pred, map_at_k


def retrieve_query(query_path: Path):
    parser = ET.XMLParser(encoding="utf-8")
    root = ET.parse(query_path, parser=parser).getroot()
    paragraphs = ""
    tags = ["summary"]  # TODO: try to include different contents

    for tag in tags:
        target = f".//{tag}"
        paragraphs += root.find(target).text

    return paragraphs


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path)
    args = parser.parse_args()

    data_path = Path("data")
    searcher = SimpleSearcher(str(args.index_dir))

    # query on training set and testing set
    for dataset in ["train", "test"]:
        # query
        pred = {}
        for query_path in (data_path / f"{dataset}_query").iterdir():
            hits = searcher.search(retrieve_query(query_path), k=50)
            pred[query_path.name] = [hit.docid for hit in hits]

        # write pred to csv
        pred_path = data_path / f"{dataset}_pred.csv"
        write_pred(pred_path, pred)

        # compute MAP@50 for training set
        if dataset == "train":
            gt_path = data_path / "train_ans.csv"
            print(f"MAP@50 = {map_at_k(pred_path, gt_path, 50)}")
