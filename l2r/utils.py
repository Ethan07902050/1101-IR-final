from pathlib import Path
import csv
from typing import Dict, List

import ml_metrics as metrics

HEADER = ["topic", "doc"]


def read_pred(pred_path: Path) -> Dict[str, List[str]]:
    pred = {}
    with open(pred_path, "r") as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=HEADER)
        for row in reader:
            query_id = row[HEADER[0]]
            doc_ids = row[HEADER[1]].split(" ")
            pred[query_id] = doc_ids
    return pred


def write_pred(pred_path: Path, pred: Dict[str, List[str]]):
    with open(pred_path, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HEADER)
        writer.writeheader()
        for query_id, doc_ids in pred.items():
            writer.writerow({HEADER[0]: query_id, HEADER[1]: " ".join(doc_ids)})


def map_at_k(pred_path: Path, gt_path: Path, k: int):
    pred = read_pred(pred_path)
    gt = read_pred(gt_path)
    assert set(pred.keys()) == set(gt.keys())

    preds = []
    gts = []
    for query_id in pred.keys():
        preds.append(pred[query_id])
        gts.append(gt[query_id])
    return metrics.mapk(actual=gts, predicted=preds, k=k)