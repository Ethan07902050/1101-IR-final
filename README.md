# Document Retrieval with Pyserini

## Prerequisite

1. Download dataset.

    Unzip the dataset and place it with the following folder structure.

    ```
    .
    ├── data
    │   ├── doc
    │   ├── test_query
    │   ├── train_query
    │   └── train_ans.csv
    ├── pyterrier
    └── ...
    ```

2. Download language model.
    ```
    pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
    ```

## Requirements

### Environment

* UNIX-like environment
* Python 3.7+

### Packages
* python-pyterrier
* spacy
* scispacy
* gensim
* open-JDK11

## Execute

```
bash pyterrier/pyterrier.sh
```

The prediction results will be stored in `./data/train_pred.csv` and `./data/test_pred.csv`.