# Document Retrieval with Pyserini

## Prerequisite

1. Download dataset.

    Unzip the dataset and place it with the following folder structure.

    ```
    .
    ├── pyserini
    ├── data
    │   ├── doc
    │   ├── test_query
    │   ├── train_query
    │   └── train_ans.csv
    └── ...
    ```

2. Download language model.
    <!-- python3 -m spacy download en_core_web_md -->
    ```
    pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
    ```

## Requirements

* pyserini
* open-JDK
* spacy
* scispacy

## Execute

* Query without Named Entity Recognition

    ```
    bash pyserini/pyserini.sh
    ```

* Query with Named Entity Recognition

    ```
    bash pyserini/pyserini.sh --named-entity
    ```