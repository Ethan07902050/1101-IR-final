#!/bin/bash

# https://github.com/castorini/pyserini/#how-do-i-index-and-search-my-own-documents

# construct collection_json
if [ ! -d "../data/doc_json" ] 
then
    echo "Build collection json..."
    python3 ./build_collection_json.py
fi

# invoke indexer
python3 -m pyserini.index \
--input ../data/doc_json \
--collection JsonCollection \
--generator DefaultLuceneDocumentGenerator \
--index indexes/ \
--threads 1 \
--storePositions --storeDocvectors --storeRaw

# search