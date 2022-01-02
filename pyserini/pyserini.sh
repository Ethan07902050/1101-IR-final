#!/bin/bash

# https://github.com/castorini/pyserini/#how-do-i-index-and-search-my-own-documents

src_path="./data/doc"
if [ $# -eq 1 ]
then
    doc_json_path="./data/ner_doc_json"
    index_path="./data/ner_indexes/"
else
    doc_json_path="./data/doc_json"
    index_path="./data/indexes/"
fi
echo "src path" $src_path
echo "doc-json path: " $doc_json_path
echo "index path: " $index_path

# construct collection_json
echo "Building collection json..."
python3 pyserini/build_collection_json.py       \
--src-dir $src_path                             \
--dest-dir $doc_json_path                       \
"$@"

# invoke indexer
echo "Building index..."
python3 -m pyserini.index                       \
--input $doc_json_path                          \
--collection JsonCollection                     \
--generator DefaultLuceneDocumentGenerator      \
--index $index_path                             \
--threads 1                                     \
--storePositions --storeDocvectors --storeRaw

# query
echo "Querying..."
python3 pyserini/query.py                       \
--index-dir $index_path