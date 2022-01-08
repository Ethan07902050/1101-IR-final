src_dir="./data/doc"
index_dir="./data/pyterrier_index"

echo "src dir:" $src_dir
echo "index dir:" $index_dir

# build index
echo "Building index..."
python3 pyterrier/build_index.py \
  --src-dir $src_dir \
  --index-dir $index_dir

# query
echo "Querying..."
python3 pyterrier/query.py \
  --index-dir $index_dir