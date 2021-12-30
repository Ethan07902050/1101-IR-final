from rank_bm25 import BM25Okapi
from tqdm import tqdm


class DocRetrieval:
    def __init__(self, path):
        self.path = path
        self.line_count = 0
        self.docno = []
        with open(path, 'r') as f:
            for line in f:
                self.line_count += 1
                self.docno.append(line.split('\t')[0])
    
    def __len__(self):
        return self.line_count
    
    def __iter__(self):
        with open(self.path, 'r') as f:
            for line in f:
                docno, text = line.strip('\n').split('\t')
                yield text.split()


def write_file(ans, path):
    with open(path, 'w') as f:
        f.write('topic,doc\n')
        for topic, docs in ans.items():
            f.write(f'{topic},{" ".join(docs)}\n')


doc = DocRetrieval('../docStem.txt')
query = DocRetrieval('../testQueryStem.txt')
bm25 = BM25Okapi(doc)
ans_dict = {}
for i, q in tqdm(enumerate(query), total=15):
    top_doc = sorted(enumerate(bm25.get_scores(q)), key=lambda x: -x[1])[:50]
    query_id = query.docno[i]
    doc_ids = [doc.docno[d[0]] for d in top_doc]
    ans_dict[query_id] = doc_ids

utils.write_file(ans_dict, 'bm25_stem.csv')