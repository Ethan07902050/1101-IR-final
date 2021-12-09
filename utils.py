def write_file(ans, path):
    with open(path, 'w') as f:
        f.write('topic,doc\n')
        for query, docs in zip(ans['topic'], ans['doc']):
            f.write(f'{query},{docs}\n')


def read_file(path):  
    data_dict = {}  

    with open(path, 'r') as f:
        first_line = 1
        for line in f:
            if first_line:
                first_line = 0
            else:
                query, docs = line.strip('\n').split(',')
                data_dict[query] = docs.split(' ')

    return data_dict


def map_score(target_path, pred_path):
    target = read_file(target_path)
    pred = read_file(pred_path)
    precision = []

    for query, docs in pred.items():
        target_set = set(target[query])
        pred_set = set(docs)
        p = len(target_set.intersection(pred_set)) / len(pred_set)
        precision.append(p)

    print(f'map@50: {sum(precision) / len(precision)}')