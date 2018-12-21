# script to search through text for wiener file
# for each query, it returns {query='query', results=[{word, bbox, document},...,{word, bbox, document}]}
# can load queries from a txt file

import argparse
import os
import json
from utils import load_wiener_collections
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--wiener_data_dir', default=os.path.join('data', 'all_wiener_segmented'),
                    help="Directory containing wiener collections")
parser.add_argument('--query_path', default='queries.txt',
                    help="path to txt file containing queries")

def run_single_query(query, all_docs):
    results = []
    with tqdm(total=len(all_docs)) as pbar:
        for document in all_docs:
            doc_name = document['doc']
            contents = document['contents']

            for candidate in contents:
                word = candidate['word']
                bbox = candidate['bbox']

                if word.lower() == query.lower():
                    results += [{'word':word, 'bbox':bbox, 'doc': doc_name}]
            pbar.update(1)

    return {'query':query, 'results':results}

def run_search(params):
    # all_docs = load_wiener_collections(params.wiener_data_dir, no_bboxes=False)
    all_docs = load_wiener_collections(params.wiener_data_dir, no_bboxes=True)

    # load queries
    with open(params.query_path) as f:
        queries = f.read().split()

    results = []

    for query in queries:
        single_query_result = run_single_query(query, all_docs)
        results += [single_query_result]

    return results

if __name__ == "__main__":
    params = parser.parse_args()

    results = run_search(params)

    results_name = params.query_path.split('.')[0] + '_results.json'

    with open(results_name, 'w') as f:
        json.dump(results, f)
