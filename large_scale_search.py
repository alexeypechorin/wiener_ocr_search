# coding=utf-8

import os
import numpy as np
from evaluation import relative_to_abs_coord
from tqdm import tqdm
import json
from nltk.tokenize import word_tokenize
import string
from evaluation import build_phoc_descriptor
from scipy.spatial.distance import cdist, pdist, squareform
import time

def load_as_words(data_dir):
    """
    :param data_dir: where files are
    :return: vocabulary: dict in form {'word1': [index1 in words list, index2 in words list, ..., indexn in words list]}
    words: list of dicts in form {'index': index,
                                   'word': word,
                                   'bbox': [xleft, ytop, xright, ybottom],
                                   'conf': confidence,
                                   'doc': doc}
    """

    collections = os.listdir(data_dir)

    collections = [coll for coll in collections if coll not in ['.', '..', '.DS_Store', '.json']]
    collections = collections[:]
    num_words = 0
    vocabulary = {}
    words = []

    word_idx = 0

    with tqdm(total=len(collections)) as pbar:
        for collection in collections:

            directory = os.path.join(data_dir, collection)

            json_files = os.listdir(directory)
            json_files = [json_file for json_file in json_files if json_file.endswith('.json')]

            for json_file in json_files:
                with open(os.path.join(directory, json_file), 'r') as f:
                    contents = json.load(f)

                for pre_word in contents:
                    # add word to list of words
                    img_name = os.path.join(directory, json_file.split('.')[0] + '.jpg')

                    # clean up string for punctuation
                    # search as original word + words made when we split by punctuation

                    current_word = pre_word['word']
                    tokenized = word_tokenize(current_word)

                    # means tokenization did something
                    if len(tokenized) > 1:
                        current_word = [current_word] + word_tokenize(current_word)
                    else:
                        current_word = [current_word]

                    for word in current_word:
                        if word in string.punctuation:
                            continue

                        words += [{'index': word_idx, 'word': word,
                                   'bbox': relative_to_abs_coord(pre_word['bbox']),
                                   'conf': pre_word['conf'],
                                   'img_path': img_name}]

                        # add index of word
                        if word not in vocabulary.keys():
                            vocabulary[word] = [word_idx]
                        else:
                            vocabulary[word] += [word_idx]

                        # increment index
                        word_idx += 1

            pbar.update(1)

    return vocabulary, words

def run_query(queries, candidate_phocs, unigrams, unigram_levels=[1,2,4,8,16]):
    query_phocs = build_phoc_descriptor(queries, unigram_levels=unigram_levels, phoc_unigrams=unigrams)

    dist = cdist(query_phocs, candidate_phocs, 'cosine')
    sorted_results = np.argsort(dist, axis=1)

    return sorted_results

def show_clean_results(queries, results, vocab_strings, vocabulary, words):
    """
    prints out clean table of results
    :param results:
    :param vocabulary:
    :param words:
    :return:
    """
    for row in range(results.shape[0]):
        print(queries[row] + ':')
        print('------------------')
        for res_idx in range(20):
            print(res_idx, vocab_strings[results[row, res_idx]])


if __name__=='__main__':
    # data_dir = '/media/data/datasets/wiener_tesseract'
    data_dir = os.path.join('data', 'all_wiener_segmented')

    if not os.path.exists(os.path.join('model_data', 'vocabulary.json')):
        print('creating vocabulary, words...')
        # create dictionary of words
        vocabulary, words = load_as_words(data_dir)
        vocab_strings = list(vocabulary.keys())

        print('saving all words, data...')
        # save
        with open(os.path.join('model_data','vocabulary.json'), 'w') as f:
            json.dump(vocabulary, f)

        with open(os.path.join('model_data', 'words.json'), 'w') as f:
            json.dump(words, f)

        with open(os.path.join('model_data', 'vocab_strings.json'), 'w') as f:
            json.dump(vocab_strings, f)
    else:
        print('loading vocabulary, words...')
        with open(os.path.join('model_data','vocabulary.json'), 'r') as f:
            vocabulary = json.load(f)

        with open(os.path.join('model_data', 'words.json'), 'r') as f:
            words = json.load(f)

        with open(os.path.join('model_data', 'vocab_strings.json'), 'r') as f:
            vocab_strings = json.load(f)

    # create unigrams for all vocabulary
    unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    unigrams += [chr(i) for i in range(ord('0'), ord('9') + 1)]
    unigrams += [chr(i) for i in range(ord('À'), ord('ü'))]
    unigrams = sorted(unigrams)

    with open(os.path.join('model_data', 'unigrams.json'), 'w') as f:
        json.dump(unigrams, f)

    # make separate candidates dictionaries
    candidates = build_phoc_descriptor(vocab_strings, phoc_unigrams=unigrams, unigram_levels=[1,2,4,8,16])

    print('saving candidates...')
    # save candidates
    np.save(os.path.join('model_data', 'candidates_all.npy'), candidates)
    print('saved candidates...')

    queries = 'Der Warszawa Pact this is another Hitler pommern'.split()
    tic = time.clock()
    results = run_query(queries, candidates, unigrams)
    toc = time.clock()
    print(toc - tic)
    clean_results = show_clean_results(queries, results, vocab_strings, vocabulary, words)

    print('end')
