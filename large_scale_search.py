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
from math import ceil
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--model_data_dir', default='model_data_deu')
parser.add_argument('--data_dir', default='/media/data/datasets/wiener_tesseract_deu')
parser.add_argument('--debug', action='store_true')

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
    # args.data_dir = '/media/data/datasets/wiener_tesseract'
    # args.data_dir = os.path.join('data', 'all_wiener_segmented')
    
    args = parser.parse_args()    

    if not os.path.isfile(os.path.join(args.model_data_dir, 'phoc_candidates.npy')) and not os.path.exists(os.path.join(args.model_data_dir, 'vocabulary.json')):
        print('creating vocabulary, words...')
        # create dictionary of words
        vocabulary, words = load_as_words(args.data_dir)
        vocab_strings = list(vocabulary.keys())

        print('saving all words, data...')
        # save
        with open(os.path.join(args.model_data_dir,'vocabulary.json'), 'w') as f:
            json.dump(vocabulary, f)

        with open(os.path.join(args.model_data_dir, 'words.json'), 'w') as f:
            json.dump(words, f)

        with open(os.path.join(args.model_data_dir, 'vocab_strings.json'), 'w') as f:
            json.dump(vocab_strings, f)
    elif not os.path.isfile(os.path.join(args.model_data_dir, 'phoc_candidates.npy')) and not args.debug:
        print('loading vocabulary, words...')
        with open(os.path.join(args.model_data_dir,'vocabulary.json'), 'r') as f:
            vocabulary = json.load(f)

        with open(os.path.join(args.model_data_dir, 'words.json'), 'r') as f:
            words = json.load(f)

        with open(os.path.join(args.model_data_dir, 'vocab_strings.json'), 'r') as f:
            vocab_strings = json.load(f)

    # create unigrams for all vocabulary
    unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    unigrams += [chr(i) for i in range(ord('0'), ord('9') + 1)]
    unigrams += [chr(i) for i in range(ord('À'), ord('ü'))]
    unigrams = sorted(unigrams)

    with open(os.path.join(args.model_data_dir, 'unigrams.json'), 'w') as f:
        json.dump(unigrams, f)
    
    if not os.path.isfile(os.path.join(args.model_data_dir, 'candidates_all.npy')): 
        # load the PHOC candidates
        if os.path.isfile(os.path.join(args.model_data_dir, 'phoc_candidates.npy')): 
            print('loading phoc candidates')
            load_start = time.clock()

            if args.debug: 
                candidates = np.random.rand(2004, 1440)
            else: 
                candidates = np.load(os.path.join(args.model_data_dir, 'phoc_candidates.npy')) 
        
            load_end = time.clock() 
            print(load_end - load_start) 
        else: 
            # make separate candidates dictionaries
            candidates = build_phoc_descriptor(vocab_strings, phoc_unigrams=unigrams, unigram_levels=[1,2,4,8])
            np.save(os.path.join(args.model_data_dir, 'phoc_candidates.npy'), candidates)

        print('candidates shape:', candidates.shape[0], candidates.shape[1])

        # subtract mean and project candidates using Wy
        print('loading Wy, mean_y...')
        Wy = np.load(os.path.join(args.model_data_dir, 'Wy.npy'))
        mean_y = np.load(os.path.join(args.model_data_dir, 'mean_y.npy'))
 
        p_time = time.clock()    
        print('subtracting mean, projecting...')
        # subtract mean
        candidates = candidates - np.transpose(mean_y).reshape(1, -1)
    
        # project into common subspace 
        candidates = np.matmul(candidates, Wy)
   
        print('normalizing...')

        # normalize candidates using the L2 norm
        cand_norms = np.linalg.norm(candidates, axis=1)
        cand_norms = np.reshape(cand_norms, (-1, 1))
        candidates = candidates/cand_norms
        p_time2 = time.clock()
        print(p_time2 - p_time)

        print('saving candidates...')
        # save candidates
        if not args.debug: 
            np.save(os.path.join(args.model_data_dir, 'candidates_all.npy'), candidates)
    else:
        print('loading projected candidates...')
        start_load = time.clock() 
        candidates = np.load(os.path.join(args.model_data_dir, 'candidates_all.npy'))
        print(time.clock() - start_load)


    # find average distance to 20 nearest neighbors (hub matrix for CSLS)
    print('finding nearest neighbors...')
    num_nn = 20
    block_size = 1000

    if os.path.isfile(os.path.join(args.model_data_dir, 'hub.npy')) 
        print('loading hub matrix...')
        start_load = time.clock() 
        summed = np.load(os.path.join(args.model_data_dir, 'hub.npy'))
        print(time.clock() - start_load)

        last_block = int((np.max(np.nonzero(hub))+1)/block_size)
    else:
        print('creating hub matrix...')
        summed = np.zeros((candidates.shape[0], 1))
        last_block = 0
    
    # try doing this block by block 
    for i in tqdm(range(last_block, ceil(float(candidates.shape[0])/block_size))): 
        start_idx = i*block_size
        if i == ceil(float(candidates.shape[0])/block_size) - 1: 
            end_idx = candidates.shape[1] # don't go beyond our range
        else: 
            end_idx = (i+1)*block_size

        distances = np.matmul(candidates[start_idx: end_idx, :], np.transpose(candidates))

        if i == 0: 
            print('distances shape:', distances.shape[0], distances.shape[1])

        sorted_dist = np.sort(distances, axis=-1)    
        trunc_distances = sorted_dist[:, :num_nn]
        summed[start_idx:end_idx, :] = np.reshape(np.sum(trunc_distances, axis=1)/num_nn, (-1,1))
    
        np.save(os.path.join(args.model_data_dir, 'hub.npy'), summed)
    
    print('hub matrix shape:', summed.shape[0], summed.shape[1])

    # save hub matrix --> in the formula, this is r_20(y)
    np.save(os.path.join(args.model_data_dir, 'hub.npy'), summed)

    
    # save a few number of candidates for later debugging purposes
    np.save(os.path.join(args.model_data_dir, 'candidates_few.npy'), candidates[:1000,:])
    
    # save few number of items in hub_matrix for debugging purposes   
    np.save(os.path.join(args.model_data_dir, 'hub_few.npy'), summed[:1000,:])
    
    # TODO --> need to redefine the run_query() function

    print('saved candidates...')
    """ 
    queries = 'Der Warszawa Pact this is another Hitler pommern'.split()
    tic = time.clock()
    results = run_query(queries, candidates, unigrams)
    toc = time.clock()
    print(toc - tic)
    clean_results = show_clean_results(queries, results, vocab_strings, vocabulary, words)
    """ 
    print('end')
