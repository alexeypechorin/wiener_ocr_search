import os
import time
import numpy as np
from evaluation import relative_to_abs_coord
from tqdm import tqdm
import json
from nltk.tokenize import word_tokenize
import string
from evaluation import build_phoc_descriptor
from scipy.spatial.distance import cdist, pdist, squareform
import argparse
import cv2 
from PIL import Image
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--query_file', default=None,
                    help='None if we are using user input. If specified, it will open the file and run all queries and save images')
parser.add_argument('--wiener_cropped_dir', default=['/media/data/datasets/wiener_cropped/wiener_images_cropped',
                                                     '/media/data/datasets/wiener1_cropped'], 
                    help='Directories containing the cropped wiener images')
parser.add_argument('--query_is_done', action='store_true', help='Set if we just want to show images')


def run_query(queries, candidates, Wx, mean_x, hub_matrix, unigrams, unigram_levels=[1,2,4,8], num_nn=20):
        # assumes that candidates are already projected and normalized 
        tic = time.process_time()
        query_phocs = build_phoc_descriptor(queries, unigram_levels=unigram_levels, phoc_unigrams=unigrams)
        toc = time.process_time()

        print('building phoc descriptor time:', toc - tic)

        tic = time.process_time()
        # project and normalize
        projected_query = np.matmul(query_phocs - np.transpose(mean_x).reshape(1, -1), Wx)
        query_norms = np.linalg.norm(projected_query, axis=1)
        query_norms = np.reshape(query_norms, (-1, 1))
        projected_query = projected_query/query_norms
        toc = time.process_time()

        print('projecting time:', toc - tic)
        
        tic = time.process_time()
        # find distance from projected query to candidates using cosine distance
        dist = 1 - np.matmul(projected_query, np.transpose(candidates))
        toc = time.process_time()

        print('main distance time:', toc - tic)

        tic = time.process_time()
        # find avg distance to nearest neighbors for query
        sorted_distances = np.sort(dist, axis=-1)
        trunc_distances = sorted_distances[:, :num_nn]
        avg_nn_dist = np.sum(trunc_distances, axis=1)/num_nn
        toc = time.process_time()

        print('nearest neighbor time:', toc - tic)

        tic = time.process_time()
        dist = 2*dist - avg_nn_dist - np.transpose(hub_matrix) 
        toc = time.process_time()

        print('final distance calculation:', toc - tic)
 
        tic = time.process_time()
        # sort the final results
        sorted_results = np.argsort(dist, axis=1)
        toc = time.process_time()

        print('sorting time:', toc - tic)

        return sorted_results

def show_bboxes(img, box):
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), thickness=2)
        return img 

def show_clean_results(queries, results, vocab_strings, vocabulary, words, wiener_dir, save_dir='results'):
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
                        query_path = os.path.join(save_dir, queries[row])
                        print('saving resulting images to', query_path) 

                        # make folder if it doesn't exist 
                        if not os.path.isdir(query_path): 
                                os.makedirs(query_path)

                        word_str = vocab_strings[results[row, res_idx]]
                        print(res_idx, word_str)

                        # vocabulary is dictionary containing list of indices to 'words'
                        word_indices = vocabulary[word_str]
                        
                        max_words = 10
                        max_idx = min(max_words, len(word_indices))
                        # save image    
                        for word_idx in word_indices[:max_words]: 
                                sub_path = '/'.join(words[word_idx]['img_path'].split('/')[-2:])
                                img_path = os.path.join(wiener_dir[0], sub_path)
                                
                                # we have two directories where images are stored
                                # so if we don't get it right the first time 
                                # we know it's in the other one 
                                if not os.path.isfile(img_path): 
                                    img_path = os.path.join(wiener_dir[1], sub_path)

                                img_name = sub_path.split('/')[-1]
                                box = words[word_idx]['bbox']                           
                                

                                # read image
                                img = cv2.imread(img_path)

                                # draw bbox 
                                img = show_bboxes(img, box)
                                        
                                # save image 
                                cv2.imwrite(os.path.join(query_path, str(res_idx) + '_' + str(word_idx) + '_' + img_name), img)


def main_search_user_input(candidates_file_path, Wx_file_path, mean_x_file_path, hub_matrix_file_path,
                               vocabulary_file_path, words_file_path, vocab_strings_file_path, unigrams_file_path,
                               query_results_directory_path, query_is_done, query_file, wiener_cropped_dir):
        if not query_is_done:
                print('loading all candidates')
                tic_cands = time.process_time()
                candidates = np.load(candidates_file_path)
                toc_cands = time.process_time()
                print(toc_cands - tic_cands, 'seconds...')

                print('loading Wx, mean_x, and hub_matrix...')
                Wx = np.load(Wx_file_path)
                mean_x = np.load(mean_x_file_path)
                hub_matrix = np.load(hub_matrix_file_path)
        print('loading vocabulary, words...')
        tic_vocab = time.process_time()
        with open(vocabulary_file_path, 'r') as f:
                vocabulary = json.load(f)
        with open(words_file_path, 'r') as f:
                words = json.load(f)
        with open(vocab_strings_file_path, 'r') as f:
                vocab_strings = json.load(f)
        with open(unigrams_file_path, 'r') as f:
                unigrams = json.load(f)
        toc_vocab = time.process_time()
        print(toc_vocab - tic_vocab, 'seconds...')
        if query_file is not None:
                with open(query_file, 'r') as f:
                        queries = f.read().split()

                print('running queries from file')
                tic = time.process_time()

                if not query_is_done:
                        # run the actual queries
                        result = run_query(queries, candidates, Wx, mean_x, hub_matrix, unigrams)
                        toc = time.process_time()
                        print(toc - tic, 'seconds...')

                        # save a small sample of results
                        short_result = result[:, 0:1000]
                        np.save('search_results.npy', short_result)
                else:
                        result = np.load('search_results.npy')

                        # print out text results for each query and save images
                show_clean_results(queries, result, vocab_strings, vocabulary, words,
                                   wiener_dir=wiener_cropped_dir)
        else:
                query_string = input('word to search for: ')

                while query_string != 'q':
                        tic = time.process_time()
                        result = run_query([query_string], candidates, Wx, mean_x, hub_matrix, unigrams)
                        toc = time.process_time()
                        print(toc - tic, 'seconds...')

                        # print text results + save images
                        show_clean_results([query_string], result, vocab_strings, vocabulary, words,
                                           wiener_dir=wiener_cropped_dir,
                                           save_dir=query_results_directory_path)

                        # ask for user input again
                        query_string = input('word to search for: ')


if __name__=='__main__':
        args = parser.parse_args()
        model_data_folder = 'model_data_deu'
        candidates_file_path = os.path.join(model_data_folder, 'candidates_all.npy')
        Wx_file_path = os.path.join(model_data_folder, 'Wx.npy')
        mean_x_file_path = os.path.join(model_data_folder, 'mean_x.npy')
        hub_matrix_file_path = os.path.join(model_data_folder, 'hub.npy')
        vocabulary_file_path = os.path.join(model_data_folder, 'vocabulary.json')
        words_file_path = os.path.join(model_data_folder, 'words.json')
        vocab_strings_file_path = os.path.join(model_data_folder, 'vocab_strings.json')
        unigrams_file_path = os.path.join(model_data_folder, 'unigrams.json')
        query_results_directory_path = 'results_user_input'

        main_search_user_input(candidates_file_path, Wx_file_path, mean_x_file_path, hub_matrix_file_path,
                               vocabulary_file_path, words_file_path, vocab_strings_file_path, unigrams_file_path,
                               query_results_directory_path, args.query_is_done, args.query_file, args.wiener_cropped_dir)
