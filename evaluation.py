import os
import json
import codecs
import cv2
from tesseracthelpers import show_bboxes_with_text
from gensim.models import FastText
from ocr_correction import correct
from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np
import tqdm

def iou(boxA, boxB):
    """
    coordinates are in [x_left, y_top, x_right, y_bottom] form
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def num_matches_phoc(gt_boxes, predicted_bboxes, iou_threshold_bbox=0.2):
    gt_abs = []
    pred_abs = []

    for box in gt_boxes:
        bbox = box['bbox']

        gt_abs += [relative_to_abs_coord(bbox)]

    for box in predicted_bboxes:
        bbox = box['bbox']
        pred_abs += [relative_to_abs_coord(bbox)]

    # list of gt words
    query_strings = []

    # index of correct label
    query_label_idx = []

    # the string representation of the candidate
    candidate_strings = []

    label_idx = 0
    # get labels + transcriptions
    for i, gt_box in enumerate(gt_abs):
        count_match = 0
        for j, pred_box in enumerate(pred_abs):
            iou_amt = iou(gt_box, pred_box)

            if iou_amt > iou_threshold_bbox:
                query_strings += [gt_boxes[i]['word']]
                query_label_idx += [label_idx]
                candidate_strings += [predicted_bboxes[j]['word'].lower()]
                label_idx += 1

    # add more noisy ocr text to candidate
    with open('long_string.txt','r') as f:
        long_string = f.read()

    long_string = long_string.split()
    candidate_strings += [word.lower() for word in long_string[:100000]]

    unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    unigrams += [chr(i) for i in range(ord('0'), ord('9') + 1)]
    unigram_levels = [1,2,3,4,5,8,16]
    # make phocs of both
    query_phocs = build_phoc_descriptor(query_strings, phoc_unigrams=unigrams, unigram_levels=unigram_levels)
    candidate_phocs = build_phoc_descriptor(candidate_strings, phoc_unigrams=unigrams, unigram_levels=unigram_levels)

    dist = cdist(query_phocs, candidate_phocs, 'cosine')
    sorted_results = np.argsort(dist, axis=1)

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    for query_idx in range(sorted_results.shape[0]):
        text_results = [candidate_strings[sorted_results[query_idx,id]] for id in range(10)]
        if query_idx == sorted_results[query_idx, 0] or query_strings[query_idx] == text_results[0]:
            recall_1 += 1
        if query_idx in sorted_results[query_idx, 0:4] or query_strings[query_idx] in text_results[:4]:
            recall_5 += 1
        if query_idx in sorted_results[query_idx, 0:9] or query_strings[query_idx] in text_results[:9]:
            recall_10 += 1

    return {"1-num_gt_bboxes": len(gt_abs),
            "2-recall_1": recall_1/float(sorted_results.shape[0]),
            "3-recall_5": recall_5/float(sorted_results.shape[0]),
            "4-recall_10": recall_10/float(sorted_results.shape[0])}

def build_phoc_descriptor(words, phoc_unigrams, unigram_levels,  #pylint: disable=too-many-arguments, too-many-branches, too-many-locals
                          bigram_levels=None, phoc_bigrams=None,
                          split_character=None, on_unknown_unigram='nothing',
                          phoc_type='phoc'):
    '''
    Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).

    Args:
        word (str): word to calculate descriptor for
        phoc_unigrams (str): string of all unigrams to use in the PHOC
        unigram_levels (list of int): the levels to use in the PHOC
        split_character (str): special character to split the word strings into characters
        on_unknown_unigram (str): What to do if a unigram appearing in a word
            is not among the supplied phoc_unigrams. Possible: 'warn', 'error', 'nothing'
        phoc_type (str): the type of the PHOC to be build. The default is the
            binary PHOC (standard version from Almazan 2014).
            Possible: phoc, spoc
    Returns:
        the PHOC for the given word
    '''
    # prepare output matrix
    if on_unknown_unigram not in ['error', 'warn','nothing']:
        raise ValueError('I don\'t know the on_unknown_unigram parameter \'%s\'' % on_unknown_unigram)
    phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)
    if phoc_bigrams is not None:
        phoc_size += len(phoc_bigrams) * np.sum(bigram_levels)
    phocs = np.zeros((len(words), phoc_size))
    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k + 1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]

    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(phoc_unigrams)}

    # iterate through all the words
    for word_index, word in enumerate(tqdm.tqdm(words)):
        if split_character is not None:
            word = word.split(split_character)

        n = len(word) #pylint: disable=invalid-name
        for index, char in enumerate(word):
            char_occ = occupancy(index, n)
            if char not in char_indices:
                if on_unknown_unigram == 'warn':
                    logger.warn('The unigram \'%s\' is unknown, skipping this character', char)
                    continue
                elif on_unknown_unigram == 'error':
                    logger.fatal('The unigram \'%s\' is unknown', char)
                    raise ValueError()
                else:
                    continue
            char_index = char_indices[char]
            for level in unigram_levels:
                for region in range(level):
                    region_occ = occupancy(region, level)
                    if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                        feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(phoc_unigrams) + region * len(phoc_unigrams) + char_index
                        if phoc_type == 'phoc':
                            phocs[word_index, feat_vec_index] = 1
                        elif phoc_type == 'spoc':
                            phocs[word_index, feat_vec_index] += 1
                        else:
                            raise ValueError('The phoc_type \'%s\' is unknown' % phoc_type)
        # add bigrams
        if phoc_bigrams is not None:
            ngram_features = np.zeros(len(phoc_bigrams) * np.sum(bigram_levels))
            ngram_occupancy = lambda k, n: [float(k) / n, float(k + 2) / n]
            for i in range(n - 1):
                ngram = word[i:i + 2]
                if phoc_bigrams.get(ngram, 0) == 0:
                    continue
                occ = ngram_occupancy(i, n)
                for level in bigram_levels:
                    for region in range(level):
                        region_occ = occupancy(region, level)
                        overlap_size = size(overlap(occ, region_occ)) / size(occ)
                        if overlap_size >= 0.5:
                            if phoc_type == 'phoc':
                                ngram_features[region * len(phoc_bigrams) + phoc_bigrams[ngram]] = 1
                            elif phoc_type == 'spoc':
                                ngram_features[region * len(phoc_bigrams) + phoc_bigrams[ngram]] += 1
                            else:
                                raise ValueError('The phoc_type \'%s\' is unknown' % phoc_type)
            phocs[word_index, -ngram_features.shape[0]:] = ngram_features
    return phocs

def num_matches(gt_boxes, predicted_bboxes, sym_spell, iou_threshold_bbox=0.8, iou_threshold_spelling=0.8, max_edit_distance=0):
    # convert from relative to absolute boxes
    gt_abs = []
    pred_abs = []

    for box in gt_boxes:
        bbox = box['bbox']

        gt_abs += [relative_to_abs_coord(bbox)]

    for box in predicted_bboxes:
        bbox = box['bbox']
        pred_abs += [relative_to_abs_coord(bbox)]

    # if IoU > iou_threshold, then we count it as a match
    total_bbox_match = 0
    total_spelling_match = 0
    match_idx = []
    recall_at_200 = 0
    mrr = 0
    for i, gt_box in enumerate(gt_abs):
        count_match = 0
        for j, pred_box in enumerate(pred_abs):
            iou_amt = iou(gt_box, pred_box)

            if iou_amt > iou_threshold_bbox:
                total_bbox_match += 1
                match_idx += [j]

            if iou_amt > iou_threshold_spelling:
                # if edit_distance(gt_boxes[i]['word'],predicted_bboxes[j]['word'].lower(), max_edit_distance):
                # try correcting
                suggestions = correct(predicted_bboxes[j]['word'].lower(), sym_spell=sym_spell)
                suggestions = [suggestion.term for suggestion in suggestions]

                gt_word = gt_boxes[i]['word']
                if gt_word in suggestions:
                    rank = suggestions.index(gt_word) + 1
                    mrr += 1/float(rank)

                    if rank == 1:
                        total_spelling_match += 1

                    recall_at_200 += 1 # += rank <= 200


    corr_matches = [predicted_bboxes[idx] for idx in match_idx]

    # return precision, recall for bboxes, spelling, number of predictions, number of correct predictions

    total_predictions = len(pred_abs)
    correct_predictions = total_bbox_match

    precision_bbox = float(total_bbox_match)/len(pred_abs)
    recall_bbox = float(total_bbox_match)/len(gt_abs)

    precision_spelling = float(total_spelling_match)/len(pred_abs)
    recall_spelling = float(total_spelling_match)/len(gt_abs)

    mrr /= len(gt_abs)
    recall_at_200 /= float(len(gt_abs))

    return {"1-num_gt_bboxes": len(gt_abs),
            "3-bbox_matches": correct_predictions,
            "2-num_bbox_preds": total_predictions,
            "4-precision_bbox": precision_bbox,
            "5-recall_bbox": recall_bbox,
            "6-precision_spelling": precision_spelling,
            "7-recall_spelling": recall_spelling,
            "8-MRR": mrr,
            "9-recall_at_200": recall_at_200}

def num_matches_fasttext(gt_boxes, predicted_bboxes, model, iou_threshold_bbox=0.1):
    """
    :param gt_boxes: ground truth boxes
    :param predicted_bboxes: get predicted boxes
    :param model: model that we're using to make predictions
    :param iou_threshold_bbox: say something is a match if there is iou > iou_threshold_box
    :return:

    return MRR,
    return recall@1, recall@2, recall@10
    """
    gt_abs = []
    pred_abs = []

    for box in gt_boxes:
        bbox = box['bbox']

        gt_abs += [relative_to_abs_coord(bbox)]

    for box in predicted_bboxes:
        bbox = box['bbox']
        pred_abs += [relative_to_abs_coord(bbox)]

    mrr = 0
    recall_1 = 0
    recall_3 = 0
    recall_10 = 0
    number_of_oov_words = 0
    for i, gt_box in enumerate(gt_abs):
        for j, pred_box in enumerate(pred_abs):
            iou_amt = iou(gt_box, pred_box)
            if iou_amt > iou_threshold_bbox:
                if predicted_bboxes[j]['word'] not in model.wv.vocab:
                    number_of_oov_words = 0
                # predicted word is spelled correctly
                if gt_boxes[i]['word'] == predicted_bboxes[j]['word'].lower():
                    mrr += 1
                    recall_1 += 1
                    recall_3 += 1
                    recall_10 += 1
                else:
                    gt_word = gt_boxes[i]['word']
                    predicted_word = predicted_bboxes[j]['word']
                    most_similar = model.wv.most_similar(predicted_bboxes[j]['word'])
                    most_similar = [word for (word, vec) in most_similar]
                    if gt_word not in most_similar:
                        continue
                    else:
                        rank = most_similar.index(gt_word) + 1
                        mrr += 1/float(rank)
                        recall_1 += rank <= 1
                        recall_3 += rank <= 3
                        recall_10 += rank <= 10

    return {'num_oov': number_of_oov_words,
            'MRR': float(mrr)/len(gt_abs),
            'recall_1': float(recall_1)/len(gt_abs),
            'recall_3': float(recall_3)/len(gt_abs),
            'recall_10': float(recall_10)/len(gt_abs)}

def edit_distance(word1, word2, distance=0):
    # if words word1, word2 are within edit distance specified by distance, return True
    return True if levenshteinDistance(word1, word2) <= distance else False

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_gt_boxes(json_file_name, img_name):
    """
    :param json_file_name:
    :return:
    get zahi's transcriptions into list format

    every item is a dictionary with "bbox", "conf", and "word"

    we get how the image was cropped from the last 4 numbers of the image name
    separated by '_' and in [x, y, w, h]
    """
    img = cv2.imread(img_name)

    with codecs.open(json_file_name, 'r', 'utf-8-sig') as f:
        json_data = json.load(f)

    boxes = []

    img_name = img_name.split('.')[0]
    img_name = img_name.split('/')[-1]
    crop_coords = img_name.split('_')[1:]
    crop_coords = [int(crop_coord) for crop_coord in crop_coords]

    # all_words
    for obj in json_data["annotation"]["object"]:
        word = obj["attributes"]

        raw_bbox = obj["polygon"]["pt"]

        left = 1000000000
        top = 100000000
        w = 0
        h = 0

        for pt in raw_bbox:
            left = min(left, int(pt['x']))
            top = min(top, int(pt['y']))
            w = max(w, int(pt['x']) - left)
            h = max(h, int(pt['y']) - top)


        # use crop_coords to get proper output of left, top, w, h
        # only left and top change

        boxes += [{"bbox": [left - crop_coords[0], top - crop_coords[1], w, h],
                   "conf": "100",
                   "word": word}]

    # show_bboxes_with_text(img, boxes, sym_spell=None)

    return boxes

def relative_to_abs_coord(coordinates):
    """
    given the coordinates in [left, top, width, height] form,
    convert it to [x_left, y_top, x_right, y_bottom] form
    """

    abs_coordinates = [coordinates[0],
                       coordinates[1],
                       coordinates[0] + coordinates[2],
                       coordinates[1] + coordinates[3]]

    return abs_coordinates

if __name__ == "__main__":
    print('shalom')
    # get_gt_boxes(os.path.join('data', 'wiener_zahi_clean', '00010072.json'), os.path.join('data', 'wiener_zahi_clean', '00010072_731_445_2255_3159.jpg'))