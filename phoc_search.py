import evaluation as eval
import numpy as np
import tqdm
from tesseracthelpers import get_data, process_tessaract_data, show_bboxes_with_text
import argparse
from ocr_correction import load_symspell
import os
import cv2
import utils
import json


parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")

def eval_phoc_search(params):
    #  = FastText.load('fasttext.model')
    overall_statistics = {}
    ocr_data = os.listdir(params.data_dir)
    ocr_images = [ocr_image for ocr_image in ocr_data if ocr_image.endswith('.jpg')]
    ocr_json = [ocr_json for ocr_json in ocr_data if ocr_json.endswith('.json')]
    # sym_spell = load_symspell(os.path.join('data', 'de_50k.txt'))
    sym_spell = None
    for i, name in enumerate(ocr_images):
        print('---' + name + '---')
        # get reading from tesseract
        img_name = os.path.join(params.data_dir, name)
        img, result = get_data(img_name)
        labels, words = process_tessaract_data(result)
        img = show_bboxes_with_text(img, words, sym_spell=sym_spell)

        img.save(os.path.join('results','ocr', name))

        # get name of json file name
        json_file_name = [ocr_json_name for ocr_json_name in ocr_json if ocr_json_name.startswith(name[:8])][0]

        json_name = os.path.join(params.data_dir, json_file_name)

        # get ground truth data
        gt_words = eval.get_gt_boxes(json_name, img_name)

        img = cv2.imread(img_name)
        # show_bboxes_with_text(img, gt_words, sym_spell=sym_spell)

        # get stats
        print('EDIT DISTANCE: ' + str(params.max_edit_distance))
        statistics = eval.num_matches_phoc(gt_words, words)
        print(statistics)
        # statistics_fasttext = eval.num_matches_fasttext(gt_words, words, iou_threshold_bbox=params.iou_threshold_bbox, model=model)

        # with open(os.path.join(params.model_dir, 'stats_' + name[:8] + '.json'), 'w') as f:
        #    json.dump(statistics, f)

        for key in statistics.keys():
            if key not in overall_statistics.keys():
                overall_statistics[key] = statistics[key]
            else:
                overall_statistics[key] += statistics[key]


    # average out certain statistics
    stats_to_average = ["4-precision_bbox", "5-recall_bbox", "6-precision_spelling", "7-recall_spelling"]

    num_docs = len(ocr_images)

    for stat in stats_to_average:
        overall_statistics[stat] /= num_docs

    print('hello')

    # with open(os.path.join(params.model_dir, 'stats_overall.json'), 'w') as f:
         #json.dump(overall_statistics, f)

if __name__=='__main__':
    # get gt
    # get predicted bboxes

    # produce PHOCS for all of them
    # for each ground truth box, we see if 1) there is a bounding box that matches it (if IOU > 0.2)
    # and find the transcription of that bounding box
    # give that transcription a special index (same words will have same PHOC, but different index).
    # search using ground truth and find nearest neighbors of ground truth
    # find where special index is in that ground truth

    # find all the gt_boxes that have the same transcription

    # TODO: find recall@1,5,10
    # TODO: need analysis based on word length + edit distance from prediction to gt

    # TODO: not waste so much fucking time writing different evaluation scripts and loading thingies for
    # the data... there must be an easier way you know.
    # just flatten out everything so I don't have to think about everything all the time

    args = parser.parse_args()

    json_path = os.path.join(args.model_dir, 'params.json')
    print(json_path)
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

    params = utils.Params(json_path)
    params.args_to_params(args)

    statistics = eval_phoc_search(params)