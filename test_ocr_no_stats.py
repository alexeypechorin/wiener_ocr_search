from tesseracthelpers import get_data, process_tessaract_data, show_bboxes_with_text
import evaluation as eval
import argparse
from ocr_correction import load_symspell
import os
import cv2
import utils
import json
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--data_dir', default='data')
parser.add_argument('--results_dir', default='results/simple_test')

def test_ocr_no_stats(params):
    overall_statistics = {}
    ocr_data = os.listdir(params.data_dir)
    ocr_images = [ocr_image for ocr_image in ocr_data if ocr_image.endswith('.jpg')]
    # sym_spell = load_symspell(os.path.join('data', 'de_50k.txt'))
    sym_spell = None
    for i, name in enumerate(ocr_images):
        print('---' + name + '---')
        # get reading from tesseract
        img_name = os.path.join(params.data_dir, name)
        img, result = get_data(img_name, lang=params.lang)
        labels, words = process_tessaract_data(result)
        with open(os.path.join(params.results_dir, name[:-4] + '.json'), 'w') as f:
            json.dump(words, f)
        img = show_bboxes_with_text(img, words, sym_spell=sym_spell)
        img.save(os.path.join(params.results_dir, name))


if __name__ == "__main__":
    # the following creates the OCR output for the entire training data
    args = parser.parse_args()

    json_path = os.path.join(args.model_dir, 'params.json')
    print(json_path)
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

    params = utils.Params(json_path)
    params.args_to_params(args)

    base_dirs = ['/media/data/datasets/wiener_cropped/wiener_images_cropped', '/media/data/datasets/wiener1_cropped/']

    for base_dir in base_dirs: 
        base_result_dir = 'wiener_tesseract_deu'
        
        # set german language settings
        params.lang = 'deu'
        collections = os.listdir(base_dir)

        collections = [coll for coll in collections if coll not in ['.', '..', '.DS_Store']]

        for collection in collections: 
            params.data_dir = os.path.join(base_dir, collection)
            params.results_dir = os.path.join(base_result_dir, collection)
	
   	    # create directories if they don't exist 
            if not os.path.exists(params.data_dir): 
                os.makedirs(params.data_dir)

            if not os.path.exists(params.results_dir): 
                os.makedirs(params.results_dir)

            test_ocr_no_stats(params)
