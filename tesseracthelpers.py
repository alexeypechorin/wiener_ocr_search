import cv2 
import os 
import pytesseract
import numpy as np
from PIL import Image
from ocr_correction import correct

def get_data(img_path, output_dir='results'):
    # Read image using opencv
    img = cv2.imread(img_path)

    # Extract the file name without the file extension
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split('/')[0]

    # Create a directory for outputs
    # output_path = os.path.join(output_dir, file_name)
    # if not os.path.exists(output_path):
        # os.makedirs(output_path)
    
    # Rescale the image, if needed.
    # img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Convert to gray
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    # kernel = np.ones((1, 1), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1)
    # Apply blur to smooth out the edges
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold to get image with only b&w (binarization)
    # img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)[1]

    # Save the filtered image in the output directory
    # save_path = os.path.join(output_path, file_name + "_filter_" + str('dilate_erode_blur_bw') + ".jpg")
    # cv2.imwrite(save_path, img)

    # Recognize text with tesseract for python
    # result = pytesseract.image_to_string(img, lang="deu")
    result = pytesseract.image_to_data(img)
    return img, result

def process_tessaract_data(data_string):
    labels = data_string.split('\n')[0].split('\t')
    words = [word.split('\t') for word in data_string.split('\n')[1:] if word.split('\t')[-2] != str(-1)]
    # wow we actually get bounding box information along with OCR text
    # try drawing bounding boxes to see how accurate bounding boxes are

    boxes = []

    for word in words:
        if len(word) != 12:
            continue

        boxes += [{"bbox": [int(word[6]), int(word[7]), int(word[8]), int(word[9])],
                  "word": word[11],
                  "conf": int(word[10])}]

    return labels, boxes

def show_bboxes_with_text(img, boxes, sym_spell=None):
    candidate_corrections = []
    for i, box in enumerate(boxes):
        bbox = box['bbox']

        if sym_spell is not None:
            candidate_correction = correct(box['word'], sym_spell)
        else:
            candidate_correction = ''

        t = '[' + box['word'] + ',' + str(box['conf']) + ']'
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 255), thickness=1)
        fontFace = 0
        fontScale = .2 * (img.shape[0] / 720.0)
        thickness = 1
        fg = (0, 120, 0)
        textSize, baseline = cv2.getTextSize(t, fontFace, fontScale, thickness)
        cv2.putText(img, t, (bbox[0], bbox[1]),
                    fontFace, fontScale, fg, thickness)

        candidate_corrections += [candidate_correction]

    img = Image.fromarray(img)
    # img.show()

    return img