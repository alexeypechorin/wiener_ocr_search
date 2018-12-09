import os
import json
import codecs
import cv2
from tesseracthelpers import show_bboxes_with_text

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

def num_matches(gt_boxes, predicted_bboxes, iou_threshold_bbox=0.8, iou_threshold_spelling=0.8, max_edit_distance=0):
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
    for i, gt_box in enumerate(gt_abs):
        count_match = 0
        for j, pred_box in enumerate(pred_abs):
            iou_amt = iou(gt_box, pred_box)

            if iou_amt > iou_threshold_bbox:
                total_bbox_match += 1
                match_idx += [j]

            if iou_amt > iou_threshold_spelling:
                if edit_distance(gt_boxes[i]['word'],predicted_bboxes[j]['word'].lower(), max_edit_distance):
                    total_spelling_match += 1

    corr_matches = [predicted_bboxes[idx] for idx in match_idx]

    # return precision, recall for bboxes, spelling, number of predictions, number of correct predictions

    total_predictions = len(pred_abs)
    correct_predictions = total_bbox_match

    precision_bbox = float(total_bbox_match)/len(pred_abs)
    recall_bbox = float(total_bbox_match)/len(gt_abs)

    precision_spelling = float(total_spelling_match)/len(pred_abs)
    recall_spelling = float(total_spelling_match)/len(gt_abs)

    return {"1-num_gt_bboxes": len(gt_abs),
            "3-bbox_matches": correct_predictions,
            "2-num_bbox_preds": total_predictions,
            "4-precision_bbox": precision_bbox,
            "5-recall_bbox": recall_bbox,
            "6-precision_spelling": precision_spelling,
            "7-recall_spelling": recall_spelling}


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