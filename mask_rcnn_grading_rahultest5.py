
"""
Matterport Mask R-CNN Grading
Example1: python3 mask_rcnn_grading.py splash --category=onion --client_name=amazon_now --response=mask_rcnn --grade_id=144822_mini --image=/home/safwan/vegetable_grading/amazon_now/onion/144822_mini.jpg
Example2: /home/deploy/vegetable_grading/kcpmc/cardamom/venv3_cardamom/bin/python3 -W ignore /home/deploy/vegetable_grading/amazon_now/mask_rcnn_grading_onion_v2.py splash --category=onion --client_name=amazon_now --grade_id=144822 --output_path=/home/deploy/imagerepo/shared/public/production_VegetableGrading_144822_output.out --environment=production --auto_validator_req=yes --image=https://s3.ap-south-1.amazonaws.com/autovalidator/production/uploads/image/file_new/145549/OAZNW181227O06.jpg

Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import random
import os
import sys
import json
import time
import numpy as np
import skimage.draw
import cv2
import urllib
import traceback
from IPython import embed
from scipy.spatial import distance
#import detect
import cloud_vision_intello_marker
import boto3
import argparse
ROOT_DIR = os.path.abspath("/home/" + os.environ["USER"] + "/mask_rcnn")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

sys.path.append(ROOT_DIR)
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn import model as modellib, utils

script_path = os.path.realpath(__file__)
# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Train Mask R-CNN to detect custom class.')
parser.add_argument("command",
                    metavar="<command>",
                    help="'splash'")
parser.add_argument("--category", required=True,
                    metavar="commodity_category",
                    help="Category of commodity that for grading")
parser.add_argument("--client_name", required=True,
                    default="demo",
                    metavar="client_name",
                    help="Name of the client")
parser.add_argument('--grade_id', required=False,
                    default=time.time(),
                    metavar="Grade_Id",
                    help='Unique grade id for each image')
parser.add_argument('--response', required=True,
                    default="mask_rcnn",
                    metavar="model response type",
                    choices=['mask_rcnn', 'faster_rcnn'],
                    help='Response required for image')
parser.add_argument("--output_path", required=False,
                    default=None,
                    metavar="/path/to/output/json/file",
                    help="Path to output Json file")
parser.add_argument("--environment", required=False,
                    default="development",
                    metavar="development/production/staging",
                    help="Development or Production or Staging")
parser.add_argument("--auto_validator_req", required=False,
                    default="no",
                    metavar="yes/no",
                    help="Is it for Autovalidator?")
parser.add_argument('--logs', required=False,
                    default=DEFAULT_LOGS_DIR,
                    metavar="/path/to/logs/",
                    help='Logs and checkpoints directory (default=logs/)')
parser.add_argument('--image', required=False,
                    metavar="path or URL to image",
                    help='Image to apply the color splash effect on')
parser.add_argument('--validation_id', required=False,
                    default=str(int(time.time())),
                    metavar="s3_path for debug images",
                    help='Image to apply the color splash effect on')
parser.add_argument('--bucket_name', required=False,
                    default="autovalidator",
                    metavar="bucket for debug images",
                    help='Bucket name where debug images are stored')

args, unknown = parser.parse_known_args()
# args = parser.parse_args()
image = args.image
try:
    path = "/".join(script_path.split("/")[0:-2])
    if path == "":
        path = os.path.dirname(os.path.realpath(__file__))
except:
    path = "/home/" + os.environ["USER"] + "/vegetable_grading/"

os.chdir(path)
original_image_path = "/".join(image.split("/")[:-1])
filename = image.split('/')[-1]

client_path = os.path.join(path, args.client_name)
category_path = os.path.join(client_path, args.category)
global configfile
f = open(os.path.join(category_path, "config_rahultest.json"), "r")
configfile = json.load(f)
print("Config file:")
print(configfile)


class CustomConfig(Config):
    NAME = configfile['category']
    BACKBONE = configfile['BACKBONE']
    NUM_CLASSES = configfile['NUM_CLASSES']
    STEPS_PER_EPOCH = configfile['STEPS_PER_EPOCH']
    if 'IMAGE_MIN_DIM' in configfile.keys():
        IMAGE_MIN_DIM = configfile['IMAGE_MIN_DIM']
    if 'IMAGE_MAX_DIM' in configfile.keys():
        IMAGE_MAX_DIM = configfile['IMAGE_MAX_DIM']
    if 'TRAIN_ROIS_PER_IMAGE' in configfile.keys():
        TRAIN_ROIS_PER_IMAGE = configfile['TRAIN_ROIS_PER_IMAGE']
    if 'DETECTION_MIN_CONFIDENCE' in configfile.keys():
        DETECTION_MIN_CONFIDENCE = configfile['DETECTION_MIN_CONFIDENCE']
    if 'MAX_GT_INSTANCES' in configfile.keys():
        MAX_GT_INSTANCES = configfile['MAX_GT_INSTANCES']
    if 'DETECTION_MAX_INSTANCES' in configfile.keys():
        DETECTION_MAX_INSTANCES = configfile['DETECTION_MAX_INSTANCES']
    if 'DETECTION_NMS_THRESHOLD' in configfile.keys():
        DETECTION_NMS_THRESHOLD = configfile['DETECTION_NMS_THRESHOLD']
    if 'RPN_TRAIN_ANCHORS_PER_IMAGE' in configfile.keys():
        RPN_TRAIN_ANCHORS_PER_IMAGE = configfile['RPN_TRAIN_ANCHORS_PER_IMAGE']
    if 'PRE_NMS_LIMIT' in configfile.keys():
        PRE_NMS_LIMIT = configfile['PRE_NMS_LIMIT']
    if 'POST_NMS_ROIS_TRAINING' in configfile.keys():
        POST_NMS_ROIS_TRAINING = configfile['POST_NMS_ROIS_TRAINING']
    if 'POST_NMS_ROIS_INFERENCE' in configfile.keys():
        POST_NMS_ROIS_INFERENCE = configfile['POST_NMS_ROIS_INFERENCE']
    if 'BACKBONE_STRIDES' in configfile.keys():
        BACKBONE_STRIDES = configfile['BACKBONE_STRIDES']
    if 'RPN_ANCHOR_SCALES' in configfile.keys():
        RPN_ANCHOR_SCALES = tuple(configfile['RPN_ANCHOR_SCALES'])
    if 'MASK_THRESHOLD' in configfile.keys():
        MASK_THRESHOLD = configfile['MASK_THRESHOLD']
    if 'ROI_POSITIVE_RATIO' in configfile.keys():
        ROI_POSITIVE_RATIO = configfile['ROI_POSITIVE_RATIO']
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 4


class mask_rcnn_grading(object):
    ROOT_DIR = os.path.abspath("/home/" + os.environ["USER"] + "/mask_rcnn")
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

    def mask_to_polygon(self, mask):
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        points_drw = []
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        if not len(contour_sizes):
            return points
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        clen = len(biggest_contour)
        clen -= 2
        points.append(biggest_contour[0][0].tolist())
        points_drw.append(contours[0])
        i = 1
        while i <= clen:
            points.append(biggest_contour[i][0].tolist())
            i += 1
        points.append(biggest_contour[clen + 1][0].tolist())
        return points

    def max_area_contour(self, mask):
        mask_image = mask.astype(np.uint8)
        mask_image *= 255
        try:
            im, contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        except:
            contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        ma=-1
        index=-1
        for i in range(len(contours)):
            if ma < cv2.contourArea(contours[i]):
                index = i
                ma = cv2.contourArea(contours[i])

        return contours, index

    def get_layer2_classification(self, category_index, contour_details, final_result_dict, detection_results, contour_images_folder, image_contours, original_image, debug_image):
        global defect_contours_final
        defect_contours_final = []
        for detail in contour_details:
            details = detail.split(";")
            image_path = details[0]
            commodity_index = os.path.splitext(os.path.basename(image_path))[0]
            x = int(float(details[1]))
            y = int(float(details[2]))
            key = "_".join(details[1:])
            image = cv2.imread(image_path)
            im_width, im_height = image.shape[1], image.shape[0]
            image_np = np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)
            output_dict = detection_results[image_path]
            scores = output_dict['scores']
            boxes = output_dict['rois']
            classes = output_dict['class_ids']
            returned_masks = output_dict['masks']
            defect_found = False
            font = cv2.FONT_HERSHEY_SIMPLEX
            contour_sub_defects = {}
            trained_defects = configfile['trained_defects']
            for defect in trained_defects.keys():
                contour_sub_defects[defect] = []

            for i in range(min(10, boxes.shape[0])):
                if scores[i] >= 0.1:  # scores is None or
                    box = tuple(boxes[i].tolist())
                    display_str = ''
                    ymin, xmin, ymax, xmax = box
                    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]
                        area = abs(left - right) * abs(top - bottom)
                        contour_sub_defects[class_name].append((scores[i], area, i))
                        contours, max_index = self.max_area_contour(returned_masks[i, :, :])
                        contour_points = np.array([])
                        contour_area = 0
                        if max_index != -1:
                            contour_points = contours[max_index]
                            contour_area = cv2.contourArea(contours[max_index])
                            if "color" in trained_defects[class_name]:
                                color_name = list(trained_defects[class_name]["color"].keys())[0]
                                defect_color = tuple(trained_defects[class_name]["color"][color_name])
                                cv2.drawContours(image_np, contours, -1, defect_color, 1)
                                cv2.putText(image_np, str(round(scores[i], 2)) + ":" + str(category_index[classes[i]])[0:2], (int((left + right) / 2), int((top + bottom) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, defect_color, 1)
                            else:
                                cv2.drawContours(image_np, contours, -1, (0, 0, 0), 1)
                                cv2.putText(image_np, str(round(scores[i], 2)) + ":" + str(category_index[classes[i]])[0:2], (int((left + right) / 2), int((top + bottom) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                        contour_points_final = [point[0] for point in contour_points.tolist()]
                        contour_length = 0
                        if len(contour_points_final):
                            contour_length = round(cv2.arcLength(np.array(contour_points_final), True) / 2.0, 2)
                            contour_points_final = (np.array(contour_points_final) + (x, y)).tolist()
                        box_list = boxes[i].tolist()
                        defect_roi_final = [box_list[0] + y, box_list[1] + x, box_list[2] + y, box_list[3] + x]
                        defect_contours_final.append({
                            "id": i,
                            "commodity_id": commodity_index,
                            "roi": defect_roi_final,
                            "score": int(scores[i] * 100),
                            "area": contour_area,
                            "sub_category": category_index[classes[i]],
                            "category": "",
                            "S3_path": "",
                            "points": contour_points_final,
                            "size": str(contour_length)
                        })
                    else:
                        class_name = 'N/A'
                    display_str = str(class_name)
                    display_str = '{}: {}'.format(display_str, int(100 * scores[i]))

            # cv2.imwrite(os.path.join(os.path.join(contour_images_folder,"final"), image_path.split("/")[-1].split(".")[0]) + '_op.jpg',image_np)
            image_contours[key]["path"] = os.path.join(contour_images_folder, "final", image_path.split("/")[-1].split(".")[0]) + '_op.jpg'
            defect_matrix = {}

            for defect in contour_sub_defects.keys():
                defect_matrix[defect] = [False, 0, 0, 0]

            area_text_ypos = 50
            contour_id = image_path.split("/")[-1].split(".")[0]
            print("\nContour ID: {}".format(contour_id))
            for defect in contour_sub_defects.keys():
                if len(contour_sub_defects[defect]) is not 0:
                    print("Defect: {}".format(defect))
                    Count = 0
                    Max = 0
                    Bool = False
                    index = -1
                    area = 0
                    for i in range(len(contour_sub_defects[defect])):
                        if contour_sub_defects[defect][i][1]:
                            if contour_sub_defects[defect][i][0] >= trained_defects[defect]['probability_threshold']:
                                Count += 1
                                Bool = True
                                contours, max_index = self.max_area_contour(returned_masks[contour_sub_defects[defect][i][2], :, :])
                                if max_index != -1:
                                    area += cv2.contourArea(contours[max_index])  # area is added only if confidence is greater that probability threshold
                                if contour_sub_defects[defect][i][0] > Max:
                                    Max = contour_sub_defects[defect][i][0]
                                    index = contour_sub_defects[defect][i][2]
                    foreground_area = ((np.count_nonzero(image != 0)) / 3)
                    area_percent = round((area / foreground_area) * 100, 1)
                    median_area_percent = round((area / area_median) * 100, 1)
                    print("Area: {}".format(area))
                    print("Foreground area: {}".format(foreground_area))
                    print("Area percentage: {}".format(area_percent))
                    if area_percent != 0:
                        cv2.putText(image_np, defect + ":" + str(area_percent) + "%", (25, int(area_text_ypos)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                        area_text_ypos += 25

                    if median_area_percent != 0:
                        cv2.putText(image_np, defect+"_m" + ":" + str(median_area_percent) + "%", (25, int(area_text_ypos)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                        area_text_ypos += 25
                            
                    if 'area_percentage_threshold' in trained_defects[defect] and trained_defects[defect]['area_percentage_threshold'] != 0:
                        if area_percent >= trained_defects[defect]['area_percentage_threshold']:
                            defect_matrix[defect][0] = Bool
                            defect_matrix[defect][1] = Count
                            defect_matrix[defect][2] = Max
                            defect_matrix[defect][3] = index
                    elif 'median_area_percentage_threshold' in trained_defects[defect] and trained_defects[defect]['median_area_percentage_threshold'] != 0:
                        if median_area_percent >= trained_defects[defect]['median_area_percentage_threshold']:
                            defect_matrix[defect][0] = Bool
                            defect_matrix[defect][1] = Count
                            defect_matrix[defect][2] = Max
                            defect_matrix[defect][3] = index
                    else:
                        defect_matrix[defect][0] = Bool
                        defect_matrix[defect][1] = Count
                        defect_matrix[defect][2] = Max
                        defect_matrix[defect][3] = index

            print("Contour sub defects: {}".format(contour_sub_defects))
            print("Defect matrix: {}".format(defect_matrix))
            cv2.imwrite(os.path.join(os.path.join(contour_images_folder, "final"), contour_id) + '_op.jpg', image_np)

            result_str = ''
            contours_final = np.array([])
            for defect in defect_matrix.keys():
                if defect_matrix[defect][0]:
                    if result_str == '':
                        result_str = defect
                        actual_defect = list(trained_defects[defect]['actual_defect'].keys())[0]
                        final_result_dict[key] = str(image_path) + ";" + str(actual_defect)
                        index = defect_matrix[defect][3]
                        contours, max_index = self.max_area_contour(returned_masks[index, :, :])

                        if len(contours):
                            contours_final = np.array(contours[max_index])
                    elif trained_defects[result_str]['priority'] < trained_defects[defect]['priority']:
                        result_str = defect
                        actual_defect = list(trained_defects[defect]['actual_defect'].keys())[0]
                        final_result_dict[key] = str(image_path) + ";" + str(actual_defect)
                        index = defect_matrix[defect][3]
                        contours, max_index = self.max_area_contour(returned_masks[index, :, :])
                        if len(contours):
                            contours_final = np.array(contours[max_index])

            if result_str == '':
                result_str = "normal"
                final_result_dict[key] = str(image_path) + ";" + str(result_str)
            if result_str != 'normal':
                if len(contours_final):
                    contours_final += (x, y)
                    defect_color = (0, 0, 0)
                    if "color" in trained_defects[result_str]:
                        color_name = list(trained_defects[result_str]["color"].keys())[0]
                        defect_color = tuple(trained_defects[result_str]["color"][color_name])
                    cv2.drawContours(debug_image, contours_final, -1, defect_color, 3)
                    cv2.putText(debug_image, str(result_str + "::" + str(defect_matrix[result_str][2])[:4]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (70, 120, 220), 3)

                    cv2.drawContours(original_image, contours_final, -1, (0, 220, 0), 3)
                    cv2.putText(original_image, str(result_str + "::" + str(defect_matrix[result_str][2])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (70, 120, 220), 3)
        cv2.imwrite(os.path.join(contour_images_folder, "mask_out.jpg"), original_image)
        cv2.imwrite(os.path.join(contour_images_folder,"final","debug_defect.jpg"),debug_image)

        return final_result_dict,image_contours

    def get_layer2_mask_object_detection_results_(self, contour_details, final_result_dict, ratio, model, contour_images_folder, original_image_path, image_contours):
        contour_paths=[]
        detection_results={}
        mapping={}

        for key in configfile['Classification_Config']['mapping'].keys():
            mapping[configfile['Classification_Config']['mapping'][key]]=key

        original_image = cv2.imread(original_image_path)
        for detail in contour_details:
            details = detail.split(";")
            contour_image_name = details[0]
            key = "_".join(details[1:])
            contour_paths.append(contour_image_name)
        images = []

        for contour_path in contour_paths:

            image = skimage.io.imread(contour_path)
            # images.append(image)
            result = model.detect([image],verbose=1)[0]
            result['scores'] = np.array(result['scores'])
            result['rois'] = np.array(result['rois'])
            result['class_ids'] = np.array(result['class_ids'])
            result['masks'] = np.array(result['masks'])
            detection_results[contour_path]=result
        print(model.config.NUM_CLASSES,"NUM_CLASSES")
        # result = model.detect(images, verbose=1)

        # batch prediction
        # for i in range(len(result)):
        #     result[i]['scores'] = np.array(result[i]['scores'])
        #     result[i]['rois'] = np.array(result[i]['rois'])
        #     result[i]['class_ids'] = np.array(result[i]['class_ids'])
        #     result[i]['masks'] = np.array(result[i]['masks'])
        #     detection_results[contour_paths[i]]=result[i]

        category_index = mapping
        debug_image = cv2.imread(contour_images_folder+"/debug_defect.jpg")

        return self.get_layer2_classification(category_index, contour_details, final_result_dict, detection_results, contour_images_folder, image_contours, original_image,debug_image)

    def get__defect_count(self,image_path,contour_details,contour_images_folder,original_image_path,category,colors,model_path,label_path,image_contours,ratio,args,model):
        defects = []
        cat_sub_cat_mapping = {"normal": "normal"}
        trained_defects = configfile['trained_defects']
        for trained_defect in trained_defects.keys():
            actual_defect = list(trained_defects[trained_defect]['actual_defect'].keys())[0]
            value = trained_defects[trained_defect]['actual_defect'][actual_defect]
            if actual_defect not in defects:
                defects.append(actual_defect)
                cat_sub_cat_mapping[actual_defect] = value
        sub_defect_count={}
        colors_dict_sub_cat={}
        for index in range(len(defects)):
            sub_defect_count[defects[index]]=0
            colors_dict_sub_cat[defects[index]]=colors[index%len(colors)]
        temp1 = sub_defect_count
        defect_count = {
            "major": 0,
            "minor": 0,
            "normal": 0
        }
        temp = defect_count
        colors_dict = {
            "major": colors[3],
            "minor": colors[1],
            "normal": colors[2]
        }
        final_result_dict = {}
        original_image = cv2.imread(image_path)
        rad = int(max(original_image.shape[0],original_image.shape[1])*0.005)
        copy_image = cv2.imread(image_path)
        sub_defect_image = cv2.imread(image_path)

        images = {}
        for key in defect_count.keys():
            images[key] = original_image.copy()
        images1 = {}
        for key in sub_defect_count.keys():
            images1[key] = original_image.copy()

        final_contours = []
        # remaining_contour_details = []
        # ignore_contour_details = []

        for index in range(len(contour_details)):
            # Read in the image_data
            detail = contour_details[index]

            final_contours.append(detail)

        final_result_dict,image_contours = self.get_layer2_mask_object_detection_results_(final_contours,final_result_dict,ratio,model,contour_images_folder,image_path,image_contours)

        for key in final_result_dict.keys():
            result = final_result_dict[key]
            contour_details = key.split("_")
            x = int(contour_details[0])
            y = int(contour_details[1])
            minor_axis = int(float(contour_details[2]))
            minor_axis_size=int(float(contour_details[3]))
            major_axis= int(float(contour_details[4]))
            major_axis_size=int(float(contour_details[5]))
            angle=float(contour_details[6])
            contour_details = result.split(";")
            contour_image_name = contour_details[0]
            human_string = contour_details[1]
            # human_string = human_string.replace(" ","_")
            if human_string in temp1.keys():
                print(human_string)
                image_contours[key]["defect_sub_cat"] = human_string
                temp1[human_string] += 1
                cv2.ellipse(sub_defect_image, ((int(x),int(y)),(minor_axis, major_axis),angle), colors_dict_sub_cat[human_string],thickness=2)
                cv2.ellipse(images1[human_string], ((int(x),int(y)),(minor_axis, major_axis),angle), colors_dict_sub_cat[human_string],thickness=2)
                if human_string != "normal":
                    new_path = "/".join(contour_image_name.split("/")[:-1])+"/"+ human_string +"/"+str(temp[cat_sub_cat_mapping[human_string]] + 1)+ "." +contour_image_name.split(".")[-1]
                    print(new_path)
                    os.system('cp '+ contour_image_name + " " + new_path)
            if human_string in cat_sub_cat_mapping.keys():
                human_string = cat_sub_cat_mapping[human_string]
            if human_string != "ignore":
                temp[human_string] += 1
                image_contours[key]["defect"] = human_string
                cv2.ellipse(images[human_string], ((int(x),int(y)),(minor_axis, major_axis),angle),colors_dict[human_string],thickness=2)
                cv2.ellipse(copy_image, ((int(x),int(y)),(minor_axis, major_axis),angle), colors_dict[human_string],thickness=2)
                cv2.putText(copy_image, str(temp[human_string]), (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, colors[0], 5)
                new_path = "/".join(contour_image_name.split("/")[:-1])+"/"+ human_string +"/"+str(temp[human_string])+ "." +contour_image_name.split(".")[-1]
                os.system('cp '+ contour_image_name + " " + new_path)
            else:
                temp[human_string] += 1
                image_contours.pop(key,None)
                new_path = "/".join(contour_image_name.split("/")[:-1])+"/ignore/"+str(temp["ignore"])+ "." +contour_image_name.split(".")[-1]
                os.rename(contour_image_name,new_path)

        height,width,channels = original_image.shape
        image_ratio = height/500
        if image_ratio == 0:
            image_ratio = 1
        for image in images.keys():
            img = images[image]
            cv2.imwrite(os.path.join(contour_images_folder,image+"_web.jpg"),img)
            img = cv2.resize(img,(int(width/image_ratio),int(height/image_ratio)))
            cv2.imwrite(os.path.join(contour_images_folder,image+".jpg"),img)
        for image in images1.keys():
            img = images1[image]
            cv2.imwrite(os.path.join(contour_images_folder,image+"_web.jpg"),img)
            img = cv2.resize(img,(int(width/image_ratio),int(height/image_ratio)))
            cv2.imwrite(os.path.join(contour_images_folder,image+".jpg"),img)

        cv2.imwrite(os.path.join(contour_images_folder,"defects_web.jpg"),copy_image)
        copy_image = cv2.resize(copy_image,(int(width/image_ratio),int(height/image_ratio)))
        cv2.imwrite(os.path.join(contour_images_folder,"defects.jpg"),copy_image)
        cv2.imwrite(os.path.join(contour_images_folder,"sub_defects_web.jpg"),sub_defect_image)
        sub_defect_image = cv2.resize(sub_defect_image,(int(width/image_ratio),int(height/image_ratio)))
        cv2.imwrite(os.path.join(contour_images_folder,"sub_defects.jpg"),sub_defect_image)

        return temp, temp1, final_contours, image_contours, final_result_dict

    def detect_and_color_splash(self, model, contour_images_folder, output_file, image_path, original_image_path, args):
        if image_path:
            import cv2
            contour_details = []
            image_contours = {}
            print("Running on {}".format(args.image))
            image = skimage.io.imread(args.image)
            r = model.detect([image], verbose=1)[0]
            del model
            r['scores'] = np.array(r['scores'])
            r['rois'] = np.array(r['rois'])
            r['class_ids'] = np.array(r['class_ids'])
            r['masks'] = np.array(r['masks'])
            input_image = cv2.imread(image_path)
            output_image = input_image.copy()
            class_mapping = {}
            for key in configfile['mapping'].keys():
                class_mapping[configfile['mapping'][key]]=key

            APIkey = 'AIzaSyAXRBE1Dp6mmn5TBpuHpUOgx94def2NzrI'
            ratio = cloud_vision_intello_marker.getRatio(contour_images_folder,APIkey,image_path)
            # crop_image = input_image.copy()
            # crop_image[:,:,:]=0

            COLORS = range(1, 256)

            image_contours_final = []
            commodity_s3_paths = {}
            img = input_image.copy()
            img_1 = input_image.copy()
            img_2 = input_image.copy()
            img_3 = input_image.copy()

            # removing overlapping objects
            if 'IOU' in configfile:
                _mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

                for i in range(len(r['rois'])):
                    _msk = r['masks'][i].astype(np.uint8)
                    inter = np.bitwise_or(_mask, _msk)
                    inter_area = np.count_nonzero(_mask) + np.count_nonzero(_msk) - np.count_nonzero(inter)
                    if inter_area > 0:
                        if (inter_area / np.count_nonzero(_msk)) < configfile['IOU']:
                            _mask = np.bitwise_or(inter, _mask)
                        else:
                            r['scores'][i] = 0.0
                    else:
                        _mask = np.bitwise_or(inter, _mask)


            identification_contour_areas_arr = []
            contour_areas_below_median_area_index = []
            global area_median
            
            for i in range(len(r['rois'])):
                if r['scores'][i] >= configfile['probablity_identification']:
                    identification_contours, identification_max_index = self.max_area_contour(r['masks'][i, :, :])
                    identification_contour_area = 0
                    if identification_max_index != -1:
                        identification_contour_area = cv2.contourArea(identification_contours[identification_max_index])
                        identification_contour_areas_arr.append(identification_contour_area)
                if(r['scores'][i] > configfile['probablity_identification']):
                    y1, x1, y2, x2 = r['rois'][i]
                    im2, contours, hierarchy = cv2.findContours(np.asarray(r['masks'][i], dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_1, contours, -1, (0, 0, 0), 3)
                    cv2.drawContours(img_3, contours, -1, (0, 0, 0), 3)
                    color_choice = (random.choice(COLORS), random.choice(COLORS), random.choice(COLORS))
                    for ii in contours:
                        cv2.fillConvexPoly(img, np.array(ii), color_choice)
                        # cv2.fillPoly(img, ii, (random.choice(COLORS),random.choice(COLORS),random.choice(COLORS)) , lineType=8, shift=0)
                    cv2.putText(img_1, str(r['scores'][i]), (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, str(r['scores'][i]), (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)

            alpha = 0.7
            cv2.addWeighted(img, alpha, img_2, 1 - alpha,0, img_2)
            cv2.imwrite(contour_images_folder + "/debug_COLOR_FILLED.jpg",img_2)
            cv2.imwrite(contour_images_folder + "/debug.jpg",img_1)
            cv2.imwrite(contour_images_folder + "/debug_defect.jpg",img_3)

            yminarr,ymaxarr,xminarr,xmaxarr = ([],[],[],[])
            for roi in r['rois']:
                yminarr.append(roi[0])
                xminarr.append(roi[1])
                ymaxarr.append(roi[2])
                xmaxarr.append(roi[3])

            xmin,ymin,xmax,ymax = (min(xminarr),min(yminarr),max(xmaxarr),max(ymaxarr))
            total_area = (xmax-xmin)*(ymax-ymin)
            foreground_area_percent = round((np.sum(identification_contour_areas_arr)/total_area)*100,2)
            foreground_area_activation = True
            if "layer2_objects_removal" in configfile and "foreground_area_percent" in configfile["layer2_objects_removal"]  and foreground_area_percent < configfile["layer2_objects_removal"]['foreground_area_percent']:
                foreground_area_activation = False
                print("foreground : false")

            area_median = np.median(np.array(identification_contour_areas_arr))
            print("Contour Areas:")
            print(identification_contour_areas_arr)
            print("Median Area :", area_median)

            if ("layer2_objects_removal" in configfile) and ("active" in configfile["layer2_objects_removal"]) and configfile["layer2_objects_removal"]["active"]=="True":
                l2_conditions = configfile["layer2_objects_removal"]
                layer2_img = input_image.copy()
                if foreground_area_activation:
                    if "mode" not in configfile["layer2_objects_removal"]:
                        print("Layer2 removal: active")
                        for i in range(len(r['rois'])):
                            if(r['scores'][i] > configfile['probablity_identification']):
                                ymin, xmin, ymax, xmax = r['rois'][i]
                                w,h = (xmax - xmin,ymax - ymin)
                                max_dim = max(h, w)
                                hw_ratio = float(h / w)
                                mask_area_coverage = (np.count_nonzero(r['masks'][i, :, :]) / (max_dim * max_dim)) * 100
                                if max_dim > l2_conditions["dimension_floor"] and hw_ratio >= l2_conditions["hw_ratio_floor"] and hw_ratio <= l2_conditions["hw_ratio_ceil"] and mask_area_coverage >= l2_conditions["mask_coverage_percent"]:
                                    # layer 1 objects
                                    pass
                                else:
                                    # layer 2 objects
                                    print("Layer2 Object ID: {}, max_dim: {}, hw_ratio: {}, mask_area_coverage: {}".format(i, max_dim, hw_ratio, mask_area_coverage))
                                    im2, contours, hierarchy = cv2.findContours(np.asarray(r['masks'][i], dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                    color_choice = (random.choice(COLORS), random.choice(COLORS), random.choice(COLORS))
                                    contour_areas_below_median_area_index.append(i)
                                    for ii in contours:
                                        cv2.fillConvexPoly(layer2_img, np.array(ii), color_choice)
                                    cv2.putText(layer2_img, str(i) + ":" + str(round(r['scores'][i], 3)), (int((xmin + xmax) / 2), int((ymin + ymax) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)
                                    r['scores'][i] = 0.0
                        alpha = 0.7
                        cv2.addWeighted(layer2_img, alpha, input_image, 1 - alpha, 0, layer2_img)
                        cv2.imwrite(contour_images_folder + "/layer2_objects.jpg", layer2_img)

                    elif "mode" in configfile["layer2_objects_removal"] and configfile["layer2_objects_removal"]["mode"] == "median":
                        print("Layer2 removal: active")
                        median_area_threshold_factor = l2_conditions['median_area_threshold_factor']
                        for i in range(len(r['rois'])):
                            if(r['scores'][i] > configfile['probablity_identification']):
                                ymin, xmin, ymax, xmax = r['rois'][i]
                                contour_area, contour_area_index = self.max_area_contour(r['masks'][i, :, :])
                                if contour_area_index != -1:
                                    contour_area = cv2.contourArea(contour_area[contour_area_index])
                                if contour_area  > area_median * median_area_threshold_factor :
                                    # layer 1 objects
                                    pass
                                else:
                                    # layer 2 objects
                                    print("Removing Layer2 Object ID: {}, mask_area_coverage: {}, median_area_threshold: {}".format(i, contour_area, area_median * median_area_threshold_factor))
                                    im2, contours, hierarchy = cv2.findContours(np.asarray(r['masks'][i], dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                    color_choice = (random.choice(COLORS), random.choice(COLORS), random.choice(COLORS))
                                    contour_areas_below_median_area_index.append(i)
                                    for ii in contours:
                                        cv2.fillConvexPoly(layer2_img, np.array(ii), color_choice)
                                    cv2.putText(layer2_img, str(i) + ":" + str(round(r['scores'][i], 3)), (int((xmin + xmax) / 2), int((ymin + ymax) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)
                                    r['scores'][i] = 0.0
                        alpha = 0.7
                        cv2.addWeighted(layer2_img, alpha, input_image, 1 - alpha, 0, layer2_img)
                        cv2.imwrite(contour_images_folder + "/layer2_objects.jpg", layer2_img)
                        print("in median mode ")
                    else:
                        print("Layer2 removal: Inactive")
                else:
                    print("Single Layer, no need layer2 removal")
            else:
                print("Layer2 removal: Inactive")       

            identification_contour_areas_new_arr = []
            for i in range(len(r['rois'])):
                if r['scores'][i] >= configfile['probablity_identification']:
                    identification_contours, identification_max_index = self.max_area_contour(r['masks'][i, :, :])
                    if identification_max_index != -1:
                        identification_contour_area = cv2.contourArea(identification_contours[identification_max_index])
                        identification_contour_areas_new_arr.append(identification_contour_area)

            area_median = np.median(np.array(identification_contour_areas_new_arr))
            print("Median Area :", area_median)

            defect_rois = []
            if 'IOU' in configfile:
                _mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            for i in range(len(r['rois'])):
                if r['scores'][i] > configfile['probablity_identification']:
                    y1, x1, y2, x2 = r['rois'][i]
                    crop_image = input_image[y1:y2,x1:x2]
                    crop_image_copy = crop_image.copy()
                    mask_copy = np.zeros(crop_image_copy.shape, dtype=np.uint8)
                    points = self.mask_to_polygon(np.array(r['masks'][i,:,:][y1:y2,x1:x2],dtype=np.uint8))
                    if not len(points):
                        continue
                    roi_corners = np.array(points, dtype=np.int32)
                    channel_count = crop_image_copy.shape[2]
                    ignore_mask_color = (255,) * channel_count
                    cv2.fillPoly(mask_copy, [roi_corners], ignore_mask_color)
                    only_foreground = cv2.bitwise_and(crop_image_copy, mask_copy)
                    if 'with_background' in configfile and configfile['with_background']:
                        only_foreground = crop_image_copy
                    extension = args.image.split('.')[-1]

                    if r['class_ids'][i] == configfile['mapping'][args.category]:
                        cv2.imwrite(os.path.join(contour_images_folder,str(i)+"."+extension),only_foreground)
                    mask = r['masks'][i,:,:]
                    mask_image = mask.astype(np.uint8)
                    mask_image*=255
                    cropped_mask = mask_image[y1:y2,x1:x2]
                    cv2.imwrite(os.path.join(contour_images_folder,"mask_"+str(i)+".jpg"),cropped_mask)
                    contours,max_index=self.max_area_contour(cropped_mask)
                    if max_index!=-1:
                        elps = cv2.fitEllipse(contours[max_index])
                    else:
                        angle=0
                    c1 = int(round((x1+x2)/2))
                    c2 = int(round((y1+y2)/2))

                    minor_axis = round(min(x2-x1,y2-y1))
                    major_axis = round(max(x2-x1,y2-y1))

                    minor_axis_size = minor_axis
                    major_axis_size = major_axis
                    padding = int(major_axis*0.05)
                    minor_axis = round(minor_axis + padding)
                    major_axis = round(major_axis + padding)
                    if max_index!=-1:
                        angle=elps[2]
                    cv2.ellipse(output_image, ((int(c1),int(c2)),(minor_axis, major_axis),angle), (255,255,255),thickness=3)
                    length=0
                    width=0
                    cv2.putText(output_image,str(int(length))+":: "+str(int(width)),(int((x1+x2)/2),int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),1,cv2.LINE_AA)

                    contour_path = os.path.join(contour_images_folder,str(i)+"."+image_path.split(".")[-1])
                    if args.response=='mask_rcnn':
                        key = str(int(x1)) + "_" + str(int(y1)) + "_" + str(minor_axis) + "_" + str(minor_axis_size)  + "_" + str(major_axis)  + "_" + str(major_axis_size)  + "_" + str(angle)
                        detail = contour_path+";"+str(int(x1))+";"+str(int(y1))+";"+str(minor_axis)+";"+str(minor_axis_size)+";"+str(major_axis)+";"+str(major_axis_size)+";"+str(angle)
                    else:
                        key = str(int(c1)) + "_" + str(int(c2)) + "_" + str(minor_axis) + "_" + str(minor_axis_size)  + "_" + str(major_axis)  + "_" + str(major_axis_size)  + "_" + str(angle)
                        detail = contour_path+";"+str(int(c1))+";"+str(int(c2))+";"+str(minor_axis)+";"+str(minor_axis_size)+";"+str(major_axis)+";"+str(major_axis_size)+";"+str(angle)

                    S3_commodity_path="S3_commodity_path"
                    S3_debug_image_path="S3_debug_image_path"

                    points = self.mask_to_polygon(np.asarray(mask, dtype=np.uint8))
                    if r['class_ids'][i] == configfile['mapping'][args.category] or r['class_ids'][i] == configfile['mapping']['normal']:
                        image_contours[key] = {
                            'id': i,
                            'width': str(width),
                            'points': points
                        }
                    if args.response == 'faster_rcnn':
                        image_contours[key]["defect"] = "normal"
                        image_contours[key]["defect_sub_cat"] = "normal"
                    # print("Class ID: {}".format(r['class_ids'][i]))
                    # print("Category ID: {}".format(configfile['mapping'][args.category]))
                    if r['class_ids'][i] == configfile['mapping'][args.category] or r['class_ids'][i] == configfile['mapping']['normal']:
                        contour_details.append(detail)
                        identification_contours, identification_max_index = self.max_area_contour(r['masks'][i, :, :])
                        identification_contour_area = 0
                        if identification_max_index != -1:
                            identification_contour_area = cv2.contourArea(identification_contours[identification_max_index])
                            identification_contour_areas_arr.append(identification_contour_area)
                    else:
                        defect_rois.append((x1,y1,x2,y2, class_mapping[r['class_ids'][i]]))
                        filename_npz = os.path.join("commodity",str(i)+'.npz')
                        local_path_npz = os.path.join(contour_images_folder,filename_npz)
                        local_path_debug_image = os.path.join(contour_images_folder,"final")+"l1"+str(i)+"."+extension
                        np.savez_compressed(local_path_npz, masks=mask)
                        filename_debug_image = "l1"+str(i)+extension
                        cv2.putText(only_foreground, class_mapping[r['class_ids'][i]], (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.imwrite(os.path.join(contour_images_folder, "final", "l1"+str(i)+"."+extension), only_foreground)

                        if "s3.ap-south-1.amazonaws.com" in original_image_path and args.auto_validator_req == 'yes':
                            S3_commodity_path = self.upload_file_to_s3(local_path_npz, original_image_path, filename_npz)
                            S3_debug_image_path = self.upload_file_to_s3(local_path_debug_image, original_image_path, filename_debug_image)
                        if "s3.ap-south-1.amazonaws.com" in original_image_path and args.auto_validator_req == 'no' and args.bucket_name=="intello-demo-images":
                            S3_commodity_path = self.upload_file_to_s3(local_path_npz, original_image_path, filename_npz,"intello-demo-images")
                            S3_debug_image_path = self.upload_file_to_s3(local_path_debug_image, original_image_path, filename_debug_image,"intello-demo-images")
                        roi = r['rois'][i]
                        score = r['scores'][i]
                        sub_category = class_mapping[r['class_ids'][i]]
                        key1 = key.split('_')
                        ellipse_details = {
                            "reference_x": key1[0],
                            "reference_y": key1[1],
                            "minor_axis": key1[2],
                            "minor_axis_size": key1[3],
                            "major_axis": key1[4],
                            "major_axis_size": key1[5],
                            "angle": key1[6]
                        }

                        image_contours_final.append({
                            "id": str(i),
                            "roi": roi.tolist(),
                            "score": str(int(score * 100)),
                            "area": identification_contour_area,
                            "sub_category": sub_category,
                            "category": configfile['category'],
                            "S3_path": S3_commodity_path,
                            "points": points,
                            "debug_image": S3_debug_image_path,
                            "overall_classification": sub_category,
                            "ellipse_details": ellipse_details
                        })

            # area_median = math.sqrt(area_median)

            cv2.imwrite(os.path.join(contour_images_folder, "output_image.jpg"), output_image)
            if 'IOU' in configfile:
                cv2.imwrite(os.path.join(contour_images_folder, "output_image_mask.jpg"), _mask * 255)

            white_color = (255, 255, 255)
            blue_color = (255, 0, 0)
            green_color = (0, 255, 0)
            red_color = (0, 0, 255)
            black_color = (0, 0, 0)
            size_count = {}
            defect_count = {}
            result = {}
            sub_defect_count = {}
            non_zero_keys = ["defects","sub_defects"]
            category =args.category
            colors = [black_color,blue_color,green_color,red_color]
            if configfile['classification']=='True':
                print("ClassificationConfig"+"="*10)
                #embed()
                model = modellib.MaskRCNN(mode="inference", config=config2, model_dir='')
                weights_version = ""
                if 'weights_version' in configfile['Classification_Config']:
                    weights_version = configfile['Classification_Config']['weights_version']
                weights_path = (os.path.join("/".join(contour_images_folder.split('/')[:-1]), str(args.category) + "_classification" + weights_version + ".h5"))
                print("Weights path: " , weights_path )
                model.load_weights(weights_path, by_name=True)
                defect_count, sub_defect_count, remaining_contours, image_contours, final_result_dict = self.get__defect_count(image_path, contour_details, contour_images_folder, original_image_path, category, colors, '', '', image_contours, ratio, args, model)

        for commodity_box_no in range(len(r['masks'])):
            commodity_s3_paths[commodity_box_no] = ""
            mask = r['masks'][commodity_box_no, :, :]
            filename = os.path.join("commodity", str(commodity_box_no) + '.npz')
            local_path = os.path.join(contour_images_folder, filename)
            np.savez_compressed(local_path, masks=mask)
            if "s3.ap-south-1.amazonaws.com" in original_image_path and args.auto_validator_req == 'yes' :
                s3_commodity_path = self.upload_file_to_s3(local_path, original_image_path, filename)
                commodity_s3_paths[commodity_box_no] = s3_commodity_path

        if args.response == 'mask_rcnn':
            first_contour = True
            for key in image_contours.keys():
                key1 = key.split('_')
                index = image_contours[key]['id']
                roi = r['rois'][index]
                points = image_contours[key]['points']
                score = r['scores'][index]
                if 'defect_sub_cat' not in image_contours[key].keys():
                    image_contours[key]['defect_sub_cat'] = 'normal'
                sub_category = image_contours[key]['defect_sub_cat'].replace("_", " ")
                if 'path' in image_contours[key].keys():
                    S3_debug_image_path = image_contours[key]['path']
                    filename = image_contours[key]['path'].split('/')[-1]
                    if "s3.ap-south-1.amazonaws.com" in original_image_path and args.auto_validator_req == 'yes':
                        S3_debug_image_path = self.upload_file_to_s3(S3_debug_image_path, original_image_path, filename)
                    if "s3.ap-south-1.amazonaws.com" in original_image_path and args.auto_validator_req == 'no' and args.bucket_name=="intello-demo-images":
                        S3_debug_image_path = self.upload_file_to_s3(S3_debug_image_path, original_image_path, filename,"intello-demo-images")

                else:
                    S3_debug_image_path = ''

                ellipse_details = {
                    "reference_x": key1[0],
                    "reference_y": key1[1],
                    "minor_axis": key1[2],
                    "minor_axis_size": key1[3],
                    "major_axis": key1[4],
                    "major_axis_size": key1[5],
                    "angle": key1[6]
                }
                identification_contours, identification_max_index = self.max_area_contour(r['masks'][index, :, :])
                identification_contour_area = 0
                if identification_max_index != -1:
                    identification_contour_area = cv2.contourArea(identification_contours[identification_max_index])
               
                image_contours_final_entry = {
                    "id": str(index),
                    "roi": roi.tolist(),
                    "score": str(int(score * 100)),
                    "area": identification_contour_area,
                    "sub_category": sub_category,
                    "category": configfile['category'].replace("-", "_"),
                    "S3_path": commodity_s3_paths[index],
                    "points": points,
                    "debug_image": S3_debug_image_path,
                    "overall_classification": sub_category,
                    "ellipse_details": ellipse_details
                }

                if first_contour:
                    image_contours_final_entry["fg_bg_ratio"] = foreground_area_percent
                    first_contour = False
                image_contours_final.append(image_contours_final_entry)

            if(args.bucket_name == "intello-demo-images"):
                S3_debug_image_path = self.upload_file_to_s3(contour_images_folder + "/final" + "/debug_defect.jpg", original_image_path, "debug_defect.jpg", "intello-demo-images")
                image_contours_final.append({"full_debug_image": S3_debug_image_path})

            output_file1 = output_file.split('.')[0] + "_commodity.out"
            output_file2 = output_file.split('.')[0] + "_defect.out"

            print("output_file:", output_file)

            with open(output_file1, 'w') as outfile1:
                json.dump(image_contours_final, outfile1)
            if json.loads(configfile['classification'].lower()):
                with open(output_file2, 'w') as outfile2:
                    json.dump(defect_contours_final, outfile2)

        if args.response == 'faster_rcnn':
            result["folder_path"] = original_image_path
            result["category"] = category.replace("-", "_")
            result["defects"] = defect_count
            result["sub_defects"] = sub_defect_count
            result["size"] = size_count
            result["image_contours"] = image_contours
            result["reference_ratio"] = ratio
            result["fg_bg_ratio"] = foreground_area_percent

            non_zero_keys += self.return_non_zero_keys(result["defects"])
            print (non_zero_keys)
            non_zero_keys += self.return_non_zero_keys(result["sub_defects"])
            print (non_zero_keys)
            non_zero_keys += self.return_non_zero_keys(result["size"])
            print (non_zero_keys)
            # if args.environment == "development" and args.auto_validator_req == "yes" and configfile['classification']=='True':
            #    result = write_defects_images_to_s3(os.path.join(contour_images_folder,"final"),args.grade_id,result,final_result_dict)
            if args.environment == "production" and configfile['classification']=='True' and args.auto_validator_req == "yes":
               result =  self.write_defects_images_to_s3(os.path.join(contour_images_folder,"final"),args.grade_id,result,final_result_dict)
            if args.environment == "production" and args.auto_validator_req == "no":
                 self.write_images_to_s3(contour_images_folder,category,args.image,non_zero_keys)
            # elif args.environment == "development" and args.auto_validator_req == "no":
            #      mask_rcnn_grading.write_images_to_folder(contour_images_folder,category,args.image,non_zero_keys)
            with open(output_file, 'w') as outfile:
                    json.dump(result, outfile)
        if args.auto_validator_req == "yes":
            try:
                # pass
                shutil.rmtree(contour_images_folder)
            except:
                "failed to remove contour images folder"

        return contour_details,image_contours
    def upload_file_to_s3(self,local_path,original_image_path,filename,bucket_name="autovalidator"):
    # configuration will be stored in ~/.aws/credentials
        try:
            s3 = boto3.resource('s3')

            s3_details = original_image_path.split(bucket_name+"/")
            s3_host = s3_details[0]
            if bucket_name == "autovalidator":
                s3_path = os.path.join(s3_details[-1], str(args.validation_id), filename)
            else:
                s3_path = os.path.join(s3_details[-1], filename)
            bucket = s3.Bucket(bucket_name)
            bucket.upload_file(local_path, s3_path, ExtraArgs={'ACL':'public-read','ContentType':'image/jpeg'})
            return os.path.join(s3_host,bucket_name,s3_path)
        except:
            print("error in uploading file to s3 :",local_path)
            print(traceback.format_exc())
            return ""
    def get__size(self,crop_img,cont,ratio,number,contour_images_folder):

        xmin=999
        xmax=-1
        ymin=999
        ymax=-1
        center=-1
        for p in range(len(cont)):
            if cont[p][0][0]>xmax:
                xmax=cont[p][0][0]
            if cont[p][0][0]<xmin:
                xmin = cont[p][0][0]
            if cont[p][0][1]>ymax:
                ymax=cont[p][0][1]
            if cont[p][0][1]<ymin:
                ymin = cont[p][0][1]
        width_upper=int((min(xmax-xmin,ymax-ymin))*ratio)

        trendp=-1
        for p in range(len(cont)-10):


            if (xmax-xmin)>(ymax-ymin):
                axis = 0
                edge = (xmax,xmin)
            else :
                axis = 1
                edge = (ymax,ymin)

            inc=0
            dec=0
            counter=1

            while counter<10:
                temp=cont[p+counter-1][0][axis]
                if temp < cont[p+counter][0][axis]:
                    dec+=1
                elif temp > cont[p+counter][0][axis]:
                    inc+=1
                counter+=1
            if inc>=dec:
                trend =1
            elif inc<dec:
                trend = 0

            if trendp != -1 and trend!=trendp:
                center = p+5
                a=(int(cont[center][0][0]),int(cont[center][0][1]))
                a=(int(cont[center][0][0]),int(cont[center][0][1]))
                cv2.putText(crop_img,"c",a,  cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1,cv2.LINE_AA)
                break

            trendp=trend
        last_j=center-1
        j=center
        width=-1
        list_j=[]
        b=True
        for i in range(center,len(cont)):
            if i in list_j:
                break
            if abs(last_j+100)<len(cont) and len(list_j) >=200 and b==True:
                s=last_j+100
                j=last_j+100
            elif abs(last_j+100)<len(cont) and len(list_j) >=200 and b==False:
                s=last_j
                j=last_j
            else:
                s=last_j-1
                j=last_j-1
            w=100000
            if len(list_j)<=200:
                counter=400
            else:
                counter=300

            while j>=0 or j<len(cont):
                if w>distance.euclidean((cont[i][0]),(cont[j][0])):
                    w=distance.euclidean((cont[i][0]),(cont[j][0]))
                    p1=tuple(cont[i][0])
                    p2=tuple(cont[j][0])
                    last_j=j

                if counter!=0:
                    counter-=1
                else :
                    break

                if j>0:
                    j-=1
                else :
                    j= len(cont)-1

            if abs(last_j-s)==0:
                b=False
            else:
                b=True
            if int(w*ratio)==60:
                cv2.line(crop_img,p1,p2,(255,255,255),1)
            elif width<w:
                cv2.line(crop_img,p1,p2,(0,0,0),1)
            else:
                cv2.line(crop_img,p1,p2,(255,0,255),1)

            if last_j not in list_j:
                list_j.append(last_j)

            if width<w and int(w*ratio)<=70:
                width=w
        length=0

        length = int(length*ratio)
        width = int(width*ratio)

        cv2.imwrite(os.path.join(contour_images_folder,"crop_img"+number+".jpg"),crop_img)

        return length,width

    def write_images_to_s3(self,contour_images_folder,category,image_url,non_zero_keys):
        print ( "uploading images")
        print ( contour_images_folder,category,image_url)
        bucket_name = "amazon-now"
        s3_path = image_url.split(bucket_name+"/")[-1]
        s3_path = "/".join(s3_path.split("/")[0:-1])
        for key in non_zero_keys:
            try:
                s3_key = key
                image = s3_key + "_web.jpg"
                local_image = key + "_web.jpg"
                upload_image(os.path.join(contour_images_folder,local_image), bucket_name, os.path.join(s3_path,image))
                image = s3_key + ".jpg"
                local_image = key + ".jpg"
                upload_image(os.path.join(contour_images_folder,local_image), bucket_name, os.path.join(s3_path,image))
            except:
                print(traceback.print_exc())
                pass

    def write_defects_images_to_s3(self, contour_images_folder, model_id, result, final_result_dict):
        print ("uploading images")
        bucket_name = "autovalidator"
        s3_path = "production/uploads/vegetable_grading/" + str(model_id)+"/contour_defects"
        for key in result["image_contours"]:
            filename = final_result_dict[key].split(";")[0].split("/")[-1].split(".")[0]+"_op.jpg"
            result["image_contours"][key]['path'] = os.path.join(s3_path, filename)

        print(result["image_contours"])

        for filename in os.listdir(contour_images_folder):
            if filename.endswith(".jpg") or filename.endswith(".JPG"):
                local_image = os.path.join(contour_images_folder,filename)
                server_image = os.path.join(s3_path,filename)
                self.upload_defects_image(os.path.join(contour_images_folder,filename), bucket_name, os.path.join(s3_path,filename),model_id)
        return result

    def get_category_specific_folders(self, category, args, contour_images_folder):
        sub_folders = ["major", "normal", "minor", "ignore", 'commodity']
        if category == args.category:
            sub_folders += ["final"]
            if configfile['classification'] == 'True':
                defects = configfile['trained_defects']
                for defect in defects.keys():
                    folder = list(defects[defect]['actual_defect'].keys())[0]
                    sub_folders.append(folder)
        return sub_folders

    def return_non_zero_keys(self, res_dict):
        keys = res_dict.keys()
        result = []
        for key in keys:
            if res_dict[key] != 0:
                result.append(key)
        return result

    def write_images_to_folder(self, contour_images_folder, category, image_path, non_zero_keys):
        print ("writing images")
        print (contour_images_folder, category, image_path)
        folder_path = "/".join(image_path.split("/")[0:-1])
        print ("folder_path : ", folder_path)
        non_zero_keys = [x for x in non_zero_keys if x != "ignore"]
        for key in non_zero_keys:
            try:
                image = key + ".jpg"
                os.system("cp " + os.path.join(contour_images_folder,image) + " " + os.path.join(folder_path,image))
                image = key + "_web.jpg"
                os.system("cp " + os.path.join(contour_images_folder,image) + " " + os.path.join(folder_path,image))
            except:
                print(traceback.print_exc())
                pass

    def upload_defects_image(self, local_path, bucket_name, s3_path, model_id):
        print(local_path, bucket_name, s3_path)
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        print(bucket)
        print(s3_path)

        client = boto3.client('s3')

        response = client.put_object(
            Bucket='autovalidator',
            Body='',
            Key='production/uploads/vegetable_grading/'+str(model_id)+"/contour_defects/"
            )

        bucket.upload_file(local_path, s3_path, ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/jpeg'})

    def download_image(self, path, image_url):
        filename = image_url.split("/")[-1]
        image_path = os.path.join(path, filename)
        try:
            urllib.request.urlretrieve(image_url, image_path)
        except:
            print("error :", traceback.format_exc())
        return image_path

    def create_folders(self, paths, args):
        print(paths)
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)


if __name__ == '__main__':
    weights_path = os.path.join(category_path, configfile['weights_path'])

    if args.command == "splash":
        assert args.image, "Provide --image or --video to apply color splash"

    print("Weights: ", weights_path)

    config = CustomConfig()
    print("IdentificationConfig" + "=" * 10)
    config.display()

    class ClassificationConfig(Config):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BATCH_SIZE=4
        #config2=len(image_contours.keys())
        NAME = configfile['category']
        if 'Classification_Config' in configfile:
            configfile_1 = configfile['Classification_Config']
            BACKBONE = configfile_1['BACKBONE']
            NUM_CLASSES = configfile_1['NUM_CLASSES']
            STEPS_PER_EPOCH = configfile_1['STEPS_PER_EPOCH']
            if 'IMAGE_MIN_DIM' in configfile_1.keys():
                IMAGE_MIN_DIM = configfile_1['IMAGE_MIN_DIM']
            if 'IMAGE_MAX_DIM' in configfile_1.keys():
                IMAGE_MAX_DIM = configfile_1['IMAGE_MAX_DIM']
            if 'TRAIN_ROIS_PER_IMAGE' in configfile_1.keys():
                TRAIN_ROIS_PER_IMAGE = configfile_1['TRAIN_ROIS_PER_IMAGE']
            if 'DETECTION_MIN_CONFIDENCE' in configfile_1.keys():
                DETECTION_MIN_CONFIDENCE = configfile_1['DETECTION_MIN_CONFIDENCE']
            if 'MAX_GT_INSTANCES' in configfile_1.keys():
                MAX_GT_INSTANCES = configfile_1['MAX_GT_INSTANCES']
            if 'DETECTION_MAX_INSTANCES' in configfile_1.keys():
                DETECTION_MAX_INSTANCES = configfile_1['DETECTION_MAX_INSTANCES']
            if 'DETECTION_NMS_THRESHOLD' in configfile_1.keys():
                DETECTION_NMS_THRESHOLD = configfile_1['DETECTION_NMS_THRESHOLD']
            if 'RPN_TRAIN_ANCHORS_PER_IMAGE' in configfile_1.keys():
                RPN_TRAIN_ANCHORS_PER_IMAGE = configfile_1['RPN_TRAIN_ANCHORS_PER_IMAGE']
            if 'PRE_NMS_LIMIT' in configfile_1.keys():
                PRE_NMS_LIMIT = configfile_1['PRE_NMS_LIMIT']
            if 'POST_NMS_ROIS_TRAINING' in configfile_1.keys():
                POST_NMS_ROIS_TRAINING = configfile_1['POST_NMS_ROIS_TRAINING']
            if 'POST_NMS_ROIS_INFERENCE' in configfile_1.keys():
                POST_NMS_ROIS_INFERENCE = configfile_1['POST_NMS_ROIS_INFERENCE']
            if 'BACKBONE_STRIDES' in configfile_1.keys():
                BACKBONE_STRIDES = configfile_1['BACKBONE_STRIDES']
            if 'RPN_ANCHOR_SCALES' in configfile_1.keys():
                RPN_ANCHOR_SCALES = tuple(configfile_1['RPN_ANCHOR_SCALES'])
            if 'MASK_THRESHOLD' in configfile_1.keys():
                MASK_THRESHOLD = configfile_1['MASK_THRESHOLD']
            if 'ROI_POSITIVE_RATIO' in configfile_1.keys():
                ROI_POSITIVE_RATIO = configfile_1['ROI_POSITIVE_RATIO']
    if json.loads(configfile['classification'].lower()):
        #embed()
        config2 = ClassificationConfig()
        print("we are about to display config>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        config2.display()

    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir='')

    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    print(script_path)
    contour_images_folder = os.path.join(category_path, args.grade_id)
    if args.output_path is None:
        args.output_path = os.path.join(contour_images_folder, "output.out")
    m = mask_rcnn_grading()
    sub_folders = m.get_category_specific_folders(args.category, args, contour_images_folder)
    sub_folder_paths = [os.path.join(contour_images_folder, folder_name) for folder_name in sub_folders]

    paths = [client_path, category_path, contour_images_folder] + sub_folder_paths
    for folder in sub_folder_paths:
        print(folder)
        paths.append(folder)
    m.create_folders(paths, args)
    if "s3.ap-south-1.amazonaws.com" in image or "local-images" in image:
        image = m.download_image(contour_images_folder, image)

    print("input image:", args.image)
    print("local image:", image)
    print("category:", args.category)
    print("output folder:", contour_images_folder)
    print("output json:", args.output_path)

    if args.command == "splash":
        contour_details, image_contours = m.detect_and_color_splash(model, contour_images_folder, args.output_path, image, original_image_path, args)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
