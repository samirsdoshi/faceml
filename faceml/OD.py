import os
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo_keras.utils import *
from yolo_keras.model import *
from PIL import Image

import argparse
import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt


classes_path = "/faceml/yolo_keras/coco_classes.txt"
anchors_path = "/faceml/yolo_keras/yolo_anchors.txt"
model_path="/faceml/yolo_keras/yolo.h5"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagedir", required=True, help="path to input directory of images")
ap.add_argument("-c", "--class", required=True, help="object class to search for as per yolo_keras/coco_classes.txt")
ap.add_argument("-k", "--confidence", required=False, nargs='?', const=80, type=int, default=80, help="minimum confidence percentage for object detection. Default 80")
ap.add_argument("-s", "--size", required=False, nargs='?', const=0, type=float, default=0, help="minimum percentage size of the object in the image.")
ap.add_argument("-n", "--count", required=False, nargs='?', const=0, type=int, default=0, help="filter images containing X count of class object")
ap.add_argument("-o", "--outdir", required=True, help="path to output directory with images having search objects")
args = vars(ap.parse_args())

# Get the COCO classes on which the model was trained

with open(classes_path) as f:
    class_names = f.readlines()
    class_names = [c.strip() for c in class_names] 
num_classes = len(class_names)

# Get the anchor box coordinates for the model

with open(anchors_path) as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
num_anchors = len(anchors)

# Set the expected image size for the model
model_image_size = (416, 416)


# Create YOLO model
#home = os.path.expanduser("~")
#model_path = os.path.join(home, "yolo.h5")
yolo_model = load_model(model_path, compile=False)

# Generate output tensor targets for bounding box predictions
# Predictions for individual objects are based on a detection probability threshold of 0.3
# and an IoU threshold for non-max suppression of 0.45
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = yolo_eval(yolo_model.output, anchors, len(class_names), input_image_shape,
                                    score_threshold=0.3, iou_threshold=0.45)

def detect_objects(image):
    
    # normalize and reshape image data
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    # Predict classes and locations using Tensorflow session
    sess = K.get_session()
    out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: image_data,
                    input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
    return out_boxes, out_scores, out_classes

imgcnt=1
for image_file in os.listdir(args["imagedir"]):
    
    # Load image
    img_path = os.path.join(args["imagedir"], image_file)
    try:
        image = Image.open(img_path)
    except:
        print("Error loading " + img_path)
        continue

    # Resize image for model input
    image = letterbox_image(image, tuple(reversed(model_image_size)))
    iw, ih = image.size
    image_area=iw*ih

    # Detect objects in the image
    out_boxes, out_scores, out_classes = detect_objects(image)

    imgcnt=imgcnt+1
    classname=args["class"]
    requiredCount=int(args["count"])
    requiredConfidence=int(args["confidence"])
    requiredSize=args["size"]
    objects=[class_names[out_classes[i]] for i in range(len(out_classes))]

    if classname!="" and (classname in objects):
        match=0
        index=0
        filter_boxes, filter_scores, filter_objects =[],[],[]
        for i in range(len(out_classes)):
            if (objects[i]==classname and out_scores[i]*100>=requiredConfidence):
                    index=i
                    box_height=(out_boxes[i][2]-out_boxes[i][0])
                    box_width=(out_boxes[i][3]-out_boxes[i][1])
                    box_area=box_width*box_height
                    if ((box_area/image_area)*100 > requiredSize):
                        match=match+1
                        filter_boxes.append(out_boxes[i])
                        filter_scores.append(out_scores[i])
                        filter_objects.append(out_classes[i])

        print(img_path, ": Found ", match, " count of ", args["class"], " objects")
        if (match > 0 and (requiredCount==0 or match==requiredCount)):        
            target_path = os.path.join(args["outdir"], image_file)
            print("Moving ", img_path, " to ", target_path)
            os.rename(img_path, target_path)
    else:
        print(img_path, ": No ", args["class"]," found")