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
ap.add_argument("-c", "--class", required=True, help="object class to search for as per http://cocodataset.org/")
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

def show_objects(image, out_boxes, out_scores, out_classes):
    
    # Set up some display formatting
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Plot the image
    img = np.array(image)
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(img)

    # Set up padding for boxes
    img_size = model_image_size[0]
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    # Use a random color for each class
    unique_labels = np.unique(out_classes)
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)

    # process each instance of each class that was found
    for i, c in reversed(list(enumerate(out_classes))):

        # Get the class name
        predicted_class = class_names[c]
        # Get the box coordinate and probability score for this instance
        box = out_boxes[i]
        score = out_scores[i]

        # Format the label to be added to the image for this instance
        label = '{} {:.2f}'.format(predicted_class, score)

        # Get the box coordinates
        top, left, bottom, right = box
        y1 = max(0, np.floor(top + 0.5).astype('int32'))
        x1 = max(0, np.floor(left + 0.5).astype('int32'))
        y2 = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        x2 = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        # Set the box dimensions
        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
        
        # Add a box with the color for this class
        color = bbox_colors[int(np.where(unique_labels == c)[0])]
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        plt.text(x1, y1, s=label, color='white', verticalalignment='top',
                bbox={'color': color, 'pad': 0})
        
    plt.axis('off')
    plt.show()

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

    # Detect objects in the image
    out_boxes, out_scores, out_classes = detect_objects(image)

    imgcnt=imgcnt+1
    objects=[class_names[out_classes[i]] for i in range(len(out_classes))]
    if (args["class"] in objects):
        target_path = os.path.join(args["outdir"], image_file)
        print("Moving ", img_path, " to ", target_path)
        os.rename(img_path, target_path)
    else:
        print(img_path, ": No Person found")