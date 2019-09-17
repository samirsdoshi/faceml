import os
import sys
import numpy as np

from keras.applications import InceptionV3
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo_keras.utils import *
from yolo_keras.model import *
from PIL import Image

import argparse
from util import *
import logging


def detect_objects(image, boxes, scores, classes, yolo_model, input_image_shape):
    
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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--imagedir", required=True, help="path to input directory of images")
    ap.add_argument("-c", "--class", required=False,  default="", help="object class to filter as per yolo_keras/coco_classes.txt")
    ap.add_argument("-t", "--threshold", required=False, nargs='?', const=70, type=int, default=70, help="minimum confidence percentage for similarity. Default 70")    
    ap.add_argument("-o", "--outdir", required=False, default="", help="path to output directory where selected images will be moved")
    ap.add_argument("-l", "--logdir", required=False, default="", help="path to log directory")
    return vars(ap.parse_args())

def open_model():
    anchors_path = "/faceml/yolo_keras/yolo_anchors.txt"
    classes_path = "/faceml/yolo_keras/coco_classes.txt"
    model_path="/faceml/yolo_keras/yolo.h5"

    # Get the anchor box coordinates for the model

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    # Get the COCO classes on which the model was trained

    with open(classes_path) as f:
        class_names = f.readlines()
        class_names = [c.strip() for c in class_names] 

    # Create YOLO model
    #home = os.path.expanduser("~")
    #model_path = os.path.join(home, "yolo.h5")
    yolo_model = load_model(model_path, compile=False)

    # Generate output tensor targets for bounding box predictions
    # Predictions for individual objects are based on a detection probability threshold of 0.3
    # and an IoU threshold for non-max suppression of 0.45

    return class_names, anchors,  yolo_model


def getBoxAreas(objects, out_classes, classname, out_scores, out_boxes, image_area):
    areas = []
    for i in range(len(out_classes)):
        if (classname=="" or objects[i]==classname):
            box_height=(out_boxes[i][2]-out_boxes[i][0])
            box_width=(out_boxes[i][3]-out_boxes[i][1])
            box_area=box_width*box_height
            areas.append(box_area)
    return areas


def keras_load_image(imgfile):
    img = image.load_img(imgfile, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

def cosine_similarity(ratings):
    sim = ratings.dot(ratings.T)
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


def processImage(classname, imgfile, boxes, scores, classes,yolo_model, class_names, input_image_shape):
    image = Image.open(imgfile)
    model_image_size = (416, 416)
    image = letterbox_image(image, tuple(reversed(model_image_size)))
    iw, ih = image.size
    image_area=iw*ih

    out_boxes, out_scores, out_classes = detect_objects(image, boxes, scores, classes,yolo_model,input_image_shape)
    objects=[class_names[out_classes[i]] for i in range(len(out_classes))]
    if (classname=="" or classname in objects):
        img_areas = getBoxAreas(objects, out_classes, classname, out_scores, out_boxes, image_area)
        #load image for vgg
        img_x = keras_load_image(imgfile)
        return img_areas, img_x

    return None, None

def compareImages(model, img_areas1, img_areas2, x):
    x=asarray(x)
    x = preprocess_input(x)
    features = model.predict(x)
    fshape = features.shape
    features_compress = features.reshape(2, fshape[1] * fshape[2] * fshape[3])
    cos_sim = cosine_similarity(features_compress)
    return cos_sim[0,1]

def main(args):
    
    logger = getLogger(args["logdir"], logfile)

    # Set the expected image size for the model
    class_names, anchors, yolo_model = open_model()
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(yolo_model.output, anchors, len(class_names), input_image_shape,
                                        score_threshold=0.3, iou_threshold=0.45)

    classname=args["class"]
    img_areas, img_x, filenames =[],[],[]
    filecnt=1
    for image_file in os.listdir(args["imagedir"]):
        # Load image
        img_path = os.path.join(args["imagedir"], image_file)
        logger.info("Processing file ",filecnt, " " ,img_path)
        filecnt=filecnt+1
        try:
            areas, x = processImage(classname, img_path, boxes, scores, classes,yolo_model, class_names, input_image_shape)
            if (areas is not None):
                img_areas.append(areas)
                if (img_x==[]):
                    img_x=x
                else:
                    img_x = np.concatenate((img_x, x))
                filenames.append(image_file)
        except Exception as e:
            logger.error(str(e))
            continue

    x=[]
    similarfiles={}
    filestoskip=[]        
    model = InceptionV3(weights='imagenet', include_top=False)
    for i in range(len(filenames)-1):
        if filenames[i] in similarfiles:
            continue
        img_areas1 = img_areas[i]
        x=img_x[i]
        for j in range(i+1, len(filenames)):
            if filenames[j] in filestoskip:
                continue
            img_areas2 = img_areas[j]
            sim = compareImages(model, img_areas1, img_areas2, [x, img_x[j]])
            if sim*100 >= args["threshold"]:
                logger.info(filenames[i], " ",filenames[j], " are ", sim*100, " percent similar")
                if not (filenames[i] in similarfiles):
                    similarfiles[filenames[i]]=[filenames[i]]
                similarfiles[filenames[i]].append(filenames[j])   
                filestoskip.append(filenames[j]) 

    if(args["outdir"]!=""):
        setupDir(args["outdir"])
        for orgfile in similarfiles:
            newOutDir = args["outdir"] + "/" + os.path.splitext(orgfile)[0]
            setupDir(newOutDir)
            for filetomove in similarfiles[orgfile]:
                source_path = os.path.join(args["imagedir"], filetomove)
                target_path = os.path.join(newOutDir, filetomove)
                logger.info("Moving ", source_path, " to ", target_path)
                os.rename(source_path, target_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)