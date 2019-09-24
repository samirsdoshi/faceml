import sys
import keras
import os
from os import listdir
from os.path import isdir
from PIL import Image
from keras.models import load_model
import numpy as np
import pickle
import argparse
from numpy import asarray
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from numpy import expand_dims
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from yolo_keras.utils import *
from yolo_keras.model import *
from kerasutil import *
from yoloutil import *
from util import *
import logging

recognizer_file = "recognizer_keras.pickle"
labelencoder_file = "labelencoder_keras.pickle"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagesdir", required=True, help="path to input directory of images")
    ap.add_argument("-m", "--modelpath", required=True, help="directory with trained model")
    ap.add_argument("-c", "--person", required=True, help="person name expression to filter -  <name1> | not [name1] | [name1] and [name2] | [name1] and ([name2] or [name3]) and not [name4] etc")
    ap.add_argument("-p", "--margin", required=False,  nargs='?', const=0, type=int, default=0, help="margin percentage pixels to include around the face")
    ap.add_argument("-k", "--confidence", required=False, nargs='?', const=70, type=int, default=70, help="minimum confidence percentage for face recognition. Default 70")
    ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store images having filter class")
    ap.add_argument("-l", "--logdir", required=False, default="", help="path to log directory")
    return vars(ap.parse_args())


# extract all faces from a given photograph
def extract_all_faces(detector, pixels, margin, required_size=(160, 160)):
    logger = getMyLogger()
    x1,y1,x2,y2 = list(),list(),list(),list()
    faces=list()
    # detect faces in the image
    results = detector.detect_faces(pixels)
   # extract the bounding box from the first face
    if (len(results) > 0):
        for i in range(len(results)):
            startX=results[i]['box'][0]
            startY=results[i]['box'][1]
            endX=startX + results[i]['box'][2]
            endY= startY + results[i]['box'][3]
            #x1[i], y1[i], width, height = results[i]['box']
            # bug fix
            startX,startY = abs(startX), abs(startY)
            if margin > 0:
                startX,startY,endX,endY = addMargin(startX,startY,endX,endY,margin)
            face = pixels[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 10 or fH < 10:
                continue
            x1.append(startX)
            y1.append(startY)
            x2.append(endX)
            y2.append(endY)
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            faces.append(face_array)
        return x1, y1, x2, y2, faces, pixels
    else:
        return None,None,None,None,None,pixels

#retry prediction with different margins around the face.
def retryPred(x1,y1,x2,y2,pixels,model,in_encoder,recognizer, required_size=(160, 160)):
    maxProb=0
    max_yhat_class=[]
    max_yhat_prob=[]
    for margin in (0,1,3,5):
        startX,startY,endX,endY = addMargin(x1,y1,x2,y2,margin)
        face = pixels[startY:endY,startX:endX]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        embedding = get_embedding(model,face_array)
        samples = in_encoder.transform(embedding.reshape(1,-1))
        yhat_class = recognizer.predict(samples)
        yhat_prob = recognizer.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        if class_probability > maxProb:
            max_yhat_prob=yhat_prob
            max_yhat_class=yhat_class
            maxProb = class_probability
        if class_probability > 95:
            break
    return max_yhat_class, max_yhat_prob

def main(args):
    logger = getLogger(args["logdir"], logfile)

    class_names, anchors, yolo_model = open_yolo_model()
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(yolo_model.output, anchors, len(class_names), input_image_shape,
                                        score_threshold=0.3, iou_threshold=0.45)

    model =load_keras_model()
    detector = MTCNN()
    recognizer = pickle.loads(open(args["modelpath"] + recognizer_file, "rb").read())
    out_encoder = pickle.loads(open(args["modelpath"] + labelencoder_file, "rb").read())
    model_image_size = (416, 416)
    filesmoved=0
    in_encoder = Normalizer(norm='l2')

    classexpr=args["person"].lower()
    search_classes, not_classes, expr = processClassName(classexpr)
    for image_file in os.listdir(args["imagesdir"]):    
        # Load image
        img_path = os.path.join(args["imagesdir"], image_file)
        logger.debug(img_path)
        try:
            orgimage = Image.open(img_path)
        except Exception as e:
            logger.error("Error loading " + img_path)   
            continue

        #scale image
        image, scale, xpad, ypad= scale_image(orgimage, tuple(reversed(model_image_size)))
        out_boxes, out_scores, out_classes = detect_objects(image, boxes, scores, classes,yolo_model,input_image_shape)
        objects=[class_names[out_classes[i]] for i in range(len(out_classes))]
        
        #get enclosing area and transpose on original image
        px1, py1, px2, py2 = getEnclosingArea(objects, "[person]", out_boxes)
        newX1, newY1, newX2, newY2 = transpose(px1,py1,px2,py2,xpad,ypad,scale)
        personimage = asarray(orgimage)[newY1:newY2, newX1:newX2]
        
        x1, y1, x2, y2, faces, pixels = extract_all_faces(detector, personimage, int(args["margin"]))
        if (x1 is None):
            continue
        logger.debug( "candidate classes found:", len(x1))

        for i in range(len(faces)):
            yhat_class, yhat_prob = retryPred(x1[i],y1[i],x2[i],y2[i],pixels,model,in_encoder,recognizer)
            logger.debug( yhat_class, yhat_prob)
            class_index = yhat_class[0]
            proba = yhat_prob[0,class_index]
            predict_names = out_encoder.inverse_transform(yhat_class)
            logger.debug(predict_names, " ",proba)
            name = predict_names[0]
            if (proba*100 >=args["confidence"]):
                objects.append(name)
        
        if(eval(expr)):
            logger.info("MATCH:",img_path)
            if args["outdir"]!="":
                setupDir(args["outdir"])
                target_path = os.path.join(args["outdir"], image_file)
                logger.info("Moving ", img_path, " to ", target_path)
                os.rename(img_path, target_path)
                filesmoved=filesmoved+1
        else:
            logger.info("NO MATCH:",img_path)

if __name__ == '__main__':
    args=parse_args()
    main(args)