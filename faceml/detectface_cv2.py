import imutils
import sys
import os
from os import listdir
from os.path import isdir
from PIL import Image
import numpy as np
import cv2
import pickle
import argparse
from numpy import asarray
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from util import *
from cv2util import *
import logging

recognizer_file = "recognizer_cv2.pickle"
labelencoder_file = "labelencoder_cv2.pickle"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagesdir", required=True, help="path to input directory of images")
    ap.add_argument("-m", "--modelpath", required=True, help="directory with trained model")
    ap.add_argument("-c", "--class", required=True, help="class name to filter (class1,class2,...)")
    ap.add_argument("-p", "--margin", required=False,  nargs='?', const=0, type=int, default=0, help="margin percentage pixels to include around the face")
    ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store images having filter class")
    ap.add_argument("-l", "--logdir", required=False, default="", help="path to log directory")
    return vars(ap.parse_args())

    
# extract a single face from a given photograph
def extract_all_faces(model,filename, margin):
    logger = getMyLogger()
    logger.debug(filename)
    x1,y1,x2,y2 = list(),list(),list(),list()
    faces=list()
    (h,w,image) = load_image(filename)
    if (image is None):
        return None,None,None,None,None,None
    blob = blob_from_image(image)
    model.setInput(blob)
    detections = model.forward()
    if (len(detections) > 0):
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if margin > 0:
                    startX,startY,endX,endY = addMargin(startX,startY,endX,endY,margin)
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 10 or fH < 10:
                    continue
                x1.append(startX)
                y1.append(startY)
                x2.append(endX)
                y2.append(endY)
                face_array = asarray(face)
                faces.append(face_array)
        return  x1, y1, x2, y2, faces, image
    else:
        logger.debug("no detections")
        return None,None,None,None,None,image

#retry prediction with different margins around the face.
def retryPred(x1,y1,x2,y2,pixels,embedder,recognizer):
    maxProb=0
    max_yhat_prob=[]
    for margin in (0,1,3,5):
        startX,startY,endX,endY = addMargin(x1,y1,x2,y2,margin)
        face = pixels[startY:endY,startX:endX]
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
            (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()
        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        if proba > maxProb:
            max_yhat_prob=preds
            maxProb=proba
        if proba > 0.95:
            break
    return max_yhat_prob

def main(args):
    logger = getLogger(args["logdir"], logfile)

    model =load_caffe_model()
    embedder =load_embedder()
    recognizer = pickle.loads(open(args["modelpath"] + recognizer_file, "rb").read())
    le = pickle.loads(open(args["modelpath"] + labelencoder_file, "rb").read())
    
    result=dict()
    classes=args["class"].split(",")
    for i in range(len(classes)):
        result[classes[i]]=0

    imgcnt=1
    filesmoved=0
    for image_file in os.listdir(args["imagesdir"]):    
        # Load image
        img_path = os.path.join(args["imagesdir"], image_file)
        x1, y1, x2, y2, faces, pixels = extract_all_faces(model, img_path, int(args["margin"]))
        if (x1 is None):
            continue
        logger.debug( "candidate classes found:", len(x1))
        for i in range(len(x1)):
            preds = retryPred(x1[i],y1[i],x2[i],y2[i],pixels,embedder,recognizer)
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            logger.debug(preds, name, proba)
            if (proba >=0.80 and name in classes):
                logger.debug( i, "MATCH:",img_path, name, proba)
                target_path = os.path.join(args["outdir"], image_file)
                logger.debug( "Moving ", img_path, " to ", target_path)
                os.rename(img_path, target_path)
                filesmoved=filesmoved+1
                result[name]=result[name]+1
                break
            else:
                logger.debug( i, "NO MATCH:",img_path, name, proba)

    for i in range(len(classes)):
        logger.debug( classes[i]," detected in ", result[classes[i]], " files")

if __name__ == '__main__':
    args =  parse_args()
    main(args)