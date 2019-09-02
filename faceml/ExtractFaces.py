import sys
import keras
import os
from os import listdir
from os.path import isdir
from PIL import Image
from keras.models import load_model
import numpy as np
import cv2
import argparse
from numpy import asarray
from util import *


protext_path = "/faceml/opencv/deploy.prototxt"
model_path = "/faceml/opencv/res10_300x300_ssd_iter_140000.caffemodel"
embedder_path = "/faceml/opencv/openface_nn4.small2.v1.t7"
recognizer_file = "recognizer.pickle"
labelencoder_file = "labelencoder.pickle"
logfile="faceml.log"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagesdir", required=True, help="path to input directory of images")
ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store face images")
ap.add_argument("-p", "--margin", required=False,  nargs='?', const=0, type=int, default=0, help="margin percentage pixels to include around the face")
ap.add_argument("-l", "--logdir", required=False, default="", help="path to log directory")

args = vars(ap.parse_args())

def load_facenet_model():
    # load the model
    return cv2.dnn.readNetFromCaffe(protext_path, model_path)

def load_image(filename):
    try:
        image = cv2.imread(filename)
        (h, w) = image.shape[:2]
        return h,w,image
    except:
        print("Error loading " + filename)
        return (None,)*3

def blob_from_image(imagearray):
    return cv2.dnn.blobFromImage(cv2.resize(imagearray, (300, 300)), 1.0, (300, 300))
    

# extract a single face from a given photograph
def extract_all_faces(model,filename,margin):
    x1,y1,x2,y2 = list(),list(),list(),list()
    faces=list()
    (h,w,image) = load_image(filename)
    if (image is None):
        return None
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
                startX = int(startX - (startX*margin/100))
                endX = int(endX + (endX*margin/100))
                startY = int(startY - (startY*margin/100))
                endY = int(endY + (endY*margin/100))
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                face_array = asarray(face)
                faces.append(face_array)
        return  faces
    else:
        print("no detections")
        return None


model =load_facenet_model()
imgcnt=1
if (args["logdir"]==""):
    flog=sys.stdout
else:    
    flog=openfile(args["logdir"] + "/" + logfile)
for image_file in os.listdir(args["imagesdir"]):    
    facesfound=0
    # Load image
    img_path = os.path.join(args["imagesdir"], image_file)
    writelog(flog, img_path)
    faces = extract_all_faces(model, img_path, int(args["margin"]))
    if (faces is None):
        continue
    for i in range(len(faces)):
        im = Image.fromarray(faces[i])
        im.save(args["outdir"] + "/" + str(i) + "_" + image_file)
        facesfound=facesfound+1
    writelog(flog, "Faces found:", facesfound)

flog.close()    