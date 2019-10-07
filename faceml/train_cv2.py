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
from cv2util import *
from util import *

recognizer_file = "recognizer_cv2.pickle"
labelencoder_file = "labelencoder_cv2.pickle"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--traindir", required=True, help="path to input directory of images for training")
    ap.add_argument("-v", "--valdir", required=True, help="path to input directory of images for training")
    ap.add_argument("-p", "--margin", required=False,  nargs='?', const=0, type=int, default=0, help="margin percentage pixels to include around the face")
    ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store trained model files")
    ap.add_argument("-l", "--logdir", required=False, default="", help="path to log directory")
    ap.add_argument("-g", "--loglevel", required=False, default="INFO", dest='log_level', type=log_level_string_to_int, help="log level: DEBUG, INFO, ERROR. Default INFO")

    return  vars(ap.parse_args())
    

# extract a single face from a given photograph
def extract_all_faces(model,filename, margin):
    logger = getMyLogger()
    logger.info(filename)
    x1,y1,x2,y2 = list(),list(),list(),list()
    faces=list()
    image = load_image(filename)
    if (image is None):
        return (None,)*5
    h,w=image.shape[:2]    
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
                x1.append(startX)
                y1.append(startY)
                x2.append(endX)
                y2.append(endY)
                face_array = asarray(face)
                faces.append(face_array)
        return  x1, y1, x2, y2, faces
    else:
        logger.info("no detections")
        return (None,)*5

# extract a single face from a given photograph
def extract_face(model,filename, margin):    
    (x1,y1,x2,y2,faces) = extract_all_faces(model, filename, margin)
    if (x1 is None or len(x1)==0):
        return (None,)*5
    return x1[0],y1[0],x2[0],y2[0],faces[0]

def load_faces(model, directory, margin):
    faces = list()
    # enumerate files
    for filename in os.listdir(directory):
        # path
        path = directory + filename
        # get face
        x1,y1,x2,y2,face = extract_face(model, path, margin)
        if face is not None:
            # store
            faces.append(face)
    return faces

def load_dataset(model, directory, margin):
    X, y = list(), list()
    logger = getMyLogger()
    knownEmbeddings = []
    knownNames = []
    embedder = load_embedder()
    # enumerate folders, on per class
    for subdir in os.listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(model,path, margin)
        if (len(faces)>0):
            # create labels
            labels = [subdir for _ in range(len(faces))]
            # summarize progress
            logger.info('loaded %d examples for class: %s' % (len(faces), subdir))
            # store
            for face in range(len(faces)):
                faceBlob = cv2.dnn.blobFromImage(faces[face], 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(labels[face])
                knownEmbeddings.append(vec.flatten())

    return knownNames, knownEmbeddings

def main(args):
    logger =  getLogger(args["logdir"], args["log_level"], logfile)

    model =load_caffe_model()
    trainY, trainX = load_dataset(model,args["traindir"], int(args["margin"]))
    logger.info(asarray(trainX).shape)
    testY, testX = load_dataset(model,args["valdir"], int(args["margin"]))
    logger.info(asarray(testX).shape)
    
    le = LabelEncoder()
    trainY = le.fit_transform(trainY)
    testY = le.fit_transform(testY)

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(trainX, trainY)

    # write the actual face recognition model to disk
    f = open(args["outdir"] + "/" + recognizer_file, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(args["outdir"] + "/" + labelencoder_file, "wb")
    f.write(pickle.dumps(le))
    f.close()

    yhat_train = recognizer.predict(trainX)
    yhat_test = recognizer.predict(testX)
    # score
    score_train = accuracy_score(trainY, yhat_train)
    score_test = accuracy_score(testY, yhat_test)
    # summarize
    logger.info('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

if __name__ == '__main__':
    args = parse_args()
    main(args)