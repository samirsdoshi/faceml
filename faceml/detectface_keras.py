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
from kerasutil import *
from util import *
import logging

recognizer_file = "recognizer_keras.pickle"
labelencoder_file = "labelencoder_keras.pickle"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagesdir", required=True, help="path to input directory of images")
    ap.add_argument("-m", "--modelpath", required=True, help="directory with trained model")
    ap.add_argument("-c", "--class", required=True, help="class name to filter (class1,class2,...)")
    ap.add_argument("-p", "--margin", required=False,  nargs='?', const=0, type=int, default=0, help="margin percentage pixels to include around the face")
    ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store images having filter class")
    ap.add_argument("-l", "--logdir", required=False, default="", help="path to log directory")
    return vars(ap.parse_args())


# extract all faces from a given photograph
def extract_all_faces(detector, filename, margin, required_size=(160, 160)):
    logger = getMyLogger()
    logger.debug(filename)
    x1,y1,x2,y2 = list(),list(),list(),list()
    faces=list()
    # load image from file
    # convert to array
    pixels = load_image(filename)
    if (pixels is None):
        return None,None,None,None,None,None
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
            #logger.debug(i, x,y,width,height,fH,fW)
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


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    #print("samples:",samples.shape)
    # make prediction to get embedding
    yhat = model.predict(samples)
    #print("yhat:",yhat.shape, yhat[0])
    return yhat[0]    

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

    model =load_keras_model()
    detector = MTCNN()
    recognizer = pickle.loads(open(args["modelpath"] + recognizer_file, "rb").read())
    out_encoder = pickle.loads(open(args["modelpath"] + labelencoder_file, "rb").read())

    imgcnt=1
    filesmoved=0
    in_encoder = Normalizer(norm='l2')
    result=dict()
    classes=args["class"].split(",")
    for i in range(len(classes)):
        result[classes[i]]=0
    for image_file in os.listdir(args["imagesdir"]):    
        # Load image
        img_path = os.path.join(args["imagesdir"], image_file)
        x1, y1, x2, y2, faces, pixels = extract_all_faces(detector, img_path, int(args["margin"]))
        if (x1 is None):
            continue
        logger.debug( "candidate classes found:", len(x1))
        for i in range(len(faces)):
            yhat_class, yhat_prob = retryPred(x1[i],y1[i],x2[i],y2[i],pixels,model,in_encoder,recognizer)
            logger.debug( yhat_class, yhat_prob)
            class_index = yhat_class[0]
            proba = yhat_prob[0,class_index]
            predict_names = out_encoder.inverse_transform(yhat_class)

            logger.debug(predict_names, proba)

            name = predict_names[0]
            if (proba >=0.80 and name in classes):
                logger.debug(i, "MATCH:",img_path, name, proba)
                target_path = os.path.join(args["outdir"], image_file)
                logger.debug("Moving ", img_path, " to ", target_path)
                os.rename(img_path, target_path)
                filesmoved=filesmoved+1
                result[name]=result[name]+1
                break
            else:
                logger.debug(i, "NO MATCH:",img_path, name, proba)

    for i in range(len(classes)):
        logger.debug(classes[i]," detected in ", result[classes[i]], " files")

if __name__ == '__main__':
    args=parse_args()
    main(args)