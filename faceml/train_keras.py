import sys
import keras
from keras import backend as K
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

recognizer_file = "recognizer_keras.pickle"
labelencoder_file = "labelencoder_keras.pickle"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--traindir", required=True, help="path to input directory of images for training")
    ap.add_argument("-p", "--margin", required=False,  nargs='?', const=0, type=int, default=0, help="margin percentage pixels to include around the face")
    ap.add_argument("-v", "--valdir", required=True, help="path to input directory of images for training")
    ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store trained model files")
    ap.add_argument("-l", "--logdir", required=False, default="", help="path to log directory")
    ap.add_argument("-g", "--loglevel", required=False, default="INFO", dest='log_level', type=log_level_string_to_int, help="log level: DEBUG, INFO, ERROR. Default INFO")

    return vars(ap.parse_args())


def extract_face(detector, filename, margin):    
    x1,y1,x2,y2,faces = extract_all_faces(detector, filename, margin)
    if (faces is None or len(faces)==0):
        return (None,)*5
    return x1[0],y1[0],x2[0],y2[0],faces[0]    

# extract a single face from a given photograph
def extract_all_faces(detector, filename, margin, required_size=(160, 160)):
    logger = getMyLogger()
    logger.info(filename)
    x1,y1,x2,y2 = list(),list(),list(),list()
    faces=list()
    # load image from file
    # convert to array
    pixels = load_image(filename)
    if (pixels is None):
        return (None,)*5
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
            # extract the face
            startX = int(startX - (startX*margin/100))
            endX = int(endX + (endX*margin/100))
            startY = int(startY - (startY*margin/100))
            endY = int(endY + (endY*margin/100))

            face = pixels[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            #print(i, x,y,width,height,fH,fW)
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
        return x1, y1, x2, y2, faces
    else:
        return (None,)*5


def load_faces(detector, directory, margin):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        x1, y1, x2, y2, face = extract_face(detector, path, margin)
        if face is not None:
            # store
            faces.append(face)
    return faces

def load_dataset(detector,directory, margin):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(detector,path, margin)
        if (len(faces)>0):
            # create labels
            labels = [subdir for _ in range(len(faces))]
            # summarize progress
            print('>loaded %d examples for class: %s' % (len(faces), subdir))
            # store
            X.extend(faces)
            y.extend(labels)
    return asarray(X), asarray(y)



## end of functions    

def main(args):
    logger =  getLogger(args["logdir"], args["log_level"], logfile)

    detector = MTCNN()
    trainX, trainY = load_dataset(detector,args["traindir"],int(args["margin"]))
    # load test dataset
    testX, testY = load_dataset(detector,args["valdir"], int(args["margin"]))

    # load the facenet model
    model = load_keras_model()
    logger.info('Loaded Model')

    # convert each face in the train set to an embedding
    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    logger.info(newTrainX.shape)
    # convert each face in the test set to an embedding
    newTestX = list()
    for face_pixels in testX:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)
    logger.info(newTestX.shape)

    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    newTrainX = in_encoder.transform(newTrainX)
    newTestX = in_encoder.transform(newTestX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainY)
    trainY = out_encoder.transform(trainY)
    testY = out_encoder.transform(testY)

    # fit model
    recognizer = SVC(C=0.9, gamma="scale", kernel="linear", probability=True)
    recognizer.fit(newTrainX, trainY)

    # write the actual face recognition model to disk
    f = open(args["outdir"] + "/" + recognizer_file, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(args["outdir"] + "/" + labelencoder_file, "wb")
    f.write(pickle.dumps(out_encoder))
    f.close()

    yhat_train = recognizer.predict(newTrainX)
    yhat_test = recognizer.predict(newTestX)
    # score
    score_train = accuracy_score(trainY, yhat_train)
    score_test = accuracy_score(testY, yhat_test)
    # summarize
    logger.info('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

if __name__ == '__main__':
    args = parse_args()
    main(args)    