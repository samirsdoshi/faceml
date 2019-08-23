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

model_path = "/faceml/keras-facenet/model/facenet_keras.h5"
model_weights_path = "/faceml/keras-facenet/weights/facenet_keras_weights.h5"
recognizer_file = "recognizer_keras.pickle"
labelencoder_file = "labelencoder_keras.pickle"

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--traindir", required=True, help="path to input directory of images for training")
ap.add_argument("-v", "--valdir", required=True, help="path to input directory of images for training")
ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store trained model files")
args = vars(ap.parse_args())


def load_keras_model():
    # load the model
    model =  load_model(model_path)
    model.load_weights(model_weights_path)
    return model

def extract_face(detector, filename):    
    x1,y1,x2,y2,faces = extract_all_faces(detector, filename)
    if (faces is None or len(faces)==0):
        return (None,)*5
    return x1[0],y1[0],x2[0],y2[0],faces[0]    

# extract a single face from a given photograph
def extract_all_faces(detector, filename, required_size=(160, 160)):
    print(filename)
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
            x=results[i]['box'][0]
            y=results[i]['box'][1]
            width=results[i]['box'][2]
            height= results[i]['box'][3]
            #x1[i], y1[i], width, height = results[i]['box']
            # bug fix
            x,y = abs(x), abs(y)
            # extract the face
            face = pixels[y:y+height, x:x+width]
            (fH, fW) = face.shape[:2]
            #print(i, x,y,width,height,fH,fW)
            if fW < 10 or fH < 10:
                continue
            x1.append(x)
            y1.append(y)
            x2.append(x1[i] + width)
            y2.append(y1[i] + height)
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            faces.append(face_array)
        return x1, y1, x2, y2, faces
    else:
        return (None,)*5


def load_faces(detector, directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        x1, y1, x2, y2, face = extract_face(detector, path)
        if face is not None:
            # store
            faces.append(face)
    return faces

def load_dataset(detector,directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(detector,path)
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

detector = MTCNN()
trainX, trainY = load_dataset(detector,args["traindir"])
print(len(trainX), len(trainY))
# load test dataset
testX, testY = load_dataset(detector,args["valdir"])
print(len(testX), len(testY))

# load the facenet model
model = load_keras_model()
print('Loaded Model')

# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)
# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)

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
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))