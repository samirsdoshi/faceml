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


protext_path = "/faceml/opencv/deploy.prototxt"
model_path = "/faceml/opencv/res10_300x300_ssd_iter_140000.caffemodel"
embedder_path = "/faceml/opencv/openface_nn4.small2.v1.t7"
recognizer_file = "recognizer_cv2.pickle"
labelencoder_file = "labelencoder_cv2.pickle"

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--traindir", required=True, help="path to input directory of images for training")
ap.add_argument("-v", "--valdir", required=True, help="path to input directory of images for training")
ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store trained model files")
args = vars(ap.parse_args())


def load_facenet_model():
    # load the model
    return cv2.dnn.readNetFromCaffe(protext_path, model_path)

def load_embedder():
    return cv2.dnn.readNetFromTorch(embedder_path)

def load_image(filename):
    try:
        image = cv2.imread(filename)
        (h, w) = image.shape[:2]
        return h,w,image
    except:
        print("Error loading " + filename)
        return (None,)*3

def blob_from_image(imagearray):
    return cv2.dnn.blobFromImage(cv2.resize(imagearray, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    

# extract a single face from a given photograph
def extract_all_faces(model,filename):
    print(filename)
    x1,y1,x2,y2 = list(),list(),list(),list()
    faces=list()
    (h,w,image) = load_image(filename)
    if (image is None):
        return (None,)*5
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
        print("no detections")
        return (None,)*5

# extract a single face from a given photograph
def extract_face(model,filename):    
    (x1,y1,x2,y2,faces) = extract_all_faces(model, filename)
    if (x1 is None or len(x1)==0):
        return (None,)*5
    return x1[0],y1[0],x2[0],y2[0],faces[0]

def load_faces(model, directory):
    faces = list()
    # enumerate files
    for filename in os.listdir(directory):
        if (filename == ".DS_Store"):
            continue
        # path
        path = directory + filename
        # get face
        x1,y1,x2,y2,face = extract_face(model, path)
        if face is not None:
            # store
            faces.append(face)
    return faces

def load_dataset(model, directory):
    X, y = list(), list()
    
    knownEmbeddings = []
    knownNames = []
    embedder = load_embedder()
    # enumerate folders, on per class
    for subdir in os.listdir(directory):
        if (subdir == ".DS_Store"):
            continue
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(model,path)
        if (len(faces)>0):
            # create labels
            labels = [subdir for _ in range(len(faces))]
            # summarize progress
            print('>loaded %d examples for class: %s' % (len(faces), subdir))
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


## end of functions    
model =load_facenet_model()
print("Load trainings")
trainY, trainX = load_dataset(model,args["traindir"])
print(len(trainX), len(trainY))
print("Load validations")
testY, testX = load_dataset(model,args["valdir"])
print(len(testX), len(testY))

le = LabelEncoder()
#print(data['names'])
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
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))