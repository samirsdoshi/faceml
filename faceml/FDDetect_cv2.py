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


protext_path = "/faceml/opencv/deploy.prototxt"
model_path = "/faceml/opencv/res10_300x300_ssd_iter_140000.caffemodel"
embedder_path = "/faceml/opencv/openface_nn4.small2.v1.t7"
recognizer_file = "recognizer_keras.pickle"
labelencoder_file = "labelencoder_keras.pickle"
logfile="faceml.log"

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--imagesdir", required=True, help="path to input directory of images")
ap.add_argument("-m", "--modelpath", required=True, help="directory with trained model")
ap.add_argument("-c", "--class", required=True, help="class name to filter")
ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store images having filter class")
ap.add_argument("-l", "--logdir", required=True, help="path to log directory")

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
                if fW < 10 or fH < 10:
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

def tostr(*args):
    retval=""
    for i in range(len(args)):
        retval=retval + str(args[i])
    return retval

model =load_facenet_model()
embedder =load_embedder()
recognizer = pickle.loads(open(args["modelpath"] + recognizer_file, "rb").read())
le = pickle.loads(open(args["modelpath"] + labelencoder_file, "rb").read())

imgcnt=1
filesmoved=0
flog=open(args["logdir"] + "/" + logfile,"w+")
for image_file in os.listdir(args["imagesdir"]):    
    # Load image
    img_path = os.path.join(args["imagesdir"], image_file)
    x1, y1, x2, y2, faces = extract_all_faces(model, img_path)
    if (x1 is None):
        continue
    print("candidate classes found:", len(x1))
    for i in range(len(x1)):
        faceBlob = cv2.dnn.blobFromImage(faces[i], 1.0 / 255, (96, 96),
            (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]
        if (proba >=0.85 and name==args["class"]):
            flog.write(tostr(i, "MATCH:",img_path, name, proba,"\n"))
            target_path = os.path.join(args["outdir"], image_file)
            flog.write(tostr("Moving ", img_path, " to ", target_path,"\n"))
            os.rename(img_path, target_path)
            filesmoved=filesmoved+1
            break
        else:
            flog.write(tostr(i, "NO MATCH:",img_path, name, proba,"\n"))

flog.write(tostr(args["class"]," detected in ", filesmoved, " files","\n"))
flog.close()