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

model_path = "/faceml/keras-facenet/model/facenet_keras.h5"
model_weights_path = "/faceml/keras-facenet/weights/facenet_keras_weights.h5"
recognizer_file = "recognizer.pickle"
labelencoder_file = "labelencoder.pickle"
logfile="faceml.log"

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--imagesdir", required=True, help="path to input directory of images")
ap.add_argument("-m", "--modelpath", required=True, help="directory with trained model")
ap.add_argument("-c", "--class", required=True, help="class name to filter")
ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store images having filter class")
ap.add_argument("-l", "--logdir", required=True, help="path to log directory")

args = vars(ap.parse_args())

def load_keras_model():
    # load the model
    model =  load_model(model_path)
    model.load_weights(model_weights_path)
    return model

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
            # bug fix
            x,y = abs(x), abs(y)
            # extract the face
            face = pixels[y:y+height, x:x+width]
            (fH, fW) = face.shape[:2]
            #print(i, x,y,width,height,fH,fW)
            if fW < 20 or fH < 20:
                continue
            x1.append(x)
            y1.append(y)
            x2.append(x + width)
            y2.append(y + height)
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            faces.append(face_array)
        return x1, y1, x2, y2, faces
    else:
        return (None,)*5



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

model =load_keras_model()
detector = MTCNN()
recognizer = pickle.loads(open(args["modelpath"] + recognizer_file, "rb").read())
out_encoder = pickle.loads(open(args["modelpath"] + labelencoder_file, "rb").read())

imgcnt=1
filesmoved=0
flog=open(args["logdir"] + "/" + logfile,"w+")
in_encoder = Normalizer(norm='l2')
for image_file in os.listdir(args["imagesdir"]):    
    # Load image
    img_path = os.path.join(args["imagesdir"], image_file)
    x1, y1, x2, y2, faces = extract_all_faces(detector, img_path)
    if (x1 is None):
        continue
    print("candidate classes found:", len(x1))
    for i in range(len(faces)):
        embedding = get_embedding(model, faces[i])
        testX = in_encoder.transform(embedding.reshape(1,-1))
        yhat_class = recognizer.predict(testX)
        yhat_prob = recognizer.predict_proba(testX)
        
        flog.write(tostr(yhat_class, yhat_prob))
        
        class_index = yhat_class[0]
        proba = yhat_prob[0,class_index]
        predict_names = out_encoder.inverse_transform(yhat_class)

        flog.write(tostr(predict_names))

        name = predict_names[0]
        if (proba >=0.80 and name==args["class"]):
            flog.write(tostr(i, "MATCH:",img_path, name, proba,"\n"))
            target_path = os.path.join(args["outdir"], image_file)
            flog.write(tostr("Moving ", img_path, " to ", target_path,"\n"))
            os.rename(img_path, target_path)
            filesmoved=filesmoved+1
            break
        else:
            flog.write(tostr(i, "NO MATCH:",img_path, name, proba,"\n"))
    flog.flush()

flog.write(tostr(args["class"]," detected in ", filesmoved, " files","\n"))
flog.close()