import imutils
from PIL import Image
import cv2


def load_caffe_model():
    # load the model
    protext_path = "/faceml/opencv/deploy.prototxt"
    model_path = "/faceml/opencv/res10_300x300_ssd_iter_140000.caffemodel"
    return cv2.dnn.readNetFromCaffe(protext_path, model_path)

def load_embedder():
    embedder_path = "/faceml/opencv/openface_nn4.small2.v1.t7"
    return cv2.dnn.readNetFromTorch(embedder_path)

def load_image_cv2(filename):
    try:
        image = cv2.imread(filename, cv2.COLOR_BGR2RGB)
        (h, w) = image.shape[:2]
        return h,w,image
    except:
        print("Error loading " + filename)
        return (None,)*3

def blob_from_image(imagearray):
    return cv2.dnn.blobFromImage(cv2.resize(imagearray, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True)
