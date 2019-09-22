from yolo_keras.utils import *
from yolo_keras.model import *
from keras.models import load_model

def open_yolo_model():
    anchors_path = "/faceml/yolo_keras/yolo_anchors.txt"
    classes_path = "/faceml/yolo_keras/coco_classes.txt"
    model_path="/faceml/yolo_keras/yolo.h5"

    # Get the anchor box coordinates for the model
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    # Get the COCO classes on which the model was trained
    with open(classes_path) as f:
        class_names = f.readlines()
        class_names = [c.strip() for c in class_names] 

    # Create YOLO model
    yolo_model = load_model(model_path, compile=False)

    return class_names, anchors,  yolo_model