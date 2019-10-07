from PIL import Image
import numpy as np
from numpy import asarray
from numpy import expand_dims
from keras.models import load_model
from keras import backend as K
import math

def load_keras_model():
    # load the model
    model_path = "/faceml/keras-facenet/model/facenet_keras.h5"
    model_weights_path = "/faceml/keras-facenet/weights/facenet_keras_weights.h5"
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    return model


def resize_image(src_image, size=(128,128)): 
    
    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS)
    
    # Create a new square background image
    new_image = Image.new("RGB", size)
    
    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
  
    # return the resized image
    return new_image        


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

def detect_objects(image, boxes, scores, classes, yolo_model, input_image_shape):
    
    # normalize and reshape image data
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    # Predict classes and locations using Tensorflow session
    sess = K.get_session()
    out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: image_data,
                    input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
    return out_boxes, out_scores, out_classes

def scale_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    xpad, ypad=(w-nw)//2,(h-nh)//2
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, (xpad, ypad))
    return new_image, scale, xpad, ypad

def getBoxArea(out_box):
    return (out_box[3]-out_box[1]) * (out_box[2]-out_box[0])

def getEnclosingArea(objects, classes, out_boxes, image_size, requiredSize):
    w,h=image_size
    image_area=w*h
    x1, y1, x2, y2 =w+1,h+1,0,0
    classcount=0
    for i in range(len(out_boxes)):
        area = getBoxArea(out_boxes[i])
        if (area/image_area)*100 > requiredSize:
            if (objects[i] in classes):
                classcount=classcount+1
                y1 = max(0, min(y1, out_boxes[i][0]))
                x1 = max(0, min(x1, out_boxes[i][1]))
                y2 = min(h, max(y2, out_boxes[i][2]))    
                x2 = min(w, max(x2, out_boxes[i][3]))
    return classcount, math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)

def transpose(x1,y1,x2,y2,xpad,ypad,scale):
    newX1 = max(0,int((x1-xpad)/scale))
    newX2 = max(0,int((x2-xpad)/scale))
    newY1 = max(0,int((y1-ypad)/scale))
    newY2 = max(0,int((y2-ypad)/scale))
    return newX1, newY1, newX2, newY2
