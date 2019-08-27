## Description

The repo provides a docker image with necessary ML base models and code for face recognition, with easy to use command-line api. I trained the ML model
using potrait images/faces of people from my family and then sorted out images containing anyone from family from your phone, camera, whatsapp groups etc, so that I can delete the remaining images from phone/camera without losing family pictures. 

The detection also tries with multiple margin% to include more of a face image and picks the one with highest confidence, to improve detection accuracy. I found that this helped quite a bit. 

## Setup

* Download docker from docker hub using 
` docker pull samirsdoshi/faceml:latest `
 OR  
* Build image locally by running build-image.sh and copy large model files  
a) Download a converted darknet YOLO model to keras [here](https://drive.google.com/drive/folders/1YHYPsN4BnXz408_SY9Mo0G4Z2cSR0WCG?usp=sharing) and copy to yolo_keras/yolo.h5. This is used in OD.py for object detection  
b) Download keras-facenet from [here](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn) and copy under keras-facenet. This is used for face detection in FDTrain_keras.py and FDDetect_keras.py  
c) Download a Torch deep learning model which produces the 128-D facial embeddings [here](https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7) file and save as opencv/openface_nn4.small2.v1.t7.  
d) Also download a pre-trained Caffe deep learning model to detect faces [here](https://github.com/mmilovec/facedetectionOpenCV/blob/master/res10_300x300_ssd_iter_140000.caffemodel) and save as opencv/res10_300x300_ssd_iter_140000.caffemodel.   
Both these files are used in FDTrain_cv2.py and FDDetect_cv2.py  

## Usage
* Setup a folder with images. for. e.g
```
    |--training
        |-person1
        |-person2
        |-person3
        ...
    |-validation
        |-person1
        |-person2
        |-person3
        ...
    |-allimages
        image1.jpg
        image2.jpg
        ...
    |-outdir
    |-logdir
``` 
* Run [faceml.sh](faceml.sh) to launch docker container. Change the volume line in faceml.sh to mount your directory with images. The container also runs jupyter notebook. It will wait for few seconds for the container to start. You can access the jupyter notebook from your browser using
http://localhost:8888/?token=<i>token</i>
* Run ` docker exec -it faceml bash `  to enter docker container

* To detect various types of [objects](yolo_keras/coco_classes.txt) and move images with objects in a seperate folder
```
root@b31ba7fcbfc1:/faceml# python OD.py --help
usage: OD.py [-h] -i IMAGEDIR -c CLASS -o OUTDIR
optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imagedir IMAGEDIR
                        path to input directory of images
  -c CLASS, --class CLASS
                        object class to search for as per yolo_keras/coco_classes.txt
  -o OUTDIR, --outdir OUTDIR
                        path to output directory with images having search
                        objects
```
* To extract faces from images and save as seperate files (could be used as inputs for training)
```
root@b31ba7fcbfc1:/faceml# python ExtractFaces.py --help
Using TensorFlow backend.
usage: ExtractFaces.py [-h] -i IMAGESDIR -o OUTDIR [-p [MARGIN]] -l LOGDIR
optional arguments:
  -h, --help            show this help message and exit
  -i IMAGESDIR, --imagesdir IMAGESDIR
                        path to input directory of images
  -o OUTDIR, --outdir OUTDIR
                        path to output directory to store face images
  -p [MARGIN], --margin [MARGIN]
                        margin percentage pixels to include around the face
  -l LOGDIR, --logdir LOGDIR
                        path to log directory
```
See [example](faceml/sampleimages/extractfaces/README.md)
* To train model (using Keras/Facenet)
```
root@b31ba7fcbfc1:/faceml# python FDTrain_keras.py --help
Using TensorFlow backend.
usage: FDTrain_keras.py [-h] -t TRAINDIR [-p [MARGIN]] -v VALDIR -o OUTDIR
optional arguments:
  -h, --help            show this help message and exit
  -t TRAINDIR, --traindir TRAINDIR
                        path to input directory of images for training
  -p [MARGIN], --margin [MARGIN]
                        margin percentage pixels to include around the face
  -v VALDIR, --valdir VALDIR
                        path to input directory of images for training
  -o OUTDIR, --outdir OUTDIR
                        path to output directory to store trained model files
```
* To detect faces and move files (using Keras/Facenet)
```
root@b31ba7fcbfc1:/faceml# python FDDetect_keras.py --help
Using TensorFlow backend.
usage: FDDetect_keras.py [-h] -i IMAGESDIR -m MODELPATH -c CLASS [-p [MARGIN]]
                         -o OUTDIR -l LOGDIR
optional arguments:
  -h, --help            show this help message and exit
  -i IMAGESDIR, --imagesdir IMAGESDIR
                        path to input directory of images
  -m MODELPATH, --modelpath MODELPATH
                        directory with trained model
  -c CLASS, --class CLASS
                        class name to filter (class1,class2,...)
  -p [MARGIN], --margin [MARGIN]
                        margin percentage pixels to include around the face
  -o OUTDIR, --outdir OUTDIR
                        path to output directory to store images having filter
                        class
  -l LOGDIR, --logdir LOGDIR
                        path to log directory
```
* To train model (using OpenCV/Caffe)
```
root@b31ba7fcbfc1:/faceml# python FDTrain_cv2.py --help
usage: FDTrain_cv2.py [-h] -t TRAINDIR -v VALDIR [-p [MARGIN]] -o OUTDIR
optional arguments:
  -h, --help            show this help message and exit
  -t TRAINDIR, --traindir TRAINDIR
                        path to input directory of images for training
  -v VALDIR, --valdir VALDIR
                        path to input directory of images for training
  -p [MARGIN], --margin [MARGIN]
                        margin percentage pixels to include around the face
  -o OUTDIR, --outdir OUTDIR
                        path to output directory to store trained model files
```
* To detect faces and move files (using OpenCV/Caffe)
```
root@b31ba7fcbfc1:/faceml# python FDDetect_cv2.py --help
usage: FDDetect_cv2.py [-h] -i IMAGESDIR -m MODELPATH -c CLASS [-p [MARGIN]]
                       -o OUTDIR -l LOGDIR
optional arguments:
  -h, --help            show this help message and exit
  -i IMAGESDIR, --imagesdir IMAGESDIR
                        path to input directory of images
  -m MODELPATH, --modelpath MODELPATH
                        directory with trained model
  -c CLASS, --class CLASS
                        class name to filter (class1,class2,...)
  -p [MARGIN], --margin [MARGIN]
                        margin percentage pixels to include around the face
  -o OUTDIR, --outdir OUTDIR
                        path to output directory to store images having filter
                        class
  -l LOGDIR, --logdir LOGDIR
                        path to log directory
```
* Utility to sort images into folders on year/month/day (hierachical or flat directories)
```
root@45df4a0417e8:/faceml# python SortImages.py --help
Using TensorFlow backend.
usage: SortImages.py [-h] -i IMAGEDIR -o OUTDIR -s SORTMODE -f FOLDERMODE

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imagedir IMAGEDIR
                        path to input directory of images for sorting
  -o OUTDIR, --outdir OUTDIR
                        path to output directory to store sorted files
  -s SORTMODE, --sortmode SORTMODE
                        sort mode (y,m,d,ym,ymd). Create folders by y (year),
                        m (month), d (day), ym, ymd
  -f FOLDERMODE, --foldermode FOLDERMODE
                        h (hierarchical) or f (flat)
```
## Example
I have the combined training/detection code as jupyter notebooks for both methods at
* [FD_keras](faceml/notebooks/FD_keras.ipynb)
* [FD_cv2](faceml/notebooks/FD_cv2.ipynb)

## Credits/References:
https://pjreddie.com/darknet/yolo/  
https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/  
https://mc.ai/face-detection-with-opencv-and-deep-learning/  
https://sb-nj-wp-prod-1.pyimagesearch.com/2018/09/24/opencv-face-recognition/  
https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/  
http://cmusatyalab.github.io/openface/  
https://cmusatyalab.github.io/openface/models-and-accuracies/  


