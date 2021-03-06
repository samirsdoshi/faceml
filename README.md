## Description

<i>Please read the full article on [LinkedIn](https://www.linkedin.com/pulse/ai-saved-my-family-photos-samir-doshi/) or [Medium](https://medium.com/p/521ce6fa5877).</i>
The github repo provides a docker image with necessary ML base models and python utilities for face recognition, with easy to use command-line api. 
Each tool has a corresponding jupyter notebook so you can test with your images, tune parameters, or just visualize results. 

## Setup

* Install [Docker](https://docs.docker.com/install/) on your local machine
* Download docker image from docker hub using 
` docker pull samirsdoshi/faceml:latest `
 OR  
* Build image locally by running build-image.sh and copy large model files  
a) Download a converted darknet YOLO model to keras [here](https://drive.google.com/drive/folders/1YHYPsN4BnXz408_SY9Mo0G4Z2cSR0WCG?usp=sharing) and copy to yolo_keras/yolo.h5. This is used in detectobjects.py for object detection  
b) Download keras-facenet from [here](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn) and copy under keras-facenet. This is used for face detection in train_keras.py and detectface_keras.py  
c) Download a Torch deep learning model which produces the 128-D facial embeddings [here](https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7) file and save as opencv/openface_nn4.small2.v1.t7.  
d) Also download a pre-trained Caffe deep learning model to detect faces [here](https://github.com/mmilovec/facedetectionOpenCV/blob/master/res10_300x300_ssd_iter_140000.caffemodel) and save as opencv/res10_300x300_ssd_iter_140000.caffemodel.   


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
* Change the volume line in faceml.sh to mount your directory with images. Run [faceml.sh](faceml.sh) to launch docker container. The container also runs jupyter notebook. It will wait for few seconds for the container to start. You can access the jupyter notebook from your browser using
http://localhost:8888/?token=<i>token</i>
* Run ` docker exec -it faceml bash `  to enter docker container
##### Command Lines
* To detect various types of [objects](yolo_keras/coco_classes.txt) and move images with objects in a seperate folder
```
root@afc13fdc17ef:/faceml# python detectobjects.py --help
Using TensorFlow backend.
usage: detectobjects.py [-h] -i IMAGEDIR -c CLASS [-k [CONFIDENCE]]
                        [-s [SIZE]] [-n [COUNT]] [-p [PORTRAIT]] [-g [GROUP]]
                        [-o OUTDIR] [-l LOGDIR] [-v LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imagedir IMAGEDIR
                        path to input directory of images
  -c CLASS, --class CLASS
                        object class expression to search for as per
                        yolo_keras/coco_classes.txt
  -k [CONFIDENCE], --confidence [CONFIDENCE]
                        minimum confidence percentage for object detection.
                        Default 80
  -s [SIZE], --size [SIZE]
                        minimum percentage size of the object in the image.
  -n [COUNT], --count [COUNT]
                        select images containing n count of class object
  -p [PORTRAIT], --portrait [PORTRAIT]
                        portrait mode. select images only if smallest of the n
                        count objects is bigger by p percentage over next
                        biggest object
  -g [GROUP], --group [GROUP]
                        group mode. select images only if n count objects are
                        within g percentage
  -o OUTDIR, --outdir OUTDIR
                        path to output directory where selected images will be
                        moved
  -l LOGDIR, --logdir LOGDIR
                        path to log directory
  -v LOG_LEVEL, --loglevel LOG_LEVEL
                        log level: DEBUG, INFO, ERROR. Default INFO
```
Class expression supports expressions like
* person
* not [person]
* [person] and [car]
* [person] and ([car] or [boat])
* [person] and [flower] and not [bird]
* ([person] and [car]) or ([flower] and [bird])
etc..

Also you can find potrait photos of a single person and use it for training the model. You can set big enough size and count to 1 to isolate images of single person, even if there are few other people in the background.  
</br>
* To detect similar images and group them under a folder, so that those can be reviewed and sorted appropriately.
```
root@afc13fdc17ef:/faceml# python detectsimilar.py --help
Using TensorFlow backend.
usage: detectsimilar.py [-h] -d IMAGEDIR [-c CLASS] [-t [THRESHOLD]]
                        [-o OUTDIR] [-l LOGDIR] [-v LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  -d IMAGEDIR, --imagedir IMAGEDIR
                        path to input directory of images
  -c CLASS, --class CLASS
                        object class to filter as per
                        yolo_keras/coco_classes.txt
  -t [THRESHOLD], --threshold [THRESHOLD]
                        minimum confidence percentage for similarity. Default
                        70
  -o OUTDIR, --outdir OUTDIR
                        path to output directory where selected images will be
                        moved
  -l LOGDIR, --logdir LOGDIR
                        path to log directory
  -v LOG_LEVEL, --loglevel LOG_LEVEL
                        log level: DEBUG, INFO, ERROR. Default INFO
```
* To extract faces from images and save as seperate files (could be used as inputs for training)
```
root@afc13fdc17ef:/faceml# python extract_faces.py --help
Using TensorFlow backend.
usage: extract_faces.py [-h] -i IMAGESDIR -o OUTDIR [-p [MARGIN]] [-l LOGDIR]
                        [-v LOG_LEVEL]

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
  -v LOG_LEVEL, --loglevel LOG_LEVEL
                        log level: DEBUG, INFO, ERROR. Default INFO
```
See [example](faceml/sampleimages/extractfaces/README.md)


* To train model (using Keras/Facenet)
```
root@7f87ba121d08:/faceml# python train_keras.py --help
Using TensorFlow backend.
usage: train_keras.py [-h] -t TRAINDIR [-p [MARGIN]] -v VALDIR -o OUTDIR
                      [-l LOGDIR] [-g LOG_LEVEL]

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
  -l LOGDIR, --logdir LOGDIR
                        path to log directory
  -g LOG_LEVEL, --loglevel LOG_LEVEL
                        log level: DEBUG, INFO, ERROR. Default INFO
```
* To detect faces and move files (using Keras/Facenet)
```
root@87b636b3469e:/faceml# python detectface_keras.py --help
Using TensorFlow backend.
usage: detectface_keras.py [-h] -i IMAGESDIR -m MODELPATH -c PERSON
                           [-p [MARGIN]] [-k [CONFIDENCE]] [-o OUTDIR]
                           [-n NOMATCHDIR] [-l LOGDIR] [-v LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGESDIR, --imagesdir IMAGESDIR
                        path to input directory of images
  -m MODELPATH, --modelpath MODELPATH
                        directory with trained model
  -c PERSON, --person PERSON
                        person name expression to filter - <name1> | not
                        [name1] | [name1] and [name2] | [name1] and ([name2]
                        or [name3]) and not [name4] etc
  -p [MARGIN], --margin [MARGIN]
                        margin percentage pixels to include around the face
  -k [CONFIDENCE], --confidence [CONFIDENCE]
                        minimum confidence percentage for face recognition.
                        Default 70
  -o OUTDIR, --outdir OUTDIR
                        path to output directory to store images having filter
                        class
  -n NOMATCHDIR, --nomatchdir NOMATCHDIR
                        directory to move image to, if not matched.
  -l LOGDIR, --logdir LOGDIR
                        path to log directory
  -v LOG_LEVEL, --loglevel LOG_LEVEL
                        log level: DEBUG, INFO, ERROR. Default INFO
```
* To train model (using OpenCV/Caffe)
```
root@7f87ba121d08:/faceml# python train_cv2.py --help
usage: train_cv2.py [-h] -t TRAINDIR -v VALDIR [-p [MARGIN]] -o OUTDIR
                    [-l LOGDIR] [-g LOG_LEVEL]

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
  -l LOGDIR, --logdir LOGDIR
                        path to log directory
  -g LOG_LEVEL, --loglevel LOG_LEVEL
                        log level: DEBUG, INFO, ERROR. Default INFO
```
* To detect faces and move files (using OpenCV/Caffe)
```
usage: detectface_cv2.py [-h] -i IMAGESDIR -m MODELPATH -c PERSON
                         [-p [MARGIN]] [-k [CONFIDENCE]] [-o OUTDIR]
                         [-n NOMATCHDIR] [-l LOGDIR] [-v LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGESDIR, --imagesdir IMAGESDIR
                        path to input directory of images
  -m MODELPATH, --modelpath MODELPATH
                        directory with trained model
  -c PERSON, --person PERSON
                        person name expression to filter - <name1> | not
                        [name1] | [name1] and [name2] | [name1] and ([name2]
                        or [name3]) and not [name4] etc
  -p [MARGIN], --margin [MARGIN]
                        margin percentage pixels to include around the face
  -k [CONFIDENCE], --confidence [CONFIDENCE]
                        minimum confidence percentage for face recognition.
                        Default 70
  -o OUTDIR, --outdir OUTDIR
                        path to output directory to store images having filter
                        class
  -n NOMATCHDIR, --nomatchdir NOMATCHDIR
                        directory to move image to, if not matched.
  -l LOGDIR, --logdir LOGDIR
                        path to log directory
  -v LOG_LEVEL, --loglevel LOG_LEVEL
                        log level: DEBUG, INFO, ERROR. Default INFO
```
* Utility to sort images into folders on year/month/day (hierachical or flat directories)
```
root@7f87ba121d08:/faceml# python sortimages.py --help
usage: sortimages.py [-h] -i IMAGEDIR -o OUTDIR -s SORTMODE -f FOLDERMODE
                     [-l LOGDIR] [-v LOG_LEVEL]

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
  -l LOGDIR, --logdir LOGDIR
                        path to log directory
  -v LOG_LEVEL, --loglevel LOG_LEVEL
                        log level: DEBUG, INFO, ERROR. Default INFO
```
## Example
I have the combined training/detection code as jupyter notebooks for both methods. Review the end of both files below for results. 

* [Object Detection](faceml/notebooks/objectdetect.ipynb)
* [Detect similar images](faceml/notebooks/detectsimilar.ipynb)
* [Face Detection using Keras](faceml/notebooks/detectface_keras.ipynb)
* [Face Detection using cv2](faceml/notebooks/detectface_cv2.ipynb)

## Credits/References:
https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/  
https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
https://pjreddie.com/darknet/yolo/  
https://mc.ai/face-detection-with-opencv-and-deep-learning/  
https://sb-nj-wp-prod-1.pyimagesearch.com/2018/09/24/opencv-face-recognition/  
http://cmusatyalab.github.io/openface/  
https://cmusatyalab.github.io/openface/models-and-accuracies/  


