FROM jupyter/tensorflow-notebook
USER root
RUN pip install PyQt5
RUN pip install opencv-python
RUN pip install --upgrade scikit-image
RUN pip install --upgrade scikit-learn
RUN pip install --upgrade opencv-python
RUN pip install mtcnn
RUN pip install imutils
RUN mkdir /faceml
COPY ./keras-facenet /faceml/keras-facenet/
COPY ./yolo_keras /faceml/yolo_keras/
COPY ./opencv /faceml/opencv/
COPY ./faceml /faceml
RUN chown jovyan /faceml/notebooks
RUN chgrp users /faceml/notebooks
RUN chown jovyan /faceml/notebooks/*
RUN chgrp users /faceml/notebooks/*
