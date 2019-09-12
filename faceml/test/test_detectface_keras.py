import unittest
import time
from datetime import datetime
from time import mktime
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util import *
from detectface_keras import *

class TestDetectFaceKeras(unittest.TestCase):
    def test_load_keras_model(self):
         model =load_keras_model()
         self.assertEqual(type(model), keras.engine.training.Model)    
    
    def test_extract_all_faces(self):
        x1, y1, x2, y2, faces, pixels = extract_all_faces(MTCNN(), "/faceml/sampleimages/pierce1.jpg", 0)
        self.assertEqual(len(x1), 5)
        self.assertEqual(len(y1), 5)
        self.assertEqual(asarray(faces).shape, (5, 160, 160, 3))
        self.assertEqual(asarray(pixels).shape, (1080, 1460, 3))

if __name__ == '__main__':
    unittest.main()
