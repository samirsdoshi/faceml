import unittest
import time
from datetime import datetime
from time import mktime
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util import *
from extract_faces import *

class TestExtractFacesl(unittest.TestCase):
    def test_extractfaces(self):
        #given
        model=load_caffe_model()
        #when
        faces = extract_all_faces(model, "/faceml/sampleimages/extractfaces/bond.jpg", 0)
        #then
        #finds 6 faces
        self.assertEquals(len(faces),6)

if __name__ == '__main__':
    unittest.main()    