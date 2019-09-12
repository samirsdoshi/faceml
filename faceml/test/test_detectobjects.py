import unittest
import time
from datetime import datetime
from time import mktime
import os
import sys
from numpy import asarray
from keras import backend as K
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detectobjects import *

class TestDO(unittest.TestCase):
            
    def test_openmodel(self):
        class_names, anchors, yolo_model = open_model()
        self.assertEqual(asarray(class_names).size, 80)
        self.assertEqual(asarray(anchors).size, 18)
        return class_names, anchors, yolo_model

    def test_detect_objects(self):
        #given
        img_path = "/faceml/sampleimages/pierce1.jpg"
        model_image_size = (416, 416)
        class_names, anchors, yolo_model = self.test_openmodel()
        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(yolo_model.output, anchors, len(class_names), input_image_shape,
                                        score_threshold=0.3, iou_threshold=0.45)
        image = letterbox_image(Image.open(img_path), tuple(reversed(model_image_size)))
        #when
        out_boxes, out_scores, out_classes =  detect_objects(image, boxes, scores, classes,yolo_model,input_image_shape)
        #then
        self.assertEqual(out_boxes.shape,(4,4))
        self.assertEqual(out_scores.size, 4)
        self.assertEqual(out_classes.shape, (4,))
        return out_boxes, out_scores, out_classes, class_names

    def test_boxareas(self):
        out_boxes, out_scores, out_classes, class_names = self.test_detect_objects()
        objects=[class_names[out_classes[i]] for i in range(len(out_classes))]
        matchedConfidence, matchedSize, areas = getBoxAreas(objects, out_classes, "person", out_scores, out_boxes, 416*416, 70, 20)       
        self.assertEqual(matchedConfidence,3)
        self.assertEqual(matchedSize,1)
        self.assertEqual(asarray(areas).size,3)
        return matchedConfidence, matchedSize, areas

    def test_categorizeimage(self):
        requiredCount, portraitDiffDefault, groupDiffDefault, portraitDiff, groupDiff =0,150,100,0,0
        matchedConfidence, matchedSize, areas = self.test_boxareas()
        
        sortedareas = sorted(areas,reverse=True)
        print("Areas:",sortedareas)
        areasDiff = list((sortedareas[i]-sortedareas[i+1])/sortedareas[i+1] for i in range(len(sortedareas)-1))
        print("Area Diffs:",areasDiff)
        
        #matches portrait and selfie by default
        isPortrait, isGroup, isSelfie = categorizeImage(areas, requiredCount, portraitDiffDefault, groupDiffDefault, portraitDiff, groupDiff)
        self.assertListEqual([isPortrait, isGroup, isSelfie],[True, False, True])

        #matches portrait with %diff of 150% over next person
        portraitDiff=150
        isPortrait, isGroup, isSelfie = categorizeImage(areas, requiredCount, portraitDiffDefault, groupDiffDefault, portraitDiff, groupDiff)
        self.assertListEqual([isPortrait, isGroup, isSelfie],[True, False, False])

        #matches selfie with %diff under 75% setting
        portraitDiff=0
        groupDiff=75   
        isPortrait, isGroup, isSelfie = categorizeImage(areas, requiredCount, portraitDiffDefault, groupDiffDefault, portraitDiff, groupDiff)
        self.assertListEqual([isPortrait, isGroup, isSelfie],[False, False, True])

        #matches as group with %diff of under 175%
        groupDiff=175   
        isPortrait, isGroup, isSelfie = categorizeImage(areas, requiredCount, portraitDiffDefault, groupDiffDefault, portraitDiff, groupDiff)
        self.assertListEqual([isPortrait, isGroup, isSelfie],[False, True, False])

if __name__ == '__main__':
    unittest.main()