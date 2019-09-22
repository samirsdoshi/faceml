import unittest
import time
from datetime import datetime
from time import mktime
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util import *

class TestUtil(unittest.TestCase):
    def test_year(self):
        x = datetime.datetime(2019, 1, 6, 15, 8, 24, 78915)
        year = getYearFromDatetime(time.mktime(x.timetuple()))
        self.assertEqual(year, 2019)

    def test_margin(self):
        x1,y1,x2,y2 = addMargin(100,50,200,80,10)
        self.assertEqual(x1, 90)
        self.assertEqual(y1,45)
        self.assertEqual(x2, 220)
        self.assertEqual(y2, 88)

    def test_processClassName(self):

        classname="person"
        search_classes, not_classes, expr = processClassName(classname)
        self.assertEqual(search_classes,["person"])
        self.assertEqual(not_classes,[])
        self.assertEqual(expr,"('person' in objects)")


        classname="[person]"
        search_classes, not_classes, expr= processClassName(classname)
        self.assertEqual(search_classes,["person"])
        self.assertEqual(not_classes,[])
        self.assertEqual(expr,"('person' in objects)")


        classname="not [person]"
        search_classes, not_classes, expr = processClassName(classname)
        self.assertEqual(search_classes,[])
        self.assertEqual(not_classes,["person"])
        self.assertEqual(expr,"not ('person' in objects)")


        classname="[person] and [flower]"
        search_classes, not_classes, expr = processClassName(classname)
        self.assertEqual(search_classes,["person","flower"])
        self.assertEqual(not_classes,[])
        self.assertEqual(expr,"('person' in objects) and ('flower' in objects)")

        classname="[person] and [flower] and not [bird]"
        search_classes, not_classes, expr= processClassName(classname)
        self.assertEqual(search_classes,["person","flower"])
        self.assertEqual(not_classes, ["bird"])
        self.assertEqual(expr,"('person' in objects) and ('flower' in objects) and not ('bird' in objects)")


        classname="([person] and [flower]) or [bird]"
        search_classes, not_classes, expr = processClassName(classname)
        self.assertEqual(search_classes,["person","flower","bird"])
        self.assertEqual(not_classes,[])
        self.assertEqual(expr,"(('person' in objects) and ('flower' in objects)) or ('bird' in objects)")


        classname="([person] and [car]) or ([flower] and [bird])"
        search_classes, not_classes, expr= processClassName(classname)
        self.assertEqual(search_classes,["person","car","flower","bird"])
        self.assertEqual(not_classes,[])
        self.assertEqual(expr,"(('person' in objects) and ('car' in objects)) or (('flower' in objects) and ('bird' in objects))")


if __name__ == '__main__':
    unittest.main()