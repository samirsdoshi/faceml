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

if __name__ == '__main__':
    unittest.main()