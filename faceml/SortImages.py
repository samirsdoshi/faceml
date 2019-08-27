import sys
import keras
import os
from os import listdir
from os.path import isdir
import numpy as np
import pickle
import argparse
from util import *

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagedir", required=True, help="path to input directory of images for sorting")
ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store sorted files")
ap.add_argument("-s", "--sortmode", required=True, help="sort mode (y,m,d,ym,ymd). Create folders by  y (year), m (month), d (day), ym, ymd")
ap.add_argument("-f", "--foldermode", required=True, help="h (hierarchical) or f (flat)")

args = vars(ap.parse_args())

def setupDir(targetdir):
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

def setupTargetDir(outdir, year, month, day, sortmode, foldermode):
    targetDir=""
    sep="/"
    if (foldermode=="f"):
        sep="_"

    if (sortmode=="y"):
         targetDir = outdir + "/" + str(year)
    if (sortmode=="m"):
         targetDir = outdir + "/" + str(month)
    if (sortmode=="d"):
         targetDir = outdir + "/" + str(day)
    if (sortmode=="ym"):
        targetDir = outdir + "/" + str(year) + sep + str(month)
    if (sortmode=="ymd"):
        targetDir = outdir + "/" + str(year) + sep + str(month) + sep + str(day)
    setupDir(targetDir)
    return targetDir

for filename in listdir(args["imagedir"]):
    try:
        path = args["imagedir"] + "/" + filename
        if isdir(path):
            continue
        crdate = creation_date(path)
        year=getYearFromDatetime(crdate)
        month=getMonthFromDatetime(crdate)
        day=getDayFromDatetime(crdate)
        targetDir = setupTargetDir(args["outdir"], year, month, day, args["sortmode"].lower(),args["foldermode"].lower())
        print(path, "-->", targetDir + "/" + filename)
        os.rename(path, targetDir + "/" + filename)
    except Exception as e:
        print("Error processing " + path + "." + str(e))