import sys
import os
from os import listdir
from os.path import isdir
import numpy as np
import pickle
import argparse
from util import *
import logging

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagedir", required=True, help="path to input directory of images for sorting")
    ap.add_argument("-o", "--outdir", required=True, help="path to output directory to store sorted files")
    ap.add_argument("-s", "--sortmode", required=True, help="sort mode (y,m,d,ym,ymd). Create folders by  y (year), m (month), d (day), ym, ymd")
    ap.add_argument("-f", "--foldermode", required=True, help="h (hierarchical) or f (flat)")
    ap.add_argument("-l", "--logdir", required=False, default="", help="path to log directory")
    return vars(ap.parse_args())


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

def main(args):
    logger = getLogger(args["logdir"], logfile)
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
            logger.info(path, "-->", targetDir + "/" + filename)
            os.rename(path, targetDir + "/" + filename)
        except Exception as e:
            logger.info("Error processing " + path + "." + str(e))

if __name__ == '__main__':
    args =parse_args()
    main(args)