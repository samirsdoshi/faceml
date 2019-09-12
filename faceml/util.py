import os
import sys
import datetime
from datetime import date
import logging


logfile = "faceml.log"

def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    stat = os.stat(path_to_file)
    try:
        return stat.st_birthtime
    except AttributeError:
        # We're probably on Linux. No easy way to get creation dates here,
        # so we'll settle for when its content was last modified.
        return stat.st_mtime

def logdebug(self, *args):
    line=''.join(str(args[i]) for i in range(len(args)))
    self.log(logging.DEBUG, line)

def logerror(self, *args):
    line=''.join(str(args[i]) for i in range(len(args)))
    self.log(logging.ERROR, line)

def getMyLogger():
    return logging.getLogger("faceml")

def getLogger(logdir, logfile):
    logger = logging.getLogger("faceml")
    logger.setLevel(logging.DEBUG)
    ch = getLogHandler(logdir, logfile)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(ch)
    logger.__class__.debug=logdebug
    logger.__class__.error=logerror
    return logger

def getLogHandler(filedir, filename):
    if (filedir==""):
        return logging.StreamHandler()
    else:    
        return logging.FileHandler(filedir + "/" + filename)

def openfile(filepath):
    return open(filepath,"w+")

def getYearFromDatetime(dttime):
     dt=date.fromtimestamp(dttime)
     return dt.year

def getMonthFromDatetime(dttime):
    dt=date.fromtimestamp(dttime)
    return dt.strftime("%b")

def getDayFromDatetime(dttime):
     dt=date.fromtimestamp(dttime)
     return dt.day

def addMargin(startX,startY,endX,endY,margin):
    startX = int(startX - (startX*margin/100))
    endX = int(endX + (endX*margin/100))
    startY = int(startY - (startY*margin/100))
    endY = int(endY + (endY*margin/100))
    return startX,startY,endX,endY