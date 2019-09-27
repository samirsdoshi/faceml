import os
import sys
import datetime
from datetime import date
import logging
import numpy as np
from numpy import asarray
import argparse

logfile = "faceml.log"
_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']

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


def log_level_string_to_int(log_level_string):
    if not log_level_string in _LOG_LEVEL_STRINGS:
        message = 'invalid choice: {0} (choose from {1})'.format(log_level_string, _LOG_LEVEL_STRINGS)
        raise argparse.ArgumentTypeError(message)
    log_level_int = getattr(logging, log_level_string, logging.INFO)
    return log_level_int

def logdebug(self, *args):
    line=''.join(str(args[i]) for i in range(len(args)))
    self.log(logging.DEBUG, line)

def loginfo(self, *args):
    line=''.join(str(args[i]) for i in range(len(args)))
    self.log(logging.INFO, line)

def logerror(self, *args):
    line=''.join(str(args[i]) for i in range(len(args)))
    self.log(logging.ERROR, line)

def getMyLogger():
    return logging.getLogger("faceml")

def getLogger(logdir, loglevel, logfile):
    logger = logging.getLogger("faceml")
    logger.setLevel(loglevel)
    ch = getLogHandler(logdir, logfile)
    ch.setLevel(loglevel)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(ch)
    logger.__class__.debug=logdebug
    logger.__class__.info=loginfo
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

def setupDir(targetdir):
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

def flatten(l): 
    return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

def processClassName(classname):
    classes=[]
    notclasses=[]
    if ("[" in classname):     
        for k in classname.split("]"):
            index=k.find("[")
            if index!=-1:
                tstr=k[index+1:]
                if ("not" in k):
                    notclasses.append(tstr)    
                else:
                    classes.append(tstr)

        expr=classname.replace("[","('")
        expr=expr.replace("]","' in objects)")
    else:
        classes=[classname]
        expr="('" + classname + "' in objects)"        
    return classes, notclasses, expr  