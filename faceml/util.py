import os
import datetime
from datetime import date

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


def tostr(*args):
    retval=""
    for i in range(len(args)):
        retval=retval + " " + str(args[i])
    return retval

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