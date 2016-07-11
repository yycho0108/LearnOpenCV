#!/usr/bin/python
import os, random

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)

BASE_DIR = 'Samples-Hexacopter-Tuna'
SUB_DIR = BASE_DIR + '/' + 'TimeSeries'

#RANDOM IMAGE
#SUB_DIR = BASE_DIR + '/' + random.choice(os.listdir(BASE_DIR))

#IMG_FILE = BASE_DIR + '/' + 'Clear' + '/' + 'P9010878.JPG' 
#IMG_FILE = 'Samples-Hexacopter-Tuna/Test/P9010093.JPG'
#IMG_FILE = 'Samples-Hexacopter-Tuna/Clear/P9011022.JPG'
#IMG_FILE = 'Samples-Hexacopter-Tuna/Range/P9011269.JPG'
#IMG_FILE = 'Samples-Hexacopter-Tuna/Clear/P9010975.JPG'

fgbg = cv2.createBackgroundSubtractorKNN(1)
#fgbg = cv2.createBackgroundSubtractorMOG2()

while cv2.waitKey() != 27:
    IMG_FILE = SUB_DIR + '/' + random.choice(os.listdir(SUB_DIR))
    #print 'FILE : {}'.format(IMG_FILE)

    orig = cv2.imread(IMG_FILE)
    frame = cv2.resize(orig,(0,0),fx=0.125,fy=0.125)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',frame)
    cv2.imshow('fgmask',fgmask)

cv2.destroyAllWindows()
