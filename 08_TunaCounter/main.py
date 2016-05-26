import os,random
import cv2
import numpy as np

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

# ATTEMPTED METHODS
#EDGE DETECTION
#edges = cv2.Canny(gray, 0,255)
#cv2.imshow("EDGES",edges)

#BACKGROUND SUBTRACTION -- DOESN'T WORK
#cv2.ocl.setUseOpenCL(False)
#fgbg = cv2.createBackgroundSubtractorMOG2()
#thresh = fgbg.apply(shifted)

#CONTOUR DETECTION
#cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#	cv2.CHAIN_APPROX_SIMPLE)[-2]
#for (i, c) in enumerate(cnts):
#	# draw the contour
#	((x, y), _) = cv2.minEnclosedCircle(c)
#	cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
#		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
 
#COUNT BLOBS
# Set up the detector with default parameters.
#detector = cv2.SimpleBlobDetector_create()
# Detect blobs.
#keypoints = detector.detect(image)
#print keypoints
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#image = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 

def process(image):
    image = cv2.resize(image, dsize=(512,512))

    shifted = cv2.pyrMeanShiftFiltering(image,9,15) # -- 9,15 arbitrary

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

    ksize = 13

    g_image = cv2.GaussianBlur(gray,(ksize,ksize),0)
    l_image = cv2.Laplacian(g_image,cv2.CV_64FC1,ksize=ksize)
    cv2.imshow("LAPLACE",l_image)

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,13,2) # -- 13,2 arbitrary
    # --> yields edges

    #val, thresh = cv2.threshold(gray,110,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.float32) # -- 3,3 arbitrary
    dilated = cv2.dilate(thresh,kernel,iterations = 3) # -- 3 arbitrary

    cnts = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

    closed = cv2.drawContours(dilated,cnts,-1,(255,255,255),-1)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)

    #image = cv2.drawContours(image,cnts,-1,(255,255,255),-1)

    return image, closed

def identify(image,processed):

    D = ndimage.distance_transform_edt(processed.copy())
    localMax = peak_local_max(D, indices=False, min_distance=60,
            labels=processed.copy())
     
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=processed.copy())

    contours = []

    for label in np.unique(labels):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                    continue
     
            # otherwise, allocate memory for the label region and draw
            # it on the mask
            mask = np.zeros(processed.shape, dtype="uint8")
            mask[labels == label] = 255
     
            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[-2]
            c = max(cnts, key=cv2.contourArea)
            contours += [c]

    c = max(contours, key=cv2.contourArea)
    ar = cv2.contourArea(c)

    valid_contours = [c for c in contours if cv2.contourArea(c) > 0.5 * ar]
    #arbitrary heuristic : valid "tuna" must be at least bigger than half of the biggest one

    for i,c in enumerate(valid_contours):
            # draw a circle enclosed the object
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.putText(image, "#{}".format(i), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return len(valid_contours)


BASE_DIR = 'Samples-Hexacopter-Tuna'

#RANDOM IMAGE
SUB_DIR = BASE_DIR + '/' + random.choice(os.listdir(BASE_DIR))
IMG_FILE = SUB_DIR + '/' +random.choice(os.listdir(SUB_DIR))

#IMG_FILE = BASE_DIR + '/' + 'Clear' + '/' + 'P9010878.JPG' 
#IMG_FILE = 'Samples-Hexacopter-Tuna/Test/P9010093.JPG'

print 'FILE : {}'.format(IMG_FILE)

image = cv2.imread(IMG_FILE)

image, processed = process(image)
n = identify(image,processed)

print("[INFO] {} unique segments found".format(n))

# show the output image
cv2.imshow("Image", image)
cv2.imshow("Processed", processed)

cv2.waitKey(0)
