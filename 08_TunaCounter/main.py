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
 

def process(image, size):

    #REDUCE NOISE -- SHIFT
    shifted = cv2.pyrMeanShiftFiltering(image,size,size) # -- 9,21 arbitrary

    # TO GRAYSCALE
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

    #REMOVE SPECULAR LIGHT
    v,trunc = cv2.threshold(gray,128,128,cv2.THRESH_TRUNC) # -- remove specular light

    cv2.imshow("TRUNC",gray)

    #SUBTRACT BACKGROUND
    gray = cv2.absdiff(trunc,cv2.mean(trunc)[0])

    #cv2.imshow("TEST",gray)

    #ksize = 7
    #g_image = cv2.GaussianBlur(gray,(ksize,ksize),0)
    #l_image = cv2.Laplacian(g_image,cv2.CV_64FC1,ksize=ksize)
    #cv2.imshow("LAPLACE",l_image)

    #APPLY ADAPTIVE THRESHOLD
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,size+1 if size%2==0 else size,2) # -- 13,2 arbitrary
    cv2.imshow("THRSH",thresh) # --> EDGES

    #val, thresh = cv2.threshold(gray,110,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)

    #COMPLETE CONTOUR
    kernel = np.ones((3,3),np.float32) # -- 3,3 arbitrary
    dilated = cv2.dilate(thresh,kernel,iterations = 1) # -- 3 arbitrary
    #cv2.imshow("DILATED",dilated)

    #FILL CONTOUR
    cnts = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    closed = cv2.drawContours(dilated,cnts,-1,(255,255,255),-1)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel) # fill holes

    #KEYPOINTS
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(gray,None)
    kpts = cv2.drawKeypoints(image,kp,image.copy(),color=(255,0,0))
    #cv2.imshow("KeyPoints",kpts)
    return closed

def within(a,b,c):
    return a<b and b<c

def circleArea(r):
    return 3.14*r*r

def identify(image,processed,size):

    D = ndimage.distance_transform_edt(processed.copy())
    localMax = peak_local_max(D, indices=False, min_distance=size/2,
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

    #c = max(contours, key=cv2.contourArea)
    #ar = cv2.contourArea(c) # max area
    ar = circleArea(size)

    valid_contours = [c for c in contours if within(ar * 0.75, cv2.contourArea(c), ar * 1.5)]
    #arbitrary heuristic : valid "tuna" must be at least bigger than half of the biggest one
    identified = image.copy()
    for i,c in enumerate(valid_contours):
            # draw a circle enclosed the object
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(identified, (int(x), int(y)), int(r), (0, 255, 0), 1)
            cv2.putText(identified, "#{}".format(i), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return len(valid_contours), identified

x_prev = 0
y_prev = 0
pts = []
drawing = False

def get_size(event, x, y, flags, param):
    global x_prev
    global y_prev
    global image
    global orig
    global pts
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        x_prev = x
        y_prev = y
        image = orig.copy()
        pts = []
        drawing = True
    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            pts += [[x,y]]
            cv2.line(image,(x_prev,y_prev),(x,y),(255,0,0),1)
            cv2.imshow("Image",image)
            x_prev,y_prev = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        pts = np.asarray(pts)
        #cv2.fitEllipse(pts)
        ar = cv2.contourArea(pts)
        r = int(np.round(np.sqrt(ar/np.pi)))
        print ("R", r)

        processed = process(orig,r)
        n,identified = identify(orig,processed,r)

        cv2.imshow("Image", image)
        cv2.imshow("Processed", processed)
        cv2.imshow("Identified", identified)

#         circleSize = np.zeros((256,256),dtype=np.uint8)
#         cv2.circle(circleSize, (128,128), int(r/2), (255,255,255),1)
#         cv2.circle(circleSize, (128,128), int(r), (255,255,255),1)
#         cv2.circle(circleSize, (128,128), int(r*2), (255,255,255),1)
#         cv2.imshow("Size", circleSize)

        print("[INFO] {} unique segments found".format(n))


BASE_DIR = 'Samples-Hexacopter-Tuna'

#RANDOM IMAGE
SUB_DIR = BASE_DIR + '/' + random.choice(os.listdir(BASE_DIR))
IMG_FILE = SUB_DIR + '/' +random.choice(os.listdir(SUB_DIR))

#IMG_FILE = BASE_DIR + '/' + 'Clear' + '/' + 'P9010878.JPG' 
#IMG_FILE = 'Samples-Hexacopter-Tuna/Test/P9010093.JPG'
#IMG_FILE = 'Samples-Hexacopter-Tuna/Clear/P9011022.JPG'

print 'FILE : {}'.format(IMG_FILE)
orig = cv2.imread(IMG_FILE)
orig = cv2.resize(orig, dsize=(512,512))

# show the output image
mainWindow = cv2.namedWindow("Image")
cv2.imshow("Image", orig)

cv2.setMouseCallback("Image", get_size)
cv2.waitKey(0)
