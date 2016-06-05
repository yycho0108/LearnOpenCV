import os,random
import cv2
import numpy as np

from scipy import ndimage
from scipy import stats
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
 
def mode(arr):
    arr = np.asarray(arr).flatten()
    u,ind = np.unique(arr,return_inverse=True)
    return u[np.argmax(np.bincount(ind))]

def process(image, size):
    size = int(np.round(size))

    #REDUCE NOISE -- SHIFT
    shifted = cv2.pyrMeanShiftFiltering(image,size,size) # -- 9,21 arbitrary

    # TO GRAYSCALE
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    # More Contrast
    #gray = cv2.equalizeHist(gray)

    #REMOVE NOISE
    gray = cv2.fastNlMeansDenoising(gray)
    cv2.imshow("GRAY",gray)

    #REMOVE SPECULAR LIGHT
    #trunc = gray
    v,trunc = cv2.threshold(gray,128,128,cv2.THRESH_TRUNC) # -- remove specular light
    cv2.imshow("TRUNC",trunc)

    #SUBTRACT BACKGROUND
    #gray = cv2.absdiff(trunc,float(mode(trunc)))


    #ksize = 7
    #g_image = cv2.GaussianBlur(gray,(ksize,ksize),0)
    #l_image = cv2.Laplacian(g_image,cv2.CV_64FC1,ksize=ksize)
    #cv2.imshow("LAPLACE",l_image)

    #APPLY ADAPTIVE THRESHOLD
    thresh = cv2.adaptiveThreshold(gray,256,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,size+1 if size%2==0 else size,2) # -- 13,2 arbitrary

    #val, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    thresh = cv2.fastNlMeansDenoising(thresh)

    cv2.imshow("THRESH",thresh) # --> EDGES

    #COMPLETE CONTOUR
    k_dilate = np.asarray([
        [0.3,0.5,0.3],
        [0.5,1.0,0.5],
        [0.3,0.5,0.3]
        ],np.float32)
    #kernel = np.ones((3,3),np.float32) # -- 3,3 arbitrary
    dilated = cv2.dilate(thresh,k_dilate,iterations = 1) # -- 3 arbitrary
    #cv2.imshow("DILATED",dilated)

    #FILL CONTOUR
    cnts = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    closed = cv2.drawContours(dilated,cnts,-1,(255,255,255),-1)

    k_fill = np.ones((3,3),np.float32)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k_fill) # fill holes
    eroded = cv2.erode(closed,k_dilate,iterations = 1)

    #KEYPOINTS
    #fast = cv2.FastFeatureDetector_create()
    #kp = fast.detect(gray,None)
    #kpts = cv2.drawKeypoints(image,kp,image.copy(),color=(255,0,0))
    #cv2.imshow("KeyPoints",kpts)
    #eroded = cv2.Canny(eroded,0,255)
    return eroded 

def within(a,b,c):
    return a<b and b<c

def circleArea(r):
    return 3.14*r*r

def identify_blobs(image,processed,size):
    identified = image.copy()


    #BLOB DETECTION ...
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 0

    params.filterByColor = True 
    params.blobColor = 255

    params.filterByArea = True 
    params.minArea = circleArea(size) * 0.3 
    params.maxArea = circleArea(size) * 2.0

    params.filterByCircularity = False

    params.filterByConvexity = True 
    params.minConvexity = 0.5

    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)

    labels = detector.detect(processed)
    cv2.drawKeypoints(identified,labels,identified,color=(255,0,0))
    return len(labels), identified

 
def identify(image,processed,size):
    identified = image.copy()

    D = ndimage.distance_transform_edt(processed.copy())
    localMax = peak_local_max(D, indices=False, min_distance=int(np.round(size)),
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

    valid_contours = [c for c in contours if within(ar * 0.4, cv2.contourArea(c), ar * 2.4)]

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
r = 1.0 # - radius of region

def calculate(r):
    print("R", r)
    global orig
    global image

    hsv = cv2.cvtColor(orig,cv2.COLOR_BGR2HSV)
    processed = process(hsv,r)
    n,identified = identify(orig,processed,r)

    cv2.imshow("Image", image)
    cv2.imshow("Processed", processed)
    cv2.imshow("Identified", identified)

    print("[INFO] {} unique segments found".format(n))


def get_size(event, x, y, flags, param):
    global x_prev
    global y_prev
    global image
    global orig
    global pts
    global drawing
    global r

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
        r = np.sqrt(ar/np.pi)
        calculate(r)

#         circleSize = np.zeros((256,256),dtype=np.uint8)
#         cv2.circle(circleSize, (128,128), int(r/2), (255,255,255),1)
#         cv2.circle(circleSize, (128,128), int(r), (255,255,255),1)
#         cv2.circle(circleSize, (128,128), int(r*2), (255,255,255),1)
#         cv2.imshow("Size", circleSize)



BASE_DIR = 'Samples-Hexacopter-Tuna'

#RANDOM IMAGE
#SUB_DIR = BASE_DIR + '/' + random.choice(os.listdir(BASE_DIR))
#IMG_FILE = SUB_DIR + '/' +random.choice(os.listdir(SUB_DIR))

#IMG_FILE = BASE_DIR + '/' + 'Clear' + '/' + 'P9010878.JPG' 
#IMG_FILE = 'Samples-Hexacopter-Tuna/Test/P9010093.JPG'
IMG_FILE = 'Samples-Hexacopter-Tuna/Clear/P9011022.JPG'
#IMG_FILE = 'Samples-Hexacopter-Tuna/Range/P9011269.JPG'

print 'FILE : {}'.format(IMG_FILE)
orig = cv2.imread(IMG_FILE)
orig = cv2.resize(orig, dsize=(512,512))
image = orig.copy()

# show the output image
mainWindow = cv2.namedWindow("Image")
cv2.imshow("Image", orig)

cv2.setMouseCallback("Image", get_size)

while(1):
    k =cv2.waitKey(0) & 255
    c = chr(k)
    if c == 'q' or k == 27:
        break
    elif c == 'u':
        r *= 1.1
        calculate(r)
    elif c == 'd':
        r *= 0.9
        calculate(r)
    elif c == 'r':
        SUB_DIR = BASE_DIR + '/' + random.choice(os.listdir(BASE_DIR))
        IMG_FILE = SUB_DIR + '/' +random.choice(os.listdir(SUB_DIR))
        print 'FILE : {}'.format(IMG_FILE)
        orig = cv2.imread(IMG_FILE)
        orig = cv2.resize(orig, dsize=(512,512))
        image = orig.copy()
        cv2.imshow("Image", orig)

