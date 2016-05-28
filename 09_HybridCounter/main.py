import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn import metrics
from matplotlib import pyplot as plt
import renders as rs

import os
import random

def show(name,image):
    resized = cv2.resize(image,dsize=(512,512))
    cv2.imshow(name,resized)

#image = cv2.imread('P9010878.jpg')
DIR = '/home/jamiecho/Projects/Learn/OpenCV/08_TunaCounter/Samples-Hexacopter-Tuna/Clear'

image = cv2.imread(DIR + '/' + random.choice(os.listdir(DIR)))
image = cv2.resize(image,(512,512))
#image = np.asarray(
#        [[0,0,1,0,0],
#         [0,1,1,1,0],
#         [1,1,1,1,1],
#         [0,1,1,1,0],
#         [0,0,1,0,0]],np.float32)
#image /= 2.0
#
#D = ndimage.distance_transform_edt(image)
#print D
#
#show("image",image)
#show("D",D)


gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
fast = cv2.FastFeatureDetector_create()
#sift = cv2.xfeatures2d.SIFT_create()

kp = fast.detect(gray,None)
image=cv2.drawKeypoints(gray,kp,image,color=(255,0,0))


kp = np.asarray([k.pt for k in kp])
kp = pd.DataFrame(kp,columns=['Dimension 1','Dimension 2'])
#print kp
#
scores_K = []
scores_G = []
n_range = range(2,20)
repeat = 1 #for smoother scores

for n in n_range:
    scores_K_n = []
    scores_G_n = []
    print n
    
    for _ in range(repeat):
        clusterer_K = KMeans(n_clusters=n)
        clusterer_G = GMM(n_components=n)

        # TODO: Predict the cluster for each data point
        preds_K = clusterer_K.fit_predict(kp)
        preds_G = clusterer_G.fit_predict(kp)

        # TODO: Find the cluster centers
        centroids_K = clusterer_K.cluster_centers_ 
        centroids_G = clusterer_G.means_

        # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
        scores_K_n += [metrics.silhouette_score(kp,preds_K)]
        scores_G_n += [metrics.silhouette_score(kp,preds_G)]
        
    #print n, scores_K_n
    scores_K += [np.average(scores_K_n)]
    scores_G += [np.average(scores_G_n)]

print pd.DataFrame(data={'Kmeans':scores_K,'GMM':scores_G},index=n_range)

plt.plot(n_range,scores_K)
plt.plot(n_range,scores_G)
plt.legend(['KMeans','GMM'])
plt.show()

#clusterer = GMM(n_components=12)
clusterer = KMeans(n_clusters=n_range[np.argmax(scores_K)])

preds = clusterer.fit_predict(kp)
#centers = clusterer.means_
centers = clusterer.cluster_centers_

#print clusterer.weights_

rs.cluster_results(kp,preds,centers,np.asarray([(0,0)]))

cv2.imwrite('sift_keypoints.jpg',image)
#cv2.imshow('keypoints',image)
plt.show()

