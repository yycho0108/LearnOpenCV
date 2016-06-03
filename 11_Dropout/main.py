import cv2
import numpy as np

def dropout(i,p):
    mask = np.empty(i.shape, dtype=np.int16)
    mask = cv2.randu(mask,0,255)

    val, mask = cv2.threshold(mask,p*255,255,cv2.THRESH_BINARY)
    mask = np.asarray(mask,dtype=np.float64) / 255.0
    return cv2.multiply(i,mask)

I = np.random.rand(5,5)
O = dropout(I,0.5)

print I

print O
