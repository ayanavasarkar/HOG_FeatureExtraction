import cv2
from sklearn.externals import joblib
#from skimage.feature import hog
import numpy as np

h = 128
w = 64
hog=cv2.HOGDescriptor()
array=np.array([])
# Load the classifier
clf = joblib.load("hog.pkl")

# Read the input image 
im = cv2.imread("alley01.jpg")

img=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
img=cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

roi_hog_fd = hog.compute(img,winStride=(64,128),padding=(0,0))
h_trans=roi_hog_fd.transpose()	


nbr = clf.predict(h_trans)

print nbr
cv2.imshow('output',im)
cv2.waitKey()

