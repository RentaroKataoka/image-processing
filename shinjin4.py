import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./picture3/image2.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# ヒストグラム
# plt.hist(img.ravel(), 256, [0, 256])
# plt.show()


# SIFT
# sift = cv2.SIFT_create()
# kp, ds = sift.detectAndCompute(gray, None)

# img_sift = cv2.drawKeypoints(gray, kp, None, flags=4)

# cv2.imwrite("./picture3/sift_keypoints2.jpg", img_sift)


#ORBを使った総当たりマッチング
import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('./picture3/image2.jpg',0)          # queryImage
img2 = cv2.imread('./picture3/image2-2.jpg',0) # trainImage

orb = cv2.ORB_create()# Initiate ORB detector

kp1, des1 = orb.detectAndCompute(img1,None)# find the keypoints and descriptors with ORB
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)# create BFMatcher object

matches = bf.match(des1,des2)# Match descriptors.

matches = sorted(matches, key = lambda x:x.distance)# Sort them in the order of their distance.

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100], None, flags=2)# Draw first 10 matches.

cv2.imwrite('./picture3/image2-3.jpg', img3)
