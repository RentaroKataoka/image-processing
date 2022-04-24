import cv2
import numpy as np

img1 = cv2.imread("./picture2/image1.png")
img2 = cv2.imread("./picture2/image2.png")

im_diff = img1.astype(int) - img2.astype(int)
im_diff_abs = np.abs(im_diff)

cv2.imwrite("./picture2/img3.png", im_diff_abs)