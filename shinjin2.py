import cv2

img = cv2.imread("./picture1/image1.jpg")
# cv2.imshow("image", img)
# cv2.waitKey(0)

height = img.shape[0]
width = img.shape[1]

img2 = cv2.resize(img, (int(width*2), int(height*2)))
img3 = cv2.resize(img, (int(width*0.5), int(height*0.5)))

center = (int(width/2), int(height/2))
angle = 45.0
scale = 1.0
trans = cv2.getRotationMatrix2D(center, angle , scale)

img4 = cv2.warpAffine(img, trans, (width, height))

threshold = 100
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img5 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

cv2.imwrite("./picture1/image2.jpg", img2)
cv2.imwrite("./picture1/image3.jpg", img3)
cv2.imwrite("./picture1/image4.jpg", img4)
cv2.imwrite("./picture1/image5.jpg", img5)
