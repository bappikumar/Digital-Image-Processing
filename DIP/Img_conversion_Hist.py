import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("1.jpg")
plt.subplot(3,3,1)
plt.title("Original Image")
plt.imshow(img)


Img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(3,3,3)
plt.title("Gray Scale Image")
plt.imshow(Img_gray, cmap = "gray")

plt.hist(img.ravel(), bins = 256, range = [0,256])
plt.subplot(3,3,2)
plt.hist(Img_gray.ravel(), bins = 256, range = [0,256])
plt.subplot(3,3,4)



plt.show()