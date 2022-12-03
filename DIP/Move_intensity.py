import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("1.jpg")
plt.figure(figsize=(15,10))
plt.subplot(331)
plt.title("Original Image")
plt.imshow(img)

ImgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(332)
plt.title("GrayScale Image")
plt.imshow(ImgGray, cmap = 'gray')
plt.subplot(333)
hist = cv2.calcHist(ImgGray,[0], None, [100], [0,255])
plt.title("Normal Intensity Image Histogram")
plt.plot(hist)

ImgGray2 = ImgGray+50
plt.subplot(334)
plt.title("High Intensity")
plt.imshow(ImgGray2, cmap = 'gray')
plt.subplot(335)
hist = cv2.calcHist(ImgGray2,[0], None, [100], [0,255])
plt.title("High Intensity Image Histogram")
plt.plot(hist)

ImgGray3 = ImgGray-20
plt.subplot(336)
plt.title("Low Intensity")
plt.imshow(ImgGray3, cmap = 'gray')
plt.subplot(337)
hist = cv2.calcHist(ImgGray3,[0], None, [100], [0,255])
plt.title("Low Intensity Image Histogram")
plt.plot(hist)

ImgGray4 = ImgGray*2
plt.subplot(338)
plt.title("Medium Intensity")
plt.imshow(ImgGray4, cmap = 'gray')
plt.subplot(339)
hist = cv2.calcHist(ImgGray4,[0], None, [100], [0,255])
plt.title("Medium Intensity Image Histogram")
plt.plot(hist)

plt.show()