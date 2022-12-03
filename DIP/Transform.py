import cv2
import numpy as np
from matplotlib import pyplot as plt

img  = plt.imread("1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

gray.shape


# 1
negative = np.zeros(gray.shape)

mx = gray.max()
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        negative[i, j] = mx - gray[i, j]

# 2
new_img = gray.copy()
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        if gray[i, j] >= 50 and gray[i, j] <=150:
            new_img[i, j] = 200

plt.subplot(1,2,1)
plt.imshow(negative, cmap='gray')
plt.title("1")

plt.subplot(1,2,2)
plt.imshow(new_img, cmap='gray')
plt.title('2')

plt.show()
