
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = plt.imread('1.jpg')
plt.subplot(3,3,1)
plt.title("Original Image")
plt.imshow(img)

r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
grayScale = r*0.299 + g*0.587 + b*0.114
plt.subplot(3,3,2)
plt.title("Gray Scale Image")
plt.imshow(grayScale, cmap = 'gray')




#Gray to Binary
row, col = grayScale.shape
binary = np.zeros((row,col),dtype=np.uint8)
threshold = 127

for i in range(row):
    for j in range(col):
        if grayScale[i,j]>=threshold:
            binary[i,j] = 255
        else:
            binary[i,j] = 0

plt.subplot(3,3,3)
plt.title("Binary Image")
plt.imshow(binary, cmap = 'gray')

#Add Noise

row, col = grayScale.shape
number_of_pixels = random.randint(400,100000)


for i in range(number_of_pixels):
    x_cord = random.randint(0,row-1)
    y_cord = random.randint(0,col-1)
    grayScale[x_cord,y_cord] = 255

for j in range(number_of_pixels):
    x_cord = random.randint(0,row-1)
    y_cord = random.randint(0,col-1)
    grayScale[x_cord,y_cord] = 0

plt.subplot(3,3,4)
plt.title("Adding Noise Image")
plt.imshow(grayScale, cmap = 'gray')


#remove noise
row, col = grayScale.shape
kernel_size =3
remove_noise = np.zeros((row,col),dtype=np.uint8)

for i in range(row):
    for j in range(col):
        remove_noise[i,j] = np.median(grayScale[i:i+kernel_size, j:j+kernel_size])

plt.subplot(3,3,5)
plt.title("Remove Noise Image")
plt.imshow(remove_noise, cmap = 'gray')

height, width = grayScale.shape

''' Generate a binary mask. '''
mask = np.zeros((height, width), dtype=np.uint8)
for i in range(150, 200):
    for j in range(160, 350):
        mask[i, j] == 1

plt.subplot(3,3,6)
plt.title("Mask")
plt.imshow(mask, cmap = 'gray')

''' Apply a binary mask on a grayscale image. '''
result = np.zeros((height, width), dtype=np.uint8)
for i in range(height):
    for j in range(width):
         if mask[i, j] == 1:
             result[i, j] = grayScale[i, j]

plt.subplot(3,3,7)
plt.title("Image with mask")
plt.imshow(result, cmap = 'gray')

'''
mask = np.zeros(grayScale.shape, dtype=np.uint8)
mask = cv2.circle(mask,(250,150),100, (255,255,255), -1)
plt.subplot(3,3,8)
plt.imshow(mask, cmap='gray')


result = cv2.bitwise_and(grayScale,mask)
result[mask==0]=255
plt.subplot(3,3,9)
plt.imshow(result,cmap = 'gray')
'''

plt.show()
