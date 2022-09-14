
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

img = plt.imread('1.jpg')


fft_img = np.fft.fft2(img)

fft_img_sort = np.sort(np.abs(fft_img.ravel()))

print(img.shape,fft_img.shape,fft_img_sort.shape)

n = len(fft_img_sort)
img_set = [img]
title_set = ['Source Image']

for keep in(0.75, 0.5, 0.1, 0.01, 0.001, 0.0001):
    thresh = fft_img_sort[int(np.floor(n*(1-keep)))]
    ind = np.abs(fft_img)>thresh
    allow_pass = fft_img*ind
    ifft_img = np.fft.fft2(allow_pass).real

    img_set.append(ifft_img)
    title_set.append("compressed (keep = {}%)".format(keep*100))

def plot_img(img_set , title_set):
    n = len(img_set)
    r , c = 2,4
    plt.figure(figsize=(20,20))
    for i in range(n):
        plt.subplot(r,c,i+1)
        plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
    
    plt.savefig('output.png')
    plt.show()

plot_img(img_set,title_set)