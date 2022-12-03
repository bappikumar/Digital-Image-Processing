import matplotlib.pyplot as plt
import cv2
import numpy as np

def conv(mat, kernel):
    row, col = mat.shape
    r, c = kernel.shape[0] // 2, kernel.shape[1] // 2
    r, c = r * 2, c * 2

    new_image = np.zeros((row - r, col - c), dtype=np.uint8)
    for i in range(row - r):
        for j in range(col - c):
            temp = np.sum(np.multiply(mat[i:3+i, j:3+j], kernel))
            if temp > 255:
                new_image[i][j] = 255
            elif temp < 0:
                new_image[i][j] = 0
            else:
                new_image[i][j] = temp

    return new_image 

def main():
    img  = plt.imread("1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # blur_kernel = np.array([[1/9,1/9,1/9], [1/9,1/9,1/9], [1/9,1/9,1/9]])


    avg = np.ones((3,3))/9   
    avg_filtered = conv(gray, avg)
    median_filtered = cv2.medianBlur(gray, 3)
    plt.subplot(1,2,1)
    plt.imshow(avg_filtered, cmap='gray')
    plt.title("Filtered By Horizontal Kernel")
    plt.subplot(1,2,2)
    plt.imshow(median_filtered, cmap='gray')
    plt.title("Filtered By Median")


    plt.show()

main()