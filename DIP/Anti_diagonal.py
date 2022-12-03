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

    horz_kernel = np.array([[-1,-1,-1], [2,2,2], [-1,-1,-1]])
    vert_kernel = np.array([[-1,2,-1], [-1,2,-1], [-1,2,-1]])
    
    '''45=[
        -1 -1 2
        -1 2 -1
        2 -1 -1
    ]
    
    -45 = [
        2 -1 -1 
        -1 2 -1
        -1 -1 2
    ]'''

    horz = conv(gray, horz_kernel)    
    vert = conv(gray, vert_kernel)    

    plt.subplot(1,2,1)
    plt.imshow(horz, cmap='gray')
    plt.title("Filtered By Diagonal Kernel")
    plt.subplot(1,2,2)
    plt.imshow(vert, cmap='gray')
    plt.title("Filtered By Anti-Diagonal Kernel")


    plt.show()

main()