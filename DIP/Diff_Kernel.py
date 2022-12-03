import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    img_path = '1.jpg'
    rgb = plt.imread(img_path)
    print(rgb.shape)

    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print(grayscale.shape, grayscale.max(), grayscale.min())

    plt.title('Grayscale')
    plt.imshow(grayscale, cmap='gray')

    kernel1 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)

    processed_img1 = cv2.filter2D(grayscale, -1, kernel1)

    kernel2 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)

    processed_img2 = cv2.filter2D(grayscale, -1, kernel2)

    kernel3 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)

    processed_img3 = cv2.filter2D(grayscale, -1, kernel3)

    kernel4 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)

    processed_img4 = cv2.filter2D(grayscale, -1, kernel4)

    kernel5 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)

    processed_img5 = cv2.filter2D(grayscale, -1, kernel5)

    kernel6 = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.int8)

    processed_img6 = cv2.filter2D(grayscale, -1, kernel6)


    img_set = [grayscale, processed_img1, processed_img2, processed_img3, processed_img4, processed_img5, processed_img6 ]
    title_set = ['Grayscale', 'kernel1', 'kernel2', 'kernel3', 'kernel4', 'kernel5', 'kernel6']
    plot_img(img_set, title_set)


def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize=(40, 40))
    for i in range(n):
        plt.subplot(6, 3, i + 1)
        plt.title(title_set[i])
        plt.imshow(img_set[i], cmap='gray')
    plt.show()

if __name__ == '__main__':
        main()

