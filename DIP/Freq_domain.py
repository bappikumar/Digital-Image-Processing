## Solution code for frequency domain

import matplotlib.pyplot as plt
import numpy as np
import cv2

img = plt.imread("1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

g = np.fft.fft2(gray)
g_shift = np.fft.fftshift(g)

M, N = gray.shape       # M->Row, N->Col
H = np.zeros((M, N), dtype=np.float32)

# parameter D0 control the shape of our gaussian filter.

D0 = 10   # cut off frequency
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M/2) ** 2 + (v - N/2) ** 2)
        H[u, v] = np.exp((-D**2) / (2 * D0 * D0))


# low pass
plt.subplot(2,1,1)
plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(g_shift * H))), cmap='gray')



plt.show()
