import numpy as np
import cv2 as cv

img = cv.imread('plate/mask.png')
count = np.count_nonzero(img)
print(count)