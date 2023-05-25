import cv2 as cv

PATH = 'data/test/phone/'

mask = cv.imread(PATH+'mask.png', cv.IMREAD_COLOR)
# mask = mask.astype('float32')
mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)[1]

est_normal = cv.imread(PATH+'est_normal.png')
est_normal = est_normal.astype('float32')

#where the mask is black, set the est_normal to white
for i in range(0, est_normal.shape[0]):
    for j in range(0, est_normal.shape[1]):
        if mask[i,j,0] == 0:
            est_normal[i,j,0] = 255
            est_normal[i,j,1] = 255
            est_normal[i,j,2] = 255



cv.imwrite(PATH+'est_normal_white.png', est_normal)

est_after_fit = cv.imread(PATH+'est_after_fit.png', cv.IMREAD_COLOR)
est_after_fit = est_after_fit.astype('float32')

#where the mask is black, set the est_after_fit to white
for i in range(0, est_after_fit.shape[0]):
    for j in range(0, est_after_fit.shape[1]):
        if mask[i,j,0] == 0:
            est_after_fit[i,j,0] = 255
            est_after_fit[i,j,1] = 255
            est_after_fit[i,j,2] = 255


cv.imwrite(PATH+'est_after_fit_white.png', est_after_fit)
