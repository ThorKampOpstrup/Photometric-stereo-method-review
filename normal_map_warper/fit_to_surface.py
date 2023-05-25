import cv2 as cv

PATH = '../data/test/viking_fibo/'

# surface_gt = cv.imread('curved_surface_normal.png')
surface_gt = cv.imread('high_angles_to_surface.png')
# do a small gausian blur on the gt surface
surface_gt = cv.GaussianBlur(surface_gt, (3,3), 0)
surface_gt = surface_gt.astype('float32')


to_fit = cv.imread(PATH+'est_normal.png')
to_fit = to_fit.astype('float32')
mask = cv.imread(PATH+'mask.png', cv.IMREAD_COLOR)

mask = mask.astype('float32')

#set all value in mask to either 0 or 255 based 0 or not
mask = cv.threshold(mask, 0, 1, cv.THRESH_BINARY)[1]


# perfect = cv.imread('curved_surface_normal.png')
# perfect = perfect.astype('float32')

# perfect[:,:,0] = 255
# perfect[:,:,1] = 128
# perfect[:,:,2] = 128

difference = cv.imread('curved_surface_normal.png')
difference = difference.astype('float32')
#subtract the perfect from the surface_gt
difference[:,:,0] = surface_gt[:,:,0] - 255.
difference[:,:,1] = surface_gt[:,:,1] - 128.
difference[:,:,2] = surface_gt[:,:,2] - 128.
# print(difference[:,:,0])


cv.imwrite('difference.png', difference)
# mask = mask / 255

#subtract difference from to_fit
to_fit[:,:,0] = to_fit[:,:,0] - (difference[:,:,0] * mask[:,:,0])
to_fit[:,:,1] = to_fit[:,:,1] - (difference[:,:,1] * mask[:,:,0])
to_fit[:,:,2] = to_fit[:,:,2] - (difference[:,:,2] * mask[:,:,0])

# print(mask)
for i in range(0, to_fit.shape[0]):
    for j in range(0, to_fit.shape[1]):
        val = mask[i,j,0]
        # print(val)
        if val != 0. and val != 1.:
            print("found: ", i," j: ",j)


#set all pixels where the mask is 0 to 0
# to_fit[:,:,0] = to_fit[:,:,0] * mask[:,:,0]
# to_fit[:,:,1] = to_fit[:,:,1] * mask[:,:,0]
# to_fit[:,:,2] = to_fit[:,:,2] * mask[:,:,0]

# to_fit[:,:,0] = 0


# print out all pixels where the mask
for i in range(0, to_fit.shape[0]):
    for j in range(0, to_fit.shape[1]):
        if to_fit[i,j,0] < 128 and to_fit[i,j,0] > 0:
            # print("found: ", i," j: ",j)
            to_fit[i,j,0] = 129
        # if mask[i,j,0] == 0:
        #     continue
        # print(to_fit[i,j,:])

cv.imwrite(PATH+'est_after_fit.png', to_fit)