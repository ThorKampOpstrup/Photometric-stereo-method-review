import cv2 as cv

PATH = 'data/test/small_half_sphere/'

mask = cv.imread(PATH+'mask.png', cv.IMREAD_COLOR)
mask = mask.astype('float32')

#set all value in mask to either 0 or 255 based 0 or not
mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)[1]

cv.imwrite(PATH+'mask.png', mask)

# for i in range(0, mask.shape[0]):
#     for j in range(0, mask.shape[1]):
#         val = mask[i,j,0]
#         # print(val)
#         if val == 0.0 or val == 1.0:
#             print("found: ", i," j: ",j)