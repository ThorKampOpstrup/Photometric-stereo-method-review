from __future__ import print_function
import cv2 as cv

import numpy as np
import time
from rps import RPS
import psutil
import psutil2

import copy

# Choose a method
# METHOD = RPS.L2_SOLVER    # Least-squares
# METHOD = RPS.L1_SOLVER_MULTICORE    # L1 residual minimization
# METHOD = RPS.SBL_SOLVER_MULTICORE    # Sparse Bayesian Learning
METHOD = RPS.RPCA_SOLVER    # Robust PCA

# Choose a dataset
# DATA_FOLDERNAME = './data/bunny/bunny_specular/'    # Specular with cast shadow
# DATA_FOLDERNAME = './data/bunny/bunny_lambert/'    # Lambertian diffuse with cast shadow
# DATA_FOLDERNAME = './data/bunny/bunny_lambert_noshadow/'    # Lambertian diffuse without cast shadow
# DATA_FOLDERNAME = './data/bunny/3images/'    # small sample set

# LIGHT_FILENAME = './data/bunny/lights.npy'
# MASK_FILENAME = './data/bunny/mask.png'
# GT_NORMAL_FILENAME = './data/bunny/gt_normal.npy'

#PATHS FOR DILIGENT

# FOLDER_PATH = 'data/DiLiGenT/pmsData/ballPNG/'
# FOLDER_PATH = 'data/DiLiGenT/pmsData/bearPNG/'
# FOLDER_PATH = 'data/DiLiGenT/pmsData/buddhaPNG/'
# FOLDER_PATH = 'data/DiLiGenT/pmsData/catPNG/'
# FOLDER_PATH = 'data/DiLiGenT/pmsData/cowPNG/'
# FOLDER_PATH = 'data/DiLiGenT/pmsData/gobletPNG/'
# FOLDER_PATH = 'data/DiLiGenT/pmsData/harvestPNG/'
# FOLDER_PATH = 'data/DiLiGenT/pmsData/pot1PNG/'
# FOLDER_PATH = 'data/DiLiGenT/pmsData/pot2PNG/'
# FOLDER_PATH = 'data/DiLiGenT/pmsData/readingPNG/'

##!OWN DATA
# FOLDER_PATH = 'data/test/mug/'
# FOLDER_PATH = 'data/test/mouse/'
# FOLDER_PATH = 'data/test/hand/'
# FOLDER_PATH = 'data/test/bottle/'
# FOLDER_PATH = 'data/test/phone/'
# FOLDER_PATH = 'data/test/peter/'
# FOLDER_PATH = 'data/test/bottle_painted/'
# FOLDER_PATH = 'data/test/large_figure/'
# FOLDER_PATH = 'data/test/small_half_sphere/'
# FOLDER_PATH = 'data/test/rod/'
# FOLDER_PATH = 'data/test/small_head/'
# FOLDER_PATH = 'data/test/small_head_camera_close/'
FOLDER_PATH = 'data/test/brick/'

##GENERAL LIST
#! LIGHT_FILENAME = FOLDER_PATH+'light_directions.txt'
LIGHT_FILENAME = FOLDER_PATH+'light_positions.txt'
FILE_NAMES_TXT = FOLDER_PATH+'filenames.txt'
GT_NORMAL_FILENAME = FOLDER_PATH+'Normal_gt.png'
MASK_FILENAME = FOLDER_PATH+'mask.png'

psutil2.load_images_from_folder(foldername=FOLDER_PATH, file_txt=FILE_NAMES_TXT)

# Photometric Stereo
rps = RPS()
rps.load_mask(filename=MASK_FILENAME)    # Load mask image
# rps.load_lightnpy(filename=LIGHT_FILENAME)    # Load light matrix
rps.load_lighttxt(filename=LIGHT_FILENAME)    # Load light matrix
# rps.load_npyimages(foldername=DATA_FOLDERNAME)    # Load observations
# rps.load_images(foldername=DATA_FOLDERNAME, ext="png")    # Load observations
rps.load_images_from_folder(foldername=FOLDER_PATH, file_txt=FILE_NAMES_TXT)    # Load observations
start = time.time()

print("rps.M.shape: ", rps.M.shape)
print("rps.L.shape: ", rps.L.shape)
rps.solve(METHOD)    # Compute
elapsed_time = time.time() - start
print("Photometric stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")
rps.save_normalmap(filename=FOLDER_PATH+"est_normal")    # Save the estimated normal map
# tmp= copy.deepcopy(rps.N)
# psutil.disp_normalmap(normal=tmpos.remove(path_to_save_images+"file.txt"), height=rps.height, width=rps.width)
cv.imwrite("normal_est.png", rps.N)
print("rps.N.shape: ", rps.N.shape)

# Evaluate the estimate
# N_gt = psutil.load_normalmap_from_npy(filename=GT_NORMAL_FILENAME)    # read out the ground truth surface normal
N_gt = psutil.load_normalmap_from_png(filename=GT_NORMAL_FILENAME)    # read out the ground truth surface normal
print("N_gt.shape: ", N_gt.shape)
print("image shape: ", rps.height, rps.width)
N_gt = np.reshape(N_gt, (rps.height*rps.width, 3))    # reshape as a normal array (p \times 3)

angular_err = psutil.evaluate_angular_error(N_gt, rps.N, rps.background_ind)    # compute angular error
print("Mean angular error [deg]: ", np.mean(angular_err[:]))
tmp= copy.deepcopy(rps.N)
psutil.save_normal_map(normal=tmp, height=rps.height, width=rps.width, path=FOLDER_PATH+'est_normal.png')
tmp= copy.deepcopy(rps.N)
# psutil.save_difference_map(GT_NORMAL_FILENAME,  est='normal.png', name='diff.png', mask=MASK_FILENAME)

print("done.")