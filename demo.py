from __future__ import print_function
import cv2 as cv

import numpy as np
import time
from rps import RPS
import psutil

import copy

# Choose a method
# METHOD = RPS.L2_SOLVER    # Least-squares
# METHOD = RPS.L1_SOLVER_MULTICORE    # L1 residual minimization
METHOD = RPS.SBL_SOLVER_MULTICORE    # Sparse Bayesian Learning
# METHOD = RPS.RPCA_SOLVER    # Robust PCA

# Choose a dataset
# DATA_FOLDERNAME = './data/bunny/bunny_specular/'    # Specular with cast shadow
DATA_FOLDERNAME = './data/bunny/bunny_lambert/'    # Lambertian diffuse with cast shadow
# DATA_FOLDERNAME = './data/bunny/bunny_lambert_noshadow/'    # Lambertian diffuse without cast shadow
# DATA_FOLDERNAME = './data/bunny/3images/'    # small sample set

LIGHT_FILENAME = './data/bunny/lights.npy'
MASK_FILENAME = './data/bunny/mask.png'
GT_NORMAL_FILENAME = './data/bunny/gt_normal.npy'

#BUDDHA
# DATA_FOLDERNAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/buddhaPNG/images/'
# LIGHT_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/buddhaPNG/light_directions.txt'
# MASK_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/buddhaPNG/mask.png'
# GT_NORMAL_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/buddhaPNG/Normal_gt.png'

#READING
# DATA_FOLDERNAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/readingPNG/images/'
# LIGHT_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/readingPNG/light_directions.txt'
# MASK_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/readingPNG/mask.png'
# GT_NORMAL_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/readingPNG/Normal_gt.png'

#READING
# DATA_FOLDERNAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/readingPNG/images3/'
# LIGHT_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/readingPNG/light_directions3.txt'
# MASK_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/readingPNG/mask.png'
# GT_NORMAL_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData_crop/readingPNG/Normal_gt.png'

#CAT3
# DATA_FOLDERNAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData/catPNG/images3/'
# LIGHT_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData/catPNG/light_directions3.txt'
# MASK_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData/catPNG/mask.png'
# GT_NORMAL_FILENAME = '/home/thor/Documents/8.Semester/Project/SDPS-Net/data/datasets/DiLiGenT/pmsData/catPNG/Normal_gt.png'



# Photometric Stereo
rps = RPS()
rps.load_mask(filename=MASK_FILENAME)    # Load mask image
rps.load_lightnpy(filename=LIGHT_FILENAME)    # Load light matrix
# rps.load_lighttxt(filename=LIGHT_FILENAME)    # Load light matrix
rps.load_npyimages(foldername=DATA_FOLDERNAME)    # Load observations
# rps.load_images(foldername=DATA_FOLDERNAME, ext="png")    # Load observations
start = time.time()

print("rps.M.shape: ", rps.M.shape)
print("rps.L.shape: ", rps.L.shape)
rps.solve(METHOD)    # Compute
elapsed_time = time.time() - start
print("Photometric stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")
rps.save_normalmap(filename="./est_normal")    # Save the estimated normal map
# tmp= copy.deepcopy(rps.N)
# psutil.disp_normalmap(normal=tmp, height=rps.height, width=rps.width)

# Evaluate the estimate
N_gt = psutil.load_normalmap_from_npy(filename=GT_NORMAL_FILENAME)    # read out the ground truth surface normal
# N_gt = psutil.load_normalmap_from_png(filename=GT_NORMAL_FILENAME)    # read out the ground truth surface normal
print("N_gt.shape: ", N_gt.shape)
print("image shape: ", rps.height, rps.width)
N_gt = np.reshape(N_gt, (rps.height*rps.width, 3))    # reshape as a normal array (p \times 3)

angular_err = psutil.evaluate_angular_error(N_gt, rps.N, rps.background_ind)    # compute angular error
print("Mean angular error [deg]: ", np.mean(angular_err[:]))
tmp= copy.deepcopy(rps.N)
psutil.disp_normalmap(normal=tmp, height=rps.height, width=rps.width)
# tmp= copy.deepcopy(rps.N)
# psutil.disp_normalmap(normal=tmp, height=rps.height, width=rps.width)
# tmp= copy.deepcopy(rps.N)
# psutil.disp_normalmap(normal=tmp, height=rps.height, width=rps.width)
# tmp= copy.deepcopy(rps.N)
# psutil.disp_normalmap(normal=tmp, height=rps.height, width=rps.width)
# tmp= copy.deepcopy(rps.N)
# psutil.disp_normalmap(normal=tmp, height=rps.height, width=rps.width)
# tmp= copy.deepcopy(rps.N)
# psutil.disp_normalmap(normal=tmp, height=rps.height, width=rps.width)
# tmp= copy.deepcopy(rps.N)
# psutil.disp_normalmap(normal=tmp, height=rps.height, width=rps.width)
# tmp= copy.deepcopy(rps.N)
# psutil.disp_normalmap(normal=tmp, height=rps.height, width=rps.width)

print("done.")