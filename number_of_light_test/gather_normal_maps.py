import cv2 as cv
from math import *
import os
import time
import numpy as np
import copy
import csv

import sys
sys.path.insert(1, '../')
import psutil
import psutil2
from rps import RPS


# Choose a method
# METHOD = RPS.L2_SOLVER    # Least-squares
# METHOD = RPS.L1_SOLVER_MULTICORE    # L1 residual minimization
# METHOD = RPS.SBL_SOLVER_MULTICORE    # Sparse Bayesian Learning
METHOD = RPS.RPCA_SOLVER    # Robust PCA

FOLDER_PATH_TO_SAVE = 'holy_metal/'
TMP_PATH = 'tmp/'

FOLDER_PATH = '../data/test/holy_metal/'

##GENERAL LIST
LIGHT_FILENAME = FOLDER_PATH+'light_positions.txt'
FILE_NAMES_TXT = FOLDER_PATH+'filenames.txt'
GS_NORMAL_FILENAME = FOLDER_PATH+'est_normal.png'#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MASK_FILENAME = FOLDER_PATH+'mask.png'

STEP_SIZE = 5

def sort_images_and_lights_with_increasing_angle_to_obejct(lights, img_names):
    sorted_lights = []
    sorted_img_names = []
    sorted_angles = []
    for i in range(len(lights)):
        angle=atan2(lights[i][1],lights[i][0])
        index=0
        while index<len(sorted_angles) and sorted_angles[index]<angle:
            index+=1
        sorted_angles.insert(index,angle)
        sorted_lights.insert(index,lights[i])
        sorted_img_names.insert(index,img_names[i])
        
    return sorted_lights, sorted_img_names

def write_names_to_file(dst_path, img_names):
    f = open(dst_path+'filenames.txt', 'w')
    for i in range(len(img_names)):
        f.write(img_names[i]+'\n')
    f.close()

def write_lights_to_file(dst_path, lights):
    f = open(dst_path+'light_positions.txt', 'w')
    for i in range(len(lights)):
        f.write(str(lights[i][0])+' '+str(lights[i][1])+' '+str(lights[i][2])+'\n')
    f.close()

def copy_images_to_folder(dst_path, src_path, img_names):
    for i in range(len(img_names)):
        img = cv.imread(src_path+img_names[i], cv.IMREAD_COLOR)
        cv.imwrite(dst_path+img_names[i], img)

def get_evenly_distributed_elements(lst, num):
    if num >= len(lst):
        return lst[:]  # Return a copy of the original list if num is greater than or equal to the list length
    else:
        interval = len(lst) / (num)  # Calculate the interval between elements
        indices = [int(round(interval * i)) for i in range(num)]  # Generate the indices of evenly distributed elements
        shifted_indices = [(i) % len(lst) for i in indices]  # Apply the current index shift
        return [lst[i] for i in shifted_indices]  # Create a new list with the selected elements
    
def narrow_to_n_lights(lights, img_names, n):
    tmp_lst = range(0, len(lights))
    # print("tmp_lst: ", tmp_lst)
    tmp_lst = get_evenly_distributed_elements(tmp_lst, n)
    tmp_lights = []
    tmp_img_names = []
    # print("image names: ", img_names)
    for i in range(len(tmp_lst)):
        # print("tmp_lst[i]: ", tmp_lst[i])
        tmp_lights.append(lights[tmp_lst[i]])
        tmp_img_names.append(img_names[tmp_lst[i]])
    return tmp_lights, tmp_img_names

def update_light_and_names(dst_path, src_path, num_images):
    img_names = [line.rstrip('\n') for line in open(src_path+'filenames.txt')]
    lights = psutil.load_lighttxt(filename=src_path+'light_positions.txt').T
    lights, img_names = sort_images_and_lights_with_increasing_angle_to_obejct(lights=lights, img_names=img_names)

    lights, img_names = narrow_to_n_lights(lights=lights, img_names=img_names, n=num_images)


    write_names_to_file(dst_path=dst_path, img_names=img_names)
    write_lights_to_file(dst_path=dst_path, lights=lights)


def delete_tmp_filenames_and_lights(dst_path):
    #check if folder is empty
    if len(os.listdir(dst_path)) == 0:
        return
    os.remove(dst_path+'light_positions.txt')
    os.remove(dst_path+'filenames.txt')


def empty_folder(dst_path):
    if len(os.listdir(dst_path)) == 0:
        return
    for filename in os.listdir(dst_path):
        os.remove(dst_path+filename)

def copy_mask_and_gt_normal(dst_path, src_path):
    mask = cv.imread(src_path+'mask.png', cv.IMREAD_COLOR)
    cv.imwrite(dst_path+'mask.png', mask)
    gt_normal = cv.imread(src_path+'Normal_gt.png', cv.IMREAD_COLOR)
    cv.imwrite(dst_path+'Normal_gt.png', gt_normal)

def copy_names_and_lights(dst_path, src_path):
    img_names = [line.rstrip('\n') for line in open(src_path+'filenames.txt')]
    lights = psutil.load_lighttxt(filename=src_path+'light_positions.txt').T
    write_names_to_file(dst_path=dst_path, img_names=img_names)
    write_lights_to_file(dst_path=dst_path, lights=lights)

def calculate_differece_between_gt_and_estimated_normals(gs, est):
    gs_normal = cv.imread(gs, cv.IMREAD_COLOR)
    #load as float
    gs_normal = gs_normal.astype(np.float32)

    est_normal = cv.imread(est, cv.IMREAD_COLOR)
    #load as float
    est_normal = est_normal.astype(np.float32)
    diff = np.abs(gs_normal - est_normal)
    cv.imwrite('diff.png', diff)
    diff_normalised = diff
    diff_normalised[:,:,0] = diff[:,:,0] / 255

    return diff_normalised
    #calulate error angle in degrees
    # error = np.arccos(np.clip(np.sum(gs_normal*est_normal, axis=2), -1, 1))
    # error = error * 180 / np.pi
    # print("error: ", error)
    # return error


def log_data_to_csv(path, angular_err, time, n):
    with open(path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        mean = np.mean(angular_err[:])
        mean_abs = np.mean(np.abs(angular_err[:]))
        median = np.median(angular_err[:])
        median_abs = np.median(np.abs(angular_err[:]))
        sum = np.sum(np.abs(angular_err[:]))
        std = np.std(angular_err[:])
        max = np.max(angular_err[:])
        min = np.min(angular_err[:])
        writer.writerow([n, mean, mean_abs, median, median_abs, sum, std, max, min, time])
        # csvfile.close()

def main():

    empty_folder(dst_path=TMP_PATH)
    copy_mask_and_gt_normal(dst_path=TMP_PATH, src_path=FOLDER_PATH)
    copy_names_and_lights(dst_path=TMP_PATH, src_path=FOLDER_PATH)    

    img_names = [line.rstrip('\n') for line in open(FILE_NAMES_TXT)]
    copy_images_to_folder(dst_path=TMP_PATH, src_path=FOLDER_PATH, img_names=img_names)
    del img_names

    lights = psutil.load_lighttxt(filename=LIGHT_FILENAME).T
    number_of_images = len(lights)
    print("lights: ", lights[0])
    print("lights name: ", LIGHT_FILENAME)
    print("number_of_images: ", number_of_images)
    #delete lights
    del lights

    rps = RPS()
    rps.load_mask(filename=MASK_FILENAME)    # Load mask image

    #go from number of images to 3
    for i in range(3, number_of_images, STEP_SIZE):
        # print("i: ", i)
        delete_tmp_filenames_and_lights(dst_path=TMP_PATH)
        update_light_and_names(dst_path=TMP_PATH, src_path=FOLDER_PATH, num_images=i)
        tmpsaæølda_lights = psutil.load_lighttxt(filename=TMP_PATH+'light_positions.txt').T
        number_of_lights = len(tmpsaæølda_lights)
        print("number_of_lights: ", number_of_lights)
        
        rps.load_lighttxt(filename=TMP_PATH+'light_positions.txt')    # Load light matrix
        rps.load_images_from_folder(foldername=TMP_PATH, file_txt=TMP_PATH+'filenames.txt')    # Load observations

        start = time.time()
        rps.solve(METHOD)    # Compute
        elapsed_time = time.time() - start
        
        tmp= copy.deepcopy(rps.N)
        current_filename = FOLDER_PATH_TO_SAVE+'est_normal_'+str(i)+'.png'
        psutil.save_normal_map(normal=tmp, height=rps.height, width=rps.width, path=current_filename)

        # N_gt = psutil.load_normalmap_from_png(filename=GS_NORMAL_FILENAME)    # read out the ground truth surface normal
        # N_gt = np.reshape(N_gt, (rps.height*rps.width, 3))    # reshape as a normal array (p \times 3)

        # angular_err = psutil.evaluate_angular_error(N_gt, rps.N, rps.background_ind)    # compute angular error
        angular_err = calculate_differece_between_gt_and_estimated_normals(GS_NORMAL_FILENAME, current_filename)
        print("Mean angular error [deg]: ", np.mean(angular_err[:]))
        print("Time: ", elapsed_time, "s")

        log_data_to_csv(path=FOLDER_PATH_TO_SAVE+'log_file.csv', angular_err=angular_err, time=elapsed_time, n=i)

        


    
#run the main function
main()
    

