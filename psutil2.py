import cv2
import glob
import numpy as np


def load_images_from_folder(foldername=None, file_txt=None): # file_txt is the name of the file that contains the list of images
    """
    Load images in the folder specified by the "foldername" that have extension "txt"
    :param foldername: foldername
    :param ext: file extension
    :return: measurement matrix (numpy array) whose column vector corresponds to an image (p \times f)
    """
    if foldername is None:
        raise ValueError("foldername is None")
    if file_txt is None:
        raise ValueError("file_txt is None")
    
    M = None
    height = 0
    width = 0
    #get each line of file_txt as a string
    img_list = [line.rstrip('\n') for line in open(file_txt)]
    for fname in img_list:
        fname = foldername + '/' + fname
        # print("fname: ", fname)
        im = cv2.imread(fname).astype(np.float64)
        if im.ndim == 3:
            # Assuming that RGBA will not be an input
            im = np.mean(im, axis=2)   # RGB -> Gray
        if M is None:
            height, width = im.shape
            M = im.reshape((-1, 1))
        else:
            M = np.append(M, im.reshape((-1, 1)), axis=1)
    return M, height, width