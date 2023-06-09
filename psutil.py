import cv2
import glob
import numpy as np


def load_lighttxt(filename=None):
    """
    Load light file specified by filename.
    The format of lights.txt should be
        light1_x light1_y light1_z
        light2_x light2_y light2_z
        ...
        lightf_x lightf_y lightf_z

    :param filename: filename of lights.txt
    :return: light matrix (3 \times f)
    """
    if filename is None:
        raise ValueError("filename is None")
    Lt = np.loadtxt(filename)
    return Lt.T


def load_lightnpy(filename=None):
    """
    Load light numpy array file specified by filename.
    The format of lights.npy should be
        light1_x light1_y light1_z
        light2_x light2_y light2_z
        ...
        lightf_x lightf_y lightf_z

    :param filename: filename of lights.npy
    :return: light matrix (3 \times f)
    """
    if filename is None:
        raise ValueError("filename is None")
    Lt = np.load(filename)
    return Lt.T


def load_image(filename=None):
    """
    Load image specified by filename (read as a gray-scale)
    :param filename: filename of the image to be loaded
    :return img: loaded image
    """
    if filename is None:
        raise ValueError("filename is None")
    return cv2.imread(filename, 0)


def load_images(foldername=None, ext=None):
    """
    Load images in the folder specified by the "foldername" that have extension "ext"
    :param foldername: foldername
    :param ext: file extension
    :return: measurement matrix (numpy array) whose column vector corresponds to an image (p \times f)
    """
    if foldername is None or ext is None:
        raise ValueError("filename/ext is None")

    M = None
    height = 0
    width = 0
    for fname in sorted(glob.glob(foldername + "*." + ext)):
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


def load_npyimages(foldername=None):
    """
    Load images in the folder specified by the "foldername" in the numpy format
    :param foldername: foldername
    :return: measurement matrix (numpy array) whose column vector corresponds to an image (p \times f)
    """
    if foldername is None:
        raise ValueError("filename is None")

    M = None
    height = 0
    width = 0
    for fname in sorted(glob.glob(foldername + "*.npy")):
        im = np.load(fname)
        if im.ndim == 3:
            im = np.mean(im, axis=2)
        if M is None:
            height, width = im.shape
            M = im.reshape((-1, 1))
        else:
            M = np.append(M, im.reshape((-1, 1)), axis=1)
    return M, height, width


def disp_normalmap(normal=None, height=None, width=None, delay=0, name=None):
    """
    Visualize normal as a normal map
    :param normal: array of surface normal (p \times 3)
    :param height: height of the image (scalar)
    :param width: width of the image (scalar)
    :param delay: duration (ms) for visualizing normal map. 0 for displaying infinitely until a key is pressed.
    :param name: display name
    :return: None
    """
    N = None
    if normal is None:
        raise ValueError("Surface normal `normal` is None")

    N = np.reshape(normal, (height, width, 3))  # Reshape to image coordinates
    N[:, :, 0], N[:, :, 2] = N[:, :, 2], N[:, :, 0].copy()  # Swap RGB <-> BGR
    N = (N + 1.0) / 2.0  # Rescale
    if name is None:
        name = 'normal map'

    img = N*255
    # cv2.imwrite('normal.png', img)
    cv2.destroyAllWindows()
    cv2.imshow(name, N)
    cv2.waitKey(delay)
    # cv2.destroyWindow(name)
    cv2.waitKey(0)    # to deal with frozen window...


def save_normal_map(normal=None, height=None, width=None, path=None):
    """
    Save normal map as a png image
    :param normal: array of surface normal (p \times 3)
    :param height: height of the image (scalar)
    :param width: width of the image (scalar)
    :param name: display name
    :return: None
    """
    N = None
    if normal is None:
        raise ValueError("Surface normal `normal` is None")

    N = np.reshape(normal, (height, width, 3))  # Reshape to image coordinates
    N[:, :, 0], N[:, :, 2] = N[:, :, 2], N[:, :, 0].copy()  # Swap RGB <-> BGR
    N = (N + 1.0) / 2.0  # Rescale
    if path is None:
        name = 'normal.png'

    img = N*255
    print("path: ", path)
    cv2.imwrite(path, img)


def save_difference_map(gt=None, est=None, mask=None, name=None):
    if gt is None:
        raise ValueError("gt is None")
    if est is None:
        raise ValueError("est is None")
    if mask is None:
        raise ValueError("mask is None")
    if name is None:
        name = 'diff.png'

    est_img = cv2.imread(est)
    gt_img = cv2.imread(gt)

    # subtract the gt from the estimated image
    mask_img = cv2.imread(mask, cv2.IMREAD_COLOR)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    mask_img = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY)[1]

    diff = cv2.subtract(gt_img, est_img, mask=mask_img)
    # diff = (diff + 1.0) / 2.0  # Rescale/
    cv2.imwrite(name, diff)


def save_normalmap_as_npy(filename=None, normal=None, height=None, width=None):
    """
    Save surface normal array as a numpy array
    :param filename: filename of the normal array
    :param normal: surface normal array (height \times width \times 3)
    :return: None
    """
    if filename is None:
        raise ValueError("filename is None")
    N = np.reshape(normal, (height, width, 3))
    np.save(filename, N)


def load_normalmap_from_npy(filename=None):
    """
    Load surface normal array (which is a numpy array)
    :param filename: filename of the normal array
    :return: surface normal (numpy array) in formatted in (height, width, 3).
    """
    if filename is None:
        raise ValueError("filename is None")
    return np.load(filename)


def load_normalmap_from_png(filename=None):
    """
    Load surface normal array (which is a png image)
    :param filename: filename of the normal array
    :return: surface normal (numpy array) in formatted in (height, width, 3).
    """
    if filename is None:
        raise ValueError("filename is None")

    im = None
    height = 0
    width = 0
    im = cv2.imread(filename).astype(np.float64)
    if im is None:
        raise ValueError("filename is not found")

    return im


def evaluate_angular_error(gtnormal=None, normal=None, background=None):
    if gtnormal is None or normal is None:
        raise ValueError("surface normal is not given")
    ae = np.multiply(gtnormal, normal)
    aesum = np.sum(ae, axis=1)
    coord = np.where(aesum > 1.0)
    aesum[coord] = 1.0
    coord = np.where(aesum < -1.0)
    aesum[coord] = -1.0
    ae = np.arccos(aesum) * 180.0 / np.pi
    if background is not None:
        ae[background] = 0
    return ae
