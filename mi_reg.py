import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.optimize import differential_evolution

FILTER_SIGMA = 1.5
BOUNDS = [(-10, 10), (-10, 10)]  # Bounds (in pixels) supported by mutual information based correlator

def main_mi_reg(ref_image, cmp_image, bounds=BOUNDS):
    """
    Correlator based onMutual Information Algorithm
    http://www.sci.utah.edu/~fletcher/CS7960/slides/Yen-Yun.pdf
    :param ref_image: ndarray, containing reference image data
    :param cmp_image: ndarray, containing comparison image data
    :param bounds: sequence, bounds paramater in scipy.optimize.differential_evolution
    :return: (residual in X, residual in Y, match height)
    """
    obj_func = lambda dx_dy: -__mutual_information(shift(ref_image, dx_dy), cmp_image)
    opt_res = differential_evolution(obj_func, bounds)
    (dx, dy), match_height = -opt_res.x, -opt_res.fun
    return dx, dy, match_height

def __mutual_information(ref_image_crop, cmp_image, bins=256, normed=True):
    """
    :param ref_image_crop: ndarray, cropped image from the center of reference image, needs to be same size as `cmp_image`
    :param cmp_image: ndarray, comparison image data data
    :param bins: number of histogram bins
    :param normed: return normalized mutual information
    :return: mutual information values
    """
    ref_range = (ref_image_crop.min(), ref_image_crop.max())
    cmp_range = (cmp_image.min(), cmp_image.max())
    joint_hist, _, _ = np.histogram2d(ref_image_crop.flatten(), cmp_image.flatten(), bins=bins, range=[ref_range, cmp_range])
    ref_hist, _ = np.histogram(ref_image_crop, bins=bins, range=ref_range)
    cmp_hist, _ = np.histogram(cmp_image, bins=bins, range=cmp_range)
    joint_ent = __entropy(joint_hist)
    ref_ent = __entropy(ref_hist)
    cmp_ent = __entropy(cmp_hist)
    mutual_info = ref_ent + cmp_ent - joint_ent
    if normed:
        mutual_info = mutual_info / np.sqrt(ref_ent * cmp_ent)
    return mutual_info


def __entropy(img_hist):
    """
    :param img_hist: Array containing image histogram
    :return: image entropy
    """
    img_hist = img_hist / float(np.sum(img_hist))
    img_hist = img_hist[np.nonzero(img_hist)]
    return -np.sum(img_hist * np.log2(img_hist))