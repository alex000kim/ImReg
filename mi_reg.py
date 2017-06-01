import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.optimize import differential_evolution

FILTER_SIGMA = 1.5
BOUNDS = [(-10, 10), (-10, 10)]  # Bounds (in pixels) supported by mutual information based correlator

def main_mi_reg(ref_chip, cmp_chip, bounds=BOUNDS):
    """
    Correlator based onMutual Information Algorithm
    http://www.sci.utah.edu/~fletcher/CS7960/slides/Yen-Yun.pdf
    :param ref_chip: ndarray, containing reference chip data
    :param cmp_chip: ndarray, containing comparison chip data
    :param bounds: sequence, bounds paramater in scipy.optimize.differential_evolution
    :return: (residual in X, residual in Y, match height)
    """
    obj_func = lambda dx_dy: -__mutual_information(shift(ref_chip, dx_dy), cmp_chip)
    opt_res = differential_evolution(obj_func, bounds)
    (dx, dy), match_height = -opt_res.x, -opt_res.fun
    return dx, dy, match_height

def __mutual_information(ref_chip_crop, cmp_chip, bins=256, normed=True):
    """
    :param ref_chip_crop: ndarray, cropped image from the center of reference chip, needs to be same size as `cmp_chip`
    :param cmp_chip: ndarray, comparison chip data data
    :param bins: number of histogram bins
    :param normed: return normalized mutual information
    :return: mutual information values
    """
    ref_range = (ref_chip_crop.min(), ref_chip_crop.max())
    cmp_range = (cmp_chip.min(), cmp_chip.max())
    joint_hist, _, _ = np.histogram2d(ref_chip_crop.flatten(), cmp_chip.flatten(), bins=bins, range=[ref_range, cmp_range])
    ref_hist, _ = np.histogram(ref_chip_crop, bins=bins, range=ref_range)
    cmp_hist, _ = np.histogram(cmp_chip, bins=bins, range=cmp_range)
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