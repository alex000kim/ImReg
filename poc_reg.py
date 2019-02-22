import numpy as np
import scipy
import scipy.fftpack
from numpy import pi, sin
from scipy.optimize import leastsq

"""
"High-AccuracySubpixelImageRegistrationBasedonPhase-OnlyCorrelation"
http://www.aoki.ecei.tohoku.ac.jp/research/docs/e86-a_8_1925.pdf
"""

FITTING_SHAPE = (8, 8)
EPS = 0.001

def zero_padding(src, shape, pos):
    y, x = (int(pos[0]), int(pos[1]))
    padded_image = np.zeros(shape)
    padded_image[y:src.shape[0] + y, x:src.shape[1] + x] = src
    return padded_image


def poc_model(al, dt1, dt2, poc, fitting_area):
    N1, N2 = poc.shape
    V1, V2 = map(lambda x: 2 * x + 1, fitting_area)
    return lambda n1, n2: al * sin((n1 + dt1) * V1 / N1 * pi) * sin((n2 + dt2) * V2 / N2 * pi) \
                          / (sin((n1 + dt1) * pi / N1) * sin((n2 + dt2) * pi / N2) * (N1 * N2))


def pocfunc(ref_image, cmp_image):
    # Windowing to reduce boundary effects
    hanning_window_x = np.hanning(ref_image.shape[0])
    hanning_window_y = np.hanning(ref_image.shape[1])
    hanning_window_2d = hanning_window_x.reshape(hanning_window_x.shape[0], 1) * hanning_window_y
    ref_image, cmp_image = [ref_image, cmp_image] * hanning_window_2d
    F = scipy.fftpack.fft2(ref_image)
    G = scipy.fftpack.fft2(cmp_image)
    G_ = np.conj(G)
    R = F * G_ / np.abs(F * G_)  # cross-phase spectrum
    R = scipy.fftpack.fftshift(R)

    # Spectral weighting technique to reduce aliasing and noise effects
    M = np.floor([ref_image.shape[0] / 2.0, ref_image.shape[1] / 2.0])  # M = [M1, M2]
    U = M / 2.0  # U = [U1, U2]
    low_pass_filter = np.ones([int(M[0]) + 1, int(M[1]) + 1])
    low_pass_filter = zero_padding(low_pass_filter, ref_image.shape, U)
    R = R * low_pass_filter
    R = scipy.fftpack.fftshift(R)
    poc = scipy.fftpack.fftshift(np.real(scipy.fftpack.ifft2(R)))
    return poc


def main_poc_reg(ref_image, cmp_image):
    # calculate phase-only correlation
    poc = pocfunc(ref_image, cmp_image)
    # get peak position peak
    max_pos = np.argmax(poc)
    peak = np.array([max_pos / ref_image.shape[1], max_pos % ref_image.shape[1]]).astype(int)
    # fitting using least-square method
    mc = np.array([FITTING_SHAPE[0] / 2.0, FITTING_SHAPE[1] / 2.0])
    fitting_area = poc[int(peak[0] - mc[0]): int(peak[0] + mc[0] + 1), int(peak[1] - mc[1]): int(peak[1] + mc[1] + 1)]
    if fitting_area.shape != (FITTING_SHAPE[0] + 1, FITTING_SHAPE[1] + 1):
        return 0.0, 0.0, 0.0
    m = np.array([ref_image.shape[0] / 2.0, ref_image.shape[1] / 2.0])
    u = (m / 2)
    y, x = np.mgrid[-mc[0]:mc[0] + 1, -mc[1]:mc[1] + 1]
    y = np.ceil(y + peak[0] - m[0])
    x = np.ceil(x + peak[1] - m[1])
    def error_func(p):
        poc_model_values = poc_model(p[0], p[1], p[2], poc, u)(y, x)
        error = poc_model_values - fitting_area
        return np.ravel(error)
    p0 = np.array([0.0, -(peak[0] - m[0]) - EPS, -(peak[1] - m[1]) - EPS])
    estimate = leastsq(error_func, p0)
    match_height = estimate[0][0]
    dx = -estimate[0][1]
    dy = -estimate[0][2]
    return dx, dy, match_height
