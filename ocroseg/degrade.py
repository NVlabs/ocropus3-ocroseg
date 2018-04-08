import random as pyr
from math import pi, cos, sin

import pylab
import numpy as np
import scipy.ndimage as ndi


def make_binary(image):
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    m = np.mean([np.amin(image), np.amax(image)])
    image = 1.0*(image > m)
    return image


def gauss_degrade(image, sigma, noise, maxnoise=1.5, delta_fraction=0.0):
    image = make_binary(image)
    #print np.amin(image), np.amax(image), np.mean(image)
    fraction = np.sum(image) / np.prod(image.shape) * (1.0 + delta_fraction)
    #print fraction
    smoothed = ndi.gaussian_filter(image, sigma)
    smoothed -= np.amin(smoothed)
    smoothed /= np.amax(smoothed)
    #print np.amin(smoothed), np.amax(smoothed), np.mean(smoothed)
    noisy = smoothed + noise * \
        np.clip(pylab.randn(*image.shape), -maxnoise, maxnoise)
    #print np.amin(noisy), np.amax(noisy), np.mean(noisy)
    threshold = np.percentile(noisy.flat, 100.0*(1-fraction))
    #print threshold
    return 1.0 * (noisy > threshold)


def random_trs(translation=0.05, rotation=2.0, scale=0.1, aniso=0.1):
    if not isinstance(translation, (tuple, list)):
        translation = (-translation, translation)
    if not isinstance(rotation, (tuple, list)):
        rotation = (-rotation, rotation)
    if not isinstance(scale, (tuple, list)):
        scale = (-scale, scale)
    if not isinstance(aniso, (tuple, list)):
        aniso = (-aniso, aniso)
    dx = pyr.uniform(*translation)
    dy = pyr.uniform(*translation)
    alpha = pyr.uniform(*rotation)
    alpha = alpha * pi / 180.0
    scale = 10**pyr.uniform(*scale)
    aniso = 10**pyr.uniform(*aniso)
    c = cos(alpha)
    s = sin(alpha)
    # print "\t", (dx, dy), alpha, scale, aniso
    sm = np.array([[scale / aniso, 0], [0, scale * aniso]], 'f')
    m = np.array([[c, -s], [s, c]], 'f')
    m = np.dot(sm, m)

    def f(image, order=1):
        w, h = image.shape
        c = np.array([w, h]) / 2.0
        d = c - np.dot(m, c) + np.array([dx * w, dy * h])
        return ndi.affine_transform(image, m, offset=d, order=order, mode="nearest")

    return f, dict(translation=(dx, dy), alpha=alpha, scale=scale, aniso=aniso)


def random_distort(images, maxdelta=2.0, sigma=30.0):
    n, m = images[0].shape
    deltas = pylab.rand(2, n, m)
    deltas = ndi.gaussian_filter(deltas, (0, sigma, sigma))
    deltas -= np.amin(deltas)
    deltas /= np.amax(deltas)
    deltas = (2*deltas-1) * maxdelta
    #print np.amin(deltas), np.amax(deltas)
    xy = np.transpose(np.array(np.meshgrid(
        range(n), range(m))), axes=[0, 2, 1])
    #print(xy.shape, deltas.shape)
    deltas += xy
    return [ndi.map_coordinates(image, deltas, order=1) for image in images]

def random_blobs(shape, numblobs, size, roughness=2.0):
    from random import randint
    h, w = shape
    mask = np.zeros((h, w), 'i')
    for i in xrange(numblobs):
        mask[randint(0, h-1), randint(0, w-1)] = 1
    dt = ndi.distance_transform_edt(1-mask)
    mask =  np.array(dt < size, 'f')
    mask = ndi.gaussian_filter(mask, size/(2*roughness))
    mask -= np.amin(mask)
    mask /= np.amax(mask)
    noise = pylab.rand(h, w)
    noise = ndi.gaussian_filter(noise, size/(2*roughness))
    noise -= np.amin(noise)
    noise /= np.amax(noise)
    return np.array(mask * noise > 0.5, 'f')
