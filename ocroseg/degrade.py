import random as pyr
from math import pi, cos, sin

import numpy as np
import scipy.ndimage as ndi


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
    print "\t", (dx, dy), alpha, scale, aniso
    sm = np.array([[scale / aniso, 0], [0, scale * aniso]], 'f')
    m = np.array([[c, -s], [s, c]], 'f')
    m = np.dot(sm, m)

    def f(image, order=1):
        w, h = image.shape
        c = np.array([w, h]) / 2.0
        d = c - np.dot(m, c) + np.array([dx * w, dy * h])
        return ndi.affine_transform(image, m, offset=d, order=order)

    return f
