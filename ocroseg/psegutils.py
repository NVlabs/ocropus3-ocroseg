from __future__ import print_function

import os

# import sl,morph
import torch
import scipy.ndimage as ndi
from pylab import *
from scipy.ndimage import filters, morphology, interpolation
from torch.autograd import Variable


def sl_width(s):
    return s.stop - s.start


def sl_area(s):
    return sl_width(s[0]) * sl_width(s[1])


def sl_dim0(s):
    return sl_width(s[0])


def sl_dim1(s):
    return sl_width(s[1])


def sl_tuple(s):
    return s[0].start, s[0].stop, s[1].start, s[1].stop


def B(a):
    if a.dtype == dtype('B'):
        return a
    return array(a, 'B')


class record:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def find_on_path(fname, path, separator=":"):
    dirs = path.split(separator)
    for dir in dirs:
        result = os.path.join(dir, fname)
        if os.path.exists(result):
            return result
    return None


def spread_labels(labels, maxdist=9999999):
    """Spread the given labels to the background"""
    distances, features = morphology.distance_transform_edt(
        labels == 0, return_distances=1, return_indices=1)
    indexes = features[0] * labels.shape[1] + features[1]
    spread = labels.ravel()[indexes.ravel()].reshape(*labels.shape)
    spread *= (distances < maxdist)
    return spread


def correspondences(labels1, labels2):
    """Given two labeled images, compute an array giving the correspondences
    between labels in the two images."""
    q = 100000
    assert amin(labels1) >= 0 and amin(labels2) >= 0
    assert amax(labels2) < q
    combo = labels1 * q + labels2
    result = unique(combo)
    result = array([result // q, result % q])
    return result


def blackout_images(image, ticlass):
    """Takes a page image and a ticlass text/image classification image and replaces
    all regions tagged as 'image' with rectangles in the page image.  The page image
    is modified in place.  All images are iulib arrays."""
    rgb = ocropy.intarray()
    ticlass.textImageProbabilities(rgb, image)
    r = ocropy.bytearray()
    g = ocropy.bytearray()
    b = ocropy.bytearray()
    ocropy.unpack_rgb(r, g, b, rgb)
    components = ocropy.intarray()
    components.copy(g)
    n = ocropy.label_components(components)
    print("[note] number of image regions", n)
    tirects = ocropy.rectarray()
    ocropy.bounding_boxes(tirects, components)
    for i in range(1, tirects.length()):
        r = tirects.at(i)
        ocropy.fill_rect(image, r, 0)
        r.pad_by(-5, -5)
        ocropy.fill_rect(image, r, 255)


def binary_objects(binary):
    labels, n = ndi.label(binary)
    objects = ndi.find_objects(labels)
    return objects


def estimate_scale(binary):
    objects = binary_objects(binary)
    bysize = sorted(objects, key=sl_area)
    scalemap = zeros(binary.shape)
    for o in bysize:
        if amax(scalemap[o]) > 0:
            continue
        scalemap[o] = sl_area(o)**0.5
    scale = median(scalemap[(scalemap > 3) & (scalemap < 100)])
    return scale


def compute_boxmap(binary, lo=10, hi=5000, dtype='i'):
    objects = binary_objects(binary)
    bysize = sorted(objects, key=sl_area)
    boxmap = zeros(binary.shape, dtype)
    for o in bysize:
        if sl_area(o)**.5 < lo:
            continue
        if sl_area(o)**.5 > hi:
            continue
        boxmap[o] = 1
    return boxmap


def compute_lines(segmentation, scale):
    """Given a line segmentation map, computes a list
    of tuples consisting of 2D slices and masked images."""
    lobjects = ndi.find_objects(segmentation)
    lines = []
    for i, o in enumerate(lobjects):
        if o is None:
            continue
        if sl_dim1(o) < 2 * scale or sl_dim0(o) < scale:
            continue
        mask = (segmentation[o] == i + 1)
        if amax(mask) == 0:
            continue
        result = record()
        result.label = i + 1
        result.bounds = o
        result.mask = mask
        lines.append(result)
    return lines


def pad_image(image, d, cval=inf):
    result = ones(array(image.shape) + 2 * d)
    result[:, :] = amax(image) if cval == inf else cval
    result[d:-d, d:-d] = image
    return result


def extract(image, y0, x0, y1, x1, mode='nearest', cval=0):
    h, w = image.shape
    ch, cw = y1 - y0, x1 - x0
    y, x = clip(y0, 0, max(h - ch, 0)), clip(x0, 0, max(w - cw, 0))
    sub = image[y:y + ch, x:x + cw]
    # print("extract", image.dtype, image.shape)
    try:
        r = interpolation.shift(sub, (y - y0, x - x0),
                                mode=mode, cval=cval, order=0)
        if cw > w or ch > h:
            pady0, padx0 = max(-y0, 0), max(-x0, 0)
            r = interpolation.affine_transform(r, eye(2), offset=(
                pady0, padx0), cval=1, output_shape=(ch, cw))
        return r

    except RuntimeError:
        # workaround for platform differences between 32bit and 64bit
        # scipy.ndimage
        dtype = sub.dtype
        sub = array(sub, dtype='float64')
        sub = interpolation.shift(
            sub, (y - y0, x - x0), mode=mode, cval=cval, order=0)
        sub = array(sub, dtype=dtype)
        return sub


def extract_masked(image, linedesc, pad=5, expand=0, background=None):
    """Extract a subimage from the image using the line descriptor.
    A line descriptor consists of bounds and a mask."""
    assert amin(image) >= 0 and amax(image) <= 1
    if background is None:
        background = amin(image)
    y0, x0, y1, x1 = [int(x) for x in [linedesc.bounds[0].start, linedesc.bounds[1].start,
                                       linedesc.bounds[0].stop, linedesc.bounds[1].stop]]
    if pad > 0:
        mask = pad_image(linedesc.mask, pad, cval=0)
    else:
        mask = linedesc.mask
    line = extract(image, y0 - pad, x0 - pad, y1 + pad, x1 + pad)
    if expand > 0:
        mask = filters.maximum_filter(mask, (expand, expand))
    line = where(mask, line, background)
    return line


def reading_order(lines, highlight=None, debug=0):
    """Given the list of lines (a list of 2D slices), computes
    the partial reading order.  The output is a binary 2D array
    such that order[i,j] is true if line i comes before line j
    in reading order."""
    order = zeros((len(lines), len(lines)), 'B')

    def x_overlaps(u, v):
        return u[1].start < v[1].stop and u[1].stop > v[1].start

    def above(u, v):
        return u[0].start < v[0].start

    def left_of(u, v):
        return u[1].stop < v[1].start

    def separates(w, u, v):
        if w[0].stop < min(u[0].start, v[0].start):
            return 0
        if w[0].start > max(u[0].stop, v[0].stop):
            return 0
        if w[1].start < u[1].stop and w[1].stop > v[1].start:
            return 1

    if highlight is not None:
        clf()
        title("highlight")
        imshow(binary)
        ginput(1, debug)
    for i, u in enumerate(lines):
        for j, v in enumerate(lines):
            if x_overlaps(u, v):
                if above(u, v):
                    order[i, j] = 1
            else:
                if [w for w in lines if separates(w, u, v)] == []:
                    if left_of(u, v):
                        order[i, j] = 1
            if j == highlight and order[i, j]:
                print((i, j), end=' ')
                y0, x0 = sl.center(lines[i])
                y1, x1 = sl.center(lines[j])
                plot([x0, x1 + 200], [y0, y1])
    if highlight is not None:
        print()
        ginput(1, debug)
    return order


def topsort(order):
    """Given a binary array defining a partial order (o[i,j]==True means i<j),
    compute a topological sort.  This is a quick and dirty implementation
    that works for up to a few thousand elements."""
    n = len(order)
    visited = zeros(n)
    L = []

    def visit(k):
        if visited[k]:
            return
        visited[k] = 1
        for l in find(order[:, k]):
            visit(l)
        L.append(k)

    for k in range(n):
        visit(k)
    return L  # [::-1]


def show_lines(image, lines, lsort):
    """Overlays the computed lines on top of the image, for debugging
    purposes."""
    ys, xs = [], []
    clf()
    cla()
    imshow(image)
    for i in range(len(lines)):
        l = lines[lsort[i]]
        y, x = sl.center(l.bounds)
        xs.append(x)
        ys.append(y)
        o = l.bounds
        r = matplotlib.patches.Rectangle((o[1].start, o[0].start),
                                         edgecolor='r',
                                         fill=0,
                                         width=sl_dim1(o),
                                         height=sl_dim0(o))
        gca().add_patch(r)
    h, w = image.shape
    ylim(h, 0)
    xlim(0, w)
    plot(xs, ys)


def propagate_labels(image, labels, conflict=0):
    """Given an image and a set of labels, apply the labels
    to all the regions in the image that overlap a label.
    Assign the value `conflict` to any labels that have a conflict."""
    rlabels, _ = ndi.label(image)
    cors = correspondences(rlabels, labels)
    outputs = zeros(amax(rlabels) + 1, 'i')
    oops = -(1 << 30)
    for o, i in cors.T:
        if outputs[o] != 0:
            outputs[o] = oops
        else:
            outputs[o] = i
    outputs[outputs == oops] = conflict
    outputs[0] = 0
    return outputs[rlabels]


def remove_noise(line, minsize=8):
    """Remove small componentsfrom an image."""
    if minsize == 0:
        return line
    bin = (line > 0.5 * amax(line))
    labels, n = ndi.label(bin)
    sums = measurements.sum(bin, labels, range(n + 1))
    sums = sums[labels]
    good = minimum(bin, 1 - (sums > 0) * (sums < minsize))
    return good


def remove_big(image, max_h=100, max_w=100):
    """Remove large components."""
    assert image.ndim == 2
    bin = (image > 0.5 * amax(image))
    labels, n = ndi.label(bin)
    objects = ndi.find_objects(labels)
    indexes = ones(n+1, 'i')
    for i, (yr, xr) in enumerate(objects):
        if yr.stop-yr.start < max_h and xr.stop-xr.start < max_w:
            continue
        indexes[i+1] = 0
    indexes[0] = 0
    return indexes[labels]


def hysteresis_threshold(image, lo, hi):
    binlo = (image > lo)
    lablo, n = ndi.label(binlo)
    n += 1
    good = set((lablo * (image > hi)).flat)
    markers = zeros(n, 'i')
    for index in good:
        if index == 0:
            continue
        markers[index] = 1
    return markers[lablo]


class LineSegmenter(object):
    def __init__(self, mname, invert=False, docthreshold=0.5, hiprob=0.5, loprob=None):
        self.hi = hiprob
        self.lo = loprob or hiprob
        self.basic_size = 10
        self.model = torch.load(mname)
        self.invert = invert
        self.model.cuda()
        self.cuinput = Variable(torch.randn(
            1, 1, 100, 100).cuda(), volatile=True)
        self.docthreshold = docthreshold

    def line_probs(self, pimage):
        if pimage.ndim == 3:
            pimage = mean(pimage, 2)
        ih, iw = pimage.shape
        pimage = pimage - amin(pimage)
        pimage /= amax(pimage)
        if self.invert:
            pimage = amax(pimage) - pimage
        self.cuinput.data.resize_(
            1, 1, *pimage.shape).copy_(torch.FloatTensor(pimage))
        cuoutput = self.model(self.cuinput)
        poutput = cuoutput.data.cpu().numpy()[0, 0]
        oh, ow = poutput.shape
        scale = oh * 1.0 / ih
        poutput = ndi.affine_transform(poutput, eye(
            2) * scale, output_shape=pimage.shape, order=1)
        self.probs = poutput
        return poutput

    def line_seeds(self, pimage):
        poutput = self.line_probs(pimage)
        binoutput = hysteresis_threshold(poutput, self.lo, self.hi)
        self.lines = binoutput
        seeds, _ = ndi.label(binoutput)
        return seeds

    def lineseg(self, pimage, max_size=(300, 300)):
        self.image = pimage
        self.binary = pimage > self.docthreshold
        if max_size is not None:
            self.binary = remove_big(self.binary, *max_size)
        self.boxmap = compute_boxmap(self.binary, dtype="B")
        self.seeds = self.line_seeds(pimage)
        self.llabels = propagate_labels(self.boxmap, self.seeds, conflict=0)
        self.spread = spread_labels(self.seeds, maxdist=self.basic_size)
        self.llabels = where(self.llabels > 0, self.llabels,
                             self.spread * self.binary)
        self.segmentation = self.llabels * self.binary
        return self.segmentation


def reordered_lines(lseg):
    lines = compute_lines(lseg, 20)
    order = reading_order([l.bounds for l in lines])
    lsort = topsort(order)
    nlabels = amax(lseg) + 1
    renumber = zeros(nlabels, 'i')
    for i, v in enumerate(lsort):
        renumber[lines[v].label] = 0x010000 + (i + 1)
    renumbered_lseg = renumber[lseg]
    sorted_lines = [lines[i] for i in lsort]
    return sorted_lines, renumbered_lseg


def extract_textlines(lseg, image, pad=5, expand=3):
    lines, segmentation = reordered_lines(lseg)
    for i, l in enumerate(lines):
        grayline = extract_masked(image, l, pad=pad, expand=expand)
        yield grayline, sl_tuple(l.bounds)
