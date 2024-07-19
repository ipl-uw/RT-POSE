# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou and Tianwei Yin 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from det3d.core.utils.circle_nms_jit import circle_nms

def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)



def gaussian2D(shape, sigma=1, modulation_coef=1.):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma)) * modulation_coef
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
    
def draw_umich_gaussian(heatmap, center, radius, k=1, modulation_coef=1.):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6, modulation_coef=modulation_coef)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

# Assums we have a 3D i.i.d. Gaussian distribution
def gaussian3D(shape, sigma=1, modulation_coef=1.):
    m, n, p = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-m:m+1,-n:n+1,-p:p+1]
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma)**(3/2)) * modulation_coef
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian3D(heatmap, center, radius, k=1, modulation_coef=1.):
    diameter = 2 * radius + 1
    gaussian = gaussian3D((diameter, diameter, diameter), sigma=diameter / 6, modulation_coef=modulation_coef)

    x, y, z = int(center[0]), int(center[1]), int(center[2])

    height, width, length = heatmap.shape[0:3]

    # naming of direction is described in 3D
    front, rear = min(x, radius), min(length - x, radius + 1)
    right, left = min(y, radius), min(width - y, radius + 1)
    bottom, top = min(z, radius), min(height - z, radius + 1)

    masked_heatmap  = heatmap[z - bottom:z + top, y - right:y + left, x - front:x + rear]
    masked_gaussian = gaussian[radius - bottom:radius + top, radius - right:radius + left, radius - front:radius + rear]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_point3D(heatmap, center, radius):
    x, y, z = int(center[0]), int(center[1]), int(center[2])
    height, width, length = heatmap.shape[0:3]
    if x >= 0 and x < length and y >= 0 and y < width and z >= 0 and z < height:
        heatmap[z, y, x] = 1.0
    return heatmap



def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 4, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(-1)) # BxSpacexClass
    feat = _gather_feat(feat, ind)
    return feat

def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep 


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    w, l = 5, 10
    radius = gaussian_radius((l, w), min_overlap=0.1)
    radius = int(max(radius, 2))
    shape = (20, 20)
    hm = np.zeros(shape)
    center = [1, 1]
    draw_umich_gaussian(hm, center, radius)
    plt.imsave('test_2d_gaussian.png', hm)

    center = [10, 10]
    draw_umich_gaussian(hm, center, radius)
    plt.imsave('test_2d_gaussian.png', hm)

    center = [2, 2]
    draw_umich_gaussian(hm, center, radius)
    plt.imsave('test_2d_gaussian.png', hm)

if __name__ == '__main__':
    det_size_3d = (10, 10, 10)
    print(gaussian_radius_3d(det_size_3d))