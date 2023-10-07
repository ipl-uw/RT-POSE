import numpy as np
from numpy.linalg import norm

def MPJPE(pred, gt):
    pred -= pred[:1, :]
    gt -= gt[:1, :]
    return ABS_MPJPE(pred, gt)

def ABS_MPJPE(pred, gt):
    return norm(pred - gt, axis=-1).mean()