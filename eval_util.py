import numpy as np
from numpy.linalg import norm

# pred: Nx3, gt: Nx3
def PJPE(pred, gt) -> np.ndarray:
    pred -= pred[:1, :]
    gt -= gt[:1, :]
    return ABS_PJPE(pred, gt)

def ABS_PJPE(pred, gt):
    return norm(pred - gt, axis=-1)