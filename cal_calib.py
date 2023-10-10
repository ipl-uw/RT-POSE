import json
import numpy as np

with open('/mnt/nas_cruw_pose/left_cam_calib.json', 'r') as f:
    P_LC = np.array(json.load(f)['extrinsic']).reshape(4, 4)

P_RC = np.array([[0.0, -1.0, 0.0, 0.3048], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
P_LR = np.linalg.inv(P_RC) @ P_LC
print(P_LR.flatten().tolist())

