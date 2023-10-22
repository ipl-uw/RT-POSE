# CRUW-POSE
This repository is developed by [UW IPL](https://ipl-uw.github.io/) and contains the code to train human pose estimation network using 4D FMCW radar as input.

It is under development.

## Set up environment

```
conda create -n cruw_pose python=3.9 -y
conda activate cruw_pose
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
cd det3d/ops/dcn && python setup.py build_ext --inplace
cd ../../..
```

## How to run the code
### Train
```
python ./tools/train_radar.py <config file>
```
For example, 
```
python ./tools/train_radar.py configs/cruw_pose/hr3d.py
```
You can optionally provide the working directory. For example,
```
python ./tools/train_radar.py configs/cruw_pose/hr3d.py --work_dir work_dirs/cruw_pose_debug
```

Radar Training:

configs/cruw_pose/hr3d.py

LiDAR training:

configs/cruw_pose/vox.py

### Test
```
python ./tools/dist_test.py <config file> --work_dir <working directory> --checkpoint <checkpoint file> --testset
```
If you want to evaluate on the validation set, you just discard the `--testset` flag in your command.

For more details about the command options, please execute the scripts with `-h`.



### TODO


#### Distributed Training
[] Enable AMP