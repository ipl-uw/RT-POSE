import argparse
import copy
import json
import os
import sys
# workaround
sys.path.append('/home/andy/ipl/CRUW-POSE')
# workaround


try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time
from det3d.torchie.utils import count_parameters 
from pathlib import Path
from collections import defaultdict


'''
viz format example:
 each frame: [{
    "obj_id": "0",
    "obj_type": "Sedan",
    "psr": {
      "position": {
        "x": 26.323807590819317,
        "y": -4.672016669736319,
        "z": -0.2779390447352759
      },
      "rotation": {
        "x": 0,
        "y": 0,
        "z": -0.016496852089850397
      },
      "scale": {
        "x": 3.280036683179418,
        "y": 1.9069884147384841,
        "z": 1.4570745923842519
      }
    }
  }]
'''

def save_pred(pred, root, checkpoint_name, dataset_split):
    with open(os.path.join('/mnt/ssd3/cruw_pose_label/file_meta_merge.txt'), 'r') as f:
        lines = f.readlines()
    seq_id_to_name = {}
    for line in lines:
        seq_id, seq_name = line.strip().split(',')
        seq_id_to_name[seq_id] = seq_name

    save_pred_dir = os.path.join(root, f"{checkpoint_name}")
    os.makedirs(save_pred_dir, exist_ok=True)
    # Sort pred by seq
    result = defaultdict(dict)
    for seq_rdr_frame, val in pred.items():
        seq, frame, rdr_frame = seq_rdr_frame.split('/')
        result[seq_id_to_name[seq]][f'{frame}_{rdr_frame}'] = val
        # workaround
        # result['2024_0218_1209'][f'{frame}_{rdr_frame}'] = val
    # sort result by seq name, and sort each seq by frame
    result = dict(sorted(result.items(), key=lambda x: x[0]))
    for seq, frames in result.items():
        result[seq] = dict(sorted(frames.items(), key=lambda x: int(x[0].split('_')[0])))
    with open(os.path.join(save_pred_dir, f"{dataset_split}_prediction.json"), "w") as f:
        json.dump(result, f, indent=2)







def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", required=True, help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        args.work_dir = os.path.dirname(args.checkpoint)
        cfg.work_dir = args.work_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_device
    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    logger.info(f'Model parameter count: {count_parameters(model)}')
    

    cfg['DATASET']['MODE'] = 'test'
    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        # model = fuse_bn_recursively(model)
        model = model.cuda()

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 /3)

    time_start = 0 
    time_end = 0 

    for i, data_batch in enumerate(data_loader):
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )
        for output in outputs:
            metadata = output.pop('metadata')
            seq_name = metadata['seq']
            frame = metadata['frame']
            rdr_frame_name = metadata['rdr_frame']
            detections.update(
                {f'{seq_name}/{frame}/{rdr_frame_name}': output,}
            )
            if args.local_rank == 0:
                prog_bar.update()
            

    synchronize()

    all_predictions = all_gather(detections)

    try:
        print("\n Total time per frame: ", (time_end -  time_start) / (end - start)) # TODO: fix bug
    except:
        pass

    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    checkpoint_file_name = args.checkpoint.split('/')[-1].split('.')[0]
    save_pred(predictions, args.work_dir, checkpoint_file_name, 'test' if args.testset else 'train')

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")
    if 'seq_results' in result_dict:
        with open(os.path.join(os.path.join(args.work_dir, checkpoint_file_name),\
                                f"{checkpoint_file_name}_seq_results_{'test' if args.testset else 'train'}.json"), "w") as f:
            json.dump(dict(sorted(result_dict['seq_results'].items())), f, indent=2)
    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
 