import argparse
import os
from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser(description="dataset IO to hugging face hub")
    parser.add_argument("seq_to_upload", help="sequence to upload and their IDs")
    parser.add_argument("sub_dirs", help="subdirs to upload")
    parser.add_argument("--seq_root", default="/mnt/nas_cruw/CRUW3D_POSE", help="local sequence root")
    parser.add_argument("--remote_seq_root", default="tmp_split", help="remote sequence root")
    parser.add_argument("--version", default="main", help="remote branch to upload")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    hf_api = HfApi()
    with open(args.seq_to_upload, 'r') as f:
        seq_to_upload = f.readlines()
    seq_to_upload = [line.strip().split(',') for line in seq_to_upload]
    with open(args.sub_dirs, 'r') as f:
        sub_dirs = f.readlines()
    sub_dirs = [line.strip() for line in sub_dirs]
    for seq in seq_to_upload:
        seq_id = seq[0]
        seq_name = seq[1]
        seq_dir = os.path.join(args.seq_root, seq_name)
        for sub_dir in sub_dirs:
            hf_api.upload_folder(
                folder_path=os.path.join(seq_dir, sub_dir),
                path_in_repo=f'{args.remote_seq_root}/{seq_id}/{sub_dir}',
                repo_id="andaba/RT-Pose",
                repo_type="dataset",
                revision=args.version,
            )