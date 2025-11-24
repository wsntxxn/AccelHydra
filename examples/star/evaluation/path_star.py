# Usage:
# export CLAP_MODEL_PATH=/cpfs02/shared/speechllm/xuxuenan/hf_cache/hub/models--lukewys--laion_clap/snapshots/b3708341862f581175dba5c356a4ebf74a9b6651/630k-audioset-best.pt
# python evaluation/tta.py \
#   --ref_audio_jsonl /cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/audiocaps/test/audio.jsonl \
#   --ref_caption_jsonl  /cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/audiocaps/test/caption.jsonl \
#   --gen_audio_dir /cpfs_shared/jiahao.mei/code/x_to_audio_generation/xxx/tta_infer \
#   --output_file evaluation/result/tta.jsonl \
#   -c 16

# gt
# python evaluation/tta.py \
#   --ref_audio_jsonl /cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/audiocaps/test/audio.jsonl \
#   --ref_caption_jsonl  /cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/audiocaps/test/caption.jsonl \
#   --gen_audio_dir /cpfs_shared/jiahao.mei/code/x_to_audio_generation/data/audiocaps/test/audio \
#   --output_file evaluation/result/tta.jsonl \
#   -c 16

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
from collections import defaultdict

import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

# Ref: https://github.com/haoheliu/audioldm_eval/tree/main
# This script uses a locally modified version of audioldm_eval.
from audioldm_eval import EvaluationHelper

# Ref: https://github.com/LAION-AI/CLAP
# The ref command for installing: pip install laion-clap
import laion_clap
import sys
sys.path.append("/hpc_stor03/sjtu_home/yixuan.li/work/x_to_audio_generation")
from utils.general import read_jsonl_to_mapping, audio_dir_to_mapping

import os
import shutil
from pathlib import Path
from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
from path_tta import compute_clap_metrics, AudioTextDataset, evaluate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_audio_dir",
        "-r",
        type=str,
        default="/hpc_stor03/sjtu_home/zeyu.xie/workspace/data/audiocaption_data/audiocaps/audio/test_16k",
        help="path to reference audio jsonl file"
    )
    parser.add_argument(
        "--ref_caption_jsonl",
        "-rc",
        type=str,
        default="/hpc_stor03/sjtu_home/zeyu.xie/workspace/speech2audio/x2audio/x_to_audio_generation/data/audiocaps/test/caption.jsonl",
        help="path to reference caption jsonl file"
    )
    parser.add_argument(
        "--gen_audio_dir",
        "-gd",
        type=str,
        help="path to generated audio directory"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        default="",
        help="path to output file"
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="sta_test",
    )
    parser.add_argument(
        "--num_workers",
        "-c",
        default=4,
        type=int,
        help="number of workers for parallel processing"
    )
    parser.add_argument(
        "--clap_per_audio",
        "-p",
        action="store_true",
        help="calculate and store CLAP score for each audio clip"
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        help="recalculate embeddings for metric scores"
    )

    args = parser.parse_args()

    evaluate(args)
