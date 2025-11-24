#!/bin/bash
export HYDRA_FULL_ERROR=1

accelerate launch --config-file configs/accelerate/nvidia/1gpu.yaml \
    inference_multi_gpu.py \
    ckpt_dir=/hpc_stor03/sjtu_home/yixuan.li/work/x_to_audio_generation/experiments/audiocaps_star_ft_speecht5/checkpoints/best \
    data@data_dict=star_audiocaps_speecht5 \
    infer_args.guidance_scale=5.0 \
    infer_args.num_steps=50 \
    test_dataloader.collate_fn.pad_keys='[]'