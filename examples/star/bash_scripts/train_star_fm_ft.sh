#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml \
 train.py \
 exp_name=audiocaps_star_ft_speecht5 \
 model=single_task_flow_matching_star_direct \
 data@data_dict=star_audiocaps_speecht5 \
 optimizer.lr=2.5e-5 \
 train_dataloader.collate_fn.pad_keys='["waveform", "duration"]' \
 val_dataloader.collate_fn.pad_keys='["waveform", "duration"]' \
 warmup_params.warmup_steps=1000 \
 epoch_length=Null \
 epochs=100 \
#  +model.pretrained_ckpt="/hpc_stor03/sjtu_home/zeyu.xie/workspace/speech2audio/x2audio/x_to_audio_generation/experiments/audiocaps_fm/checkpoints/epoch_100/model.safetensors" \