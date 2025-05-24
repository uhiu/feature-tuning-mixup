#!/bin/bash

# Set GPU device
export CUDA_VISIBLE_DEVICES=2

# Set model name
MODEL=ResNet50

# Run FTM attack
python main.py \
    --model_name $MODEL \
    --save_dir ./exp/$MODEL/ftm \
    --eval

# Run FTM-E attack 
python main.py \
    --model_name $MODEL \
    --save_dir ./exp/$MODEL/ftm_e \
    --ensemble_size 2 \
    --eval
