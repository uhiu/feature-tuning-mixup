#!/bin/bash

# Set GPU device
export CUDA_VISIBLE_DEVICES=1

# Run FTM attack with debug mode
python main.py \
    --model_name ResNet50 \
    --eval \
    --debug

# Run FTM-E attack with debug mode
python main.py \
    --model_name ResNet50 \
    --ensemble_size 2 \
    --eval \
    --debug