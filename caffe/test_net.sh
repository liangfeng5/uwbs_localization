#!/bin/bash
MODEL_DEF=$1
WEIGHTS=$2
VAL_FILE=data/data_val.h5

caffe_validate_h5.py --model_def ${MODEL_DEF} --pretrained_model ${WEIGHTS} --gpu $VAL_FILE results/out.npy 
#caffe_validatebatch_h5.py --model_def ${MODEL_DEF} --pretrained_model ${WEIGHTS} --gpu $VAL_FILE results/out.npy 
