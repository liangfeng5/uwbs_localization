#!/bin/bash
rm data_root
export UWBS_PATH=~/data/poking_data
ln -sf ${UWBS_PATH}/data data_root
rm snapshots
ln -sf ${UWBS_PATH}/caffe_snapshots/snapshots_temp snapshots
rm results
ln -sf ${UWBS_PATH}/results results
