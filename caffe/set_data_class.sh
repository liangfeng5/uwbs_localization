#!/bin/bash
rm data/data_train.h5
rm data/data_val.h5
rm data/data_test.h5
ln -s data_train_${1}.h5 data/data_train.h5  
ln -s data_val_${1}.h5 data/data_val.h5  
ln -s data_test_${1}.h5 data/data_test.h5  

