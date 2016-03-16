#!/bin/bash
#caffe train --solver=solver_ilsvrc14.prototxt
#caffe train --solver=models/solver_regr1d.prototxt 
caffe train --solver=models/solver_test.prototxt 
#caffe train --solver=solver_ilsvrc14.prototxt --weights snapshots/train_lwd_ilsvrc_120x120_i35000/caffenet_train_iter_35000.caffemodel
#caffe train --solver=solver_ilsvrc14.prototxt --weights snapshots/train_lwd_ilsvrc_90x90/caffenet_train_iter_30000.caffemodel
#caffe train --solver=solver_ilsvrc14.prototxt --weights snapshots/train_lwd_ilsvrc_90x90_dataset_v10/caffenet_train_iter_10000.caffemodel
#caffe train --solver=solver_ilsvrc14_short.prototxt --weights snapshots/best/ilsvrc14_acc94.5_i11000_dataset_v10.caffemodel
