deploy_model = 'models/net_uwbs_fc_n900_out80_deploy.prototxt';

%Model output correspondence to grids: 5cm 
% x      80
% y      40

%% Snapshots
snapshot = 'snapshots/caffenet_train_iter_10000.caffemodel'; 


%% Lbl values
test_files.gt_meta = 'data_root/uwbs_ground_00/train_val/meta_data.mat';

%% Validataion files
test_files.all_val_data = 'data/data_test.mat';
test_files.h5_val       = 'data/data_test.h5';


%% Execution
[test_res] = uwbs_caffe_test_classif( ...
        deploy_model, ...
        snapshot, ...
        'x', ...
        test_files );