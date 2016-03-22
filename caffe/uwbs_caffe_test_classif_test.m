deploy_model = 'models/net_uwbs_fc_3lyr_100neu_entr_out80_deploy.prototxt';

%Model output correspondence to grids: 50mm, 40mm, 20mm, 10mm, 5mm
% x      80 100 200 400 800
% y      40  50 100 200 400

%% Snapshots
snapshot = 'snapshots/caffenet_train_iter_9000.caffemodel'; 


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