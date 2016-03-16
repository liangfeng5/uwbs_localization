deploy_model = 'models/net_poke_fc_n900_out80_deploy.prototxt';

%Model output correspondence to grids:(2cm;10deg / 1cm; 5deg / 0.5cm; 2.5deg)
% x      20 / 40 / 80
% y      10 / 20 / 40
% z      15 / 30 / 60 

%% Snapshots
snapshot = '/media/Store_2/poking_data/results/kirk_pb_wood__pac_test_xyzyp__1cm5deg__2016_03_01/plastic_box/cross_01/best_yaw__iter_004700__acc_0.624.caffemodel'; 


%% Lbl values
test_files.lbl_values = '/media/Store_2/poking_data/data/plastic_box_lefthand_vert00__2016_02_07/train_val_xyzyp_1cm5deg_pac/label_values.mat';

%% Validataion files
test_files.all_val_data = 'data_root/plastic_box_lefthand_vert00__2016_02_07/train_val_xyzyp_1cm5deg_test_pac/cross_01/data_test.mat';
test_files.h5_val       = 'data_root/plastic_box_lefthand_vert00__2016_02_07/train_val_xyzyp_1cm5deg_test_pac/cross_01/data_test_x.h5';


%% Execution
[test_res] = uwbs_caffe_test_classif( ...
        deploy_model, ...
        snapshot, ...
        'x', ...
        test_files );