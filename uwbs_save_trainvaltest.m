function [ crossval_indx ] = uwbs_save_trainvaltest( gt, filenames, precision )
%% Description:
% Given ground truth save train/val/test
% --- Arguments:
% [ crossval_indx ] = uwbs_save_trainvaltest( gt, train_prop, filenames )


%% Execution:

%% Divide into training and validation sets
samp_num = size(gt.data, 1);

val_size  = samp_num * gt.train_prop(2);
test_size = samp_num * gt.train_prop(3);

% crossval_iter_num = int32(1.0 / (1.0 - train_ratio));
crossval_iter_num = 1;
crossval_indx{crossval_iter_num} = 0;

indices_shuffled = randperm(samp_num);
for val_i=1:crossval_iter_num
    val_start_i = (val_i - 1) * (val_size + test_size) + 1;
    val_end_i   = val_start_i + val_size - 1;
    
    test_start_i = val_end_i + 1;
    test_end_i = test_start_i + test_size - 1;
    
    crossval_indx{val_i}.val  = indices_shuffled( val_start_i:val_end_i );
    crossval_indx{val_i}.test = indices_shuffled( test_start_i:test_end_i );
    crossval_indx{val_i}.train   = setdiff( indices_shuffled, crossval_indx{val_i}.val );
    crossval_indx{val_i}.train   = setdiff( crossval_indx{val_i}.train, crossval_indx{val_i}.test );
end

% fprintf('Train indices: \n');
% crossval_indx{val_i}.train
% 
% fprintf('Val indices: \n');
% crossval_indx{val_i}.val
% 
% fprintf('Test indices: \n');
% crossval_indx{val_i}.test


%% Saving data
for cross_i=1:crossval_iter_num
    % --- New version
    train_indx = crossval_indx{cross_i}.train;
    val_indx   = crossval_indx{cross_i}.val;
    test_indx  = crossval_indx{cross_i}.test;
        
    %Making filenames
    crossval_dir = [filenames.output_dir '/' ...
        filenames.crossval_prefix sprintf('_%02d', cross_i) '/'];
    % Creating the folder structure
    if ~exist(crossval_dir, 'dir')
        mkdir(crossval_dir);
    end
    
    h5_train_fullname  = [crossval_dir  filenames.h5_train];
    h5_val_fullname    = [crossval_dir  filenames.h5_val];
    h5_test_fullname   = [crossval_dir  filenames.h5_test];
    
    fprintf('%s : HDF5: Training file = %s \n', mfilename, h5_train_fullname);
    uwbs_save_h5(gt, train_indx, h5_train_fullname, precision);
    fprintf('%s : HDF5: Validation file = %s \n', mfilename, h5_val_fullname);
    uwbs_save_h5(gt, val_indx, h5_val_fullname, precision);
    fprintf('%s : HDF5: Test file = %s \n', mfilename, h5_test_fullname );
    uwbs_save_h5(gt, test_indx, h5_test_fullname, precision);
end

%% Saving other data
% --- Saving meta data
%Saving some metadata
meta_except = {'feat', 'data', 'labels', 'prob'};
gt_meta = gt;
rmfield(gt_meta, meta_except);
save([filenames.output_dir '/meta_data'], 'gt_meta');

% --- Saving all data
save([filenames.output_dir '/all_data'], 'gt', 'crossval_indx');