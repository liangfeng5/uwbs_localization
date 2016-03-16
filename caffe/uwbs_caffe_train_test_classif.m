function [ results ] = uwbs_caffe_train_test_classif( solvers, dirs, data_files, varargin )
%% Description:
% the function trains different classifiers for the dataset (such as
% combined classifier for 3D and classifiers for individual dimensions and
% spits out the results of the classification
%
% [ results ] = uwbs_caffe_train_classif( solver, dirs, data_files )
% --- INPUT:
% solvers = is a structure of solver filenames to train different classifiers. 
%   Must contain fields:
%   .xyz/.x/.y/.z = i.e. 4 solvers is required
% dirs = different directories:
%   .data = directory containing h5 files:
%       - xyz = data labeled in 3d grid
%       - x/y/z = data labeled for individual dimensions
%   .model = directory where model is (also used as directory for temporary solver file)
%   .results = directory to save results into
% data_files = different auxiliary data files required for training and testing
%   .lbl_values = file containing all label values (such as what is the
%   value associated with particular label, etc ..)
%
% --- OUTPUT:
% results = structure containing all results that you need

%% Notes
% data directories will be substituted (soft link)
% I would recommend to substitute the snapshot directory too, just to see
% what it is saving at any moment

%% Parameters
snapshot_prefix_def = [dirs.results '/best_classif'];
gpu_mode = 1;
save_intermediate_snapshot = 0;

%% Arguments
var_i = 1;
classifiers = {'x', 'y', 'z'};
if length(varargin) >= var_i
    classifiers = varargin{var_i};
end

% Creating data structure
% Creating the folder structure
if ~exist(dirs.model, 'dir')
    mkdir(dirs.model);
end

% Creating the folder structure
if ~exist(dirs.results, 'dir')
    mkdir(dirs.results);
end

% Changing soft link for data folder
fprintf('%s : changing soft links for data dir to %s ...\n', mfilename, dirs.data );
delete('data');
sys_res = system(['ln -s ' dirs.data ' data']);
test_files.h5_val       = 'data/data_test.h5';
test_files.all_val_data = 'data/data_test.mat';
test_files.gt_meta   = data_files.gt_meta;

%% Execution
solver_filename{length(classifiers)} = '';
net_model_deploy{length(classifiers)} = '';

for class_i = 1:length(classifiers)
    fprintf('----- %s : Class = %s training ...\n', mfilename, classifiers{class_i});
    solver_filename{class_i} = solvers.(classifiers{class_i});
    solver_struct{class_i} = caffe_read_solverprototxt(solver_filename{class_i});
    snapshot_prefix{class_i} = snapshot_prefix_def;
    if isfield(solver_struct{class_i}, 'snapshot_prefix')
        snapshot_prefix{class_i} = solver_struct{class_i}.snapshot_prefix;
    end
    
    % Forming deploy model name
    [net_model.path net_model.name net_model.ext] = fileparts(solver_struct{class_i}.net);
    net_model_deploy{class_i} = [net_model.path '/' net_model.name '_deploy' net_model.ext];
    
    fprintf('%s : Changing soft links for train val files ...\n', mfilename);
    sys_res = system('rm -f data/data_val.h5');
    sys_res = system('rm -f data/data_train.h5');
    sys_res = system('rm -f data/data_test.h5');
    sys_res = system(sprintf('ln -s ./data_val_%s.h5   data/data_val.h5', classifiers{class_i}));
    sys_res = system(sprintf('ln -s ./data_train_%s.h5 data/data_train.h5', classifiers{class_i}));
    sys_res = system(sprintf('ln -s ./data_test_%s.h5  data/data_test.h5', classifiers{class_i}));

    [ results.(classifiers{class_i}).loss_train_val, ...
        results.(classifiers{class_i}).best_iter, ...
        results.(classifiers{class_i}).train_stat ] ...
        = ...
        caffe_trainloss_net_shell(...
        solver_filename{class_i}, ...
        [ dirs.results '/best_' classifiers{class_i} ], ...
        [ dirs.results '/log_training_' classifiers{class_i} '.txt']);
    
    [test_res] = uwbs_caffe_test_classif( ...
        net_model_deploy{class_i}, ...
        results.(classifiers{class_i}).train_stat.best_snapshot_name, ...
        classifiers{class_i}, ...
        test_files );
    results.(classifiers{class_i}) = catstruct(results.(classifiers{class_i}), test_res);
end

% --- What expect giving estimation from individual coordinates (separate
% classifiers cumulative results)
if isfield(results, 'x') && isfield(results, 'y') && isfield(results, 'z')
    results.xyz_sep.error_vec = [results.x.error_vec, results.y.error_vec, results.z.error_vec];
    results.xyz_sep.error = sqrt( sum( abs(results.xyz_sep.error_vec).^2, 2  ) );
end

% classifiers cumulative results)
if isfield(results, 'x') && isfield(results, 'y')
    results.xy_sep.error_vec = [results.x.error_vec, results.y.error_vec];
    results.xy_sep.error = sqrt( sum( abs(results.xyz_sep.error_vec).^2, 2  ) );
end

end

