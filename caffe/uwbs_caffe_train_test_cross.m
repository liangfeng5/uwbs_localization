%% Description:
% Run crossvalidations for all objects
%
% --- NOTES:
% Results for an object: object{}.results.x/y/z
% Results for crossval:  object{}.cross{}.results.x/y/z

%% Parameters:
data_root = 'data_root';
dataset_root = 'dataset_root';
% data_train_val = 'train_val';

crossval_prefix = 'cross_';
results_root    = 'results';

% --- Objects
%Model output correspondence to grids: 50mm, 40mm, 20mm, 10mm, 5mm
% x      80 100 200 400 800
% y      40  50 100 200 400

grid_cur = 0;

% grid_cur = grid_cur + 1;
% grid_res{grid_cur}.name = 'sigma_005mm';
% grid_res{grid_cur}.dir  = 'train_val_grid010mm_sigma005mm';
% grid_res{grid_cur}.solvers.x_prob = 'models/solver_x.prototxt';
% grid_res{grid_cur}.solvers.y_prob = 'models/solver_y.prototxt';
% grid_res{grid_cur}.solvers.z_prob = 'models/solver_z.prototxt';

grid_cur = grid_cur + 1;
grid_res{grid_cur}.name = 'sigma_020mm';
grid_res{grid_cur}.dir  = 'train_val_grid040mm_sigma020mm';
grid_res{grid_cur}.solvers.x_prob = 'models/solver_x.prototxt';
grid_res{grid_cur}.solvers.y_prob = 'models/solver_y.prototxt';
grid_res{grid_cur}.solvers.z_prob = 'models/solver_z.prototxt';

grid_cur = grid_cur + 1;
grid_res{grid_cur}.name = 'sigma_040mm';
grid_res{grid_cur}.dir  = 'train_val_grid040mm_sigma040mm';
grid_res{grid_cur}.solvers.x_prob = 'models/solver_x.prototxt';
grid_res{grid_cur}.solvers.y_prob = 'models/solver_y.prototxt';
grid_res{grid_cur}.solvers.z_prob = 'models/solver_z.prototxt';

grid_cur = grid_cur + 1;
grid_res{grid_cur}.name = 'sigma_160mm';
grid_res{grid_cur}.dir  = 'train_val_grid040mm_sigma160mm';
grid_res{grid_cur}.solvers.x_prob = 'models/solver_x.prototxt';
grid_res{grid_cur}.solvers.y_prob = 'models/solver_y.prototxt';
grid_res{grid_cur}.solvers.z_prob = 'models/solver_z.prototxt';

grid_cur = grid_cur + 1;
grid_res{grid_cur}.name = 'sigma_640mm';
grid_res{grid_cur}.dir  = 'train_val_grid040mm_sigma640mm';
grid_res{grid_cur}.solvers.x_prob = 'models/solver_x.prototxt';
grid_res{grid_cur}.solvers.y_prob = 'models/solver_y.prototxt';
grid_res{grid_cur}.solvers.z_prob = 'models/solver_z.prototxt';

% --- Classifiers
classifiers = {'x_prob','y_prob'};

%% Execution:
results_run_dir = [ results_root '/' datestr(now,'yyyy_mm_dd_HH_MM_SS') ];

for grid_i=1:numel(grid_res)
    grid_res{grid_i}.dirlist = dir( [dataset_root '/' grid_res{grid_i}.dir '/' crossval_prefix '*'] );
    grid_res{grid_i}.crossval_num =  numel( grid_res{grid_i}.dirlist ); %How many crossvalidataion are run for an object
    
    fprintf('--- Training for object %s. Crossvalidations found = %d ...\n', grid_res{grid_i}.dir, grid_res{grid_i}.crossval_num );
    
    for cross_i=1:grid_res{grid_i}.crossval_num
        fprintf('--- Obj = %s : Crossvalidation training = %d \n', grid_res{grid_i}.name, cross_i);
        grid_res{grid_i}.data_files.gt_meta = [dataset_root '/' grid_res{grid_i}.dir '/meta_data.mat'];
        grid_res{grid_i}.cross{cross_i}.dirs.data    = [ ...
            dataset_root '/' ...
            grid_res{grid_i}.dir ...
            sprintf('/%s%02d', crossval_prefix, cross_i') ];
        grid_res{grid_i}.cross{cross_i}.dirs.results = ...
            [results_run_dir sprintf('/%s/%s%02d', grid_res{grid_i}.name, crossval_prefix, cross_i) ];
        grid_res{grid_i}.cross{cross_i}.dirs.model   = grid_res{grid_i}.cross{cross_i}.dirs.results;
        
        grid_res{grid_i}.cross{cross_i}.results = uwbs_caffe_train_test_classif( ...
            grid_res{grid_i}.solvers, ...
            grid_res{grid_i}.cross{cross_i}.dirs, ...
            grid_res{grid_i}.data_files, ...
            classifiers );
    end
    
    % Averaging results for objects
    fprintf('\n------------------------------------------------\n');
    fprintf('Results: \n');
    for class_i=1:numel(classifiers)
        grid_res{grid_i}.results.(classifiers{class_i}).accuracy = 0;
        grid_res{grid_i}.results.(classifiers{class_i}).error = [];
        grid_res{grid_i}.results.(classifiers{class_i}).error_vec = [];
        grid_res{grid_i}.results.(classifiers{class_i}).gt = [];
        grid_res{grid_i}.results.(classifiers{class_i}).label_pred = [];
        grid_res{grid_i}.results.(classifiers{class_i}).pred = [];
        
        for cross_i=1:grid_res{grid_i}.crossval_num
            % Accuracy
            grid_res{grid_i}.results.(classifiers{class_i}).accuracy = ...
                grid_res{grid_i}.results.(classifiers{class_i}).accuracy + ...
                grid_res{grid_i}.cross{cross_i}.results.(classifiers{class_i}).accuracy;
            
            % Errors
            grid_res{grid_i}.results.(classifiers{class_i}).error_vec = ...
                [grid_res{grid_i}.results.(classifiers{class_i}).error_vec ; ...
                grid_res{grid_i}.cross{cross_i}.results.(classifiers{class_i}).error_vec];
            
            % Abs Errors
            grid_res{grid_i}.results.(classifiers{class_i}).error = ...
                [grid_res{grid_i}.results.(classifiers{class_i}).error ; ...
                grid_res{grid_i}.cross{cross_i}.results.(classifiers{class_i}).error];
            
            % Ground truth
            grid_res{grid_i}.results.(classifiers{class_i}).gt = ...
                [grid_res{grid_i}.results.(classifiers{class_i}).gt ; ...
                grid_res{grid_i}.cross{cross_i}.results.(classifiers{class_i}).gt];
            
            % Predictions
            grid_res{grid_i}.results.(classifiers{class_i}).label_pred = ...
                [grid_res{grid_i}.results.(classifiers{class_i}).label_pred ; ...
                grid_res{grid_i}.cross{cross_i}.results.(classifiers{class_i}).label_pred];
            
            % Prediction probabilities
            grid_res{grid_i}.results.(classifiers{class_i}).pred = ...
                [grid_res{grid_i}.results.(classifiers{class_i}).pred ; ...
                grid_res{grid_i}.cross{cross_i}.results.(classifiers{class_i}).pred];
        end
        
        grid_res{grid_i}.results.(classifiers{class_i}).accuracy   = grid_res{grid_i}.results.(classifiers{class_i}).accuracy / grid_res{grid_i}.crossval_num;
        grid_res{grid_i}.results.(classifiers{class_i}).error_mean = mean(grid_res{grid_i}.results.(classifiers{class_i}).error);
        grid_res{grid_i}.results.(classifiers{class_i}).error_dev  = std(grid_res{grid_i}.results.(classifiers{class_i}).error);
        grid_res{grid_i}.results.(classifiers{class_i}).error_vec_mean = mean(grid_res{grid_i}.results.(classifiers{class_i}).error_vec, 1);
        grid_res{grid_i}.results.(classifiers{class_i}).error_vec_dev  = std(grid_res{grid_i}.results.(classifiers{class_i}).error_vec, 0, 1);
        
        
        fprintf('Obj= %s Class= %s :: Accuracy= %5.2f Err(mean/dev)= %5.5f / %5.5f \n', ...
            grid_res{grid_i}.name, ...
            classifiers{class_i}, ...
            grid_res{grid_i}.results.(classifiers{class_i}).accuracy, ...
            grid_res{grid_i}.results.(classifiers{class_i}).error_mean, ...
            grid_res{grid_i}.results.(classifiers{class_i}).error_dev );
    end
    
    if isfield( grid_res{grid_i}.cross{1}.results, 'xyz_sep' )
        % Special calculations
        grid_res{grid_i}.results.xyz_sep.error = [];
        for cross_i=1:grid_res{grid_i}.crossval_num
            % Errors
            grid_res{grid_i}.results.xyz_sep.error = ...
                [grid_res{grid_i}.results.xyz_sep.error ; ...
                grid_res{grid_i}.cross{cross_i}.results.xyz_sep.error];
        end
        grid_res{grid_i}.results.xyz_sep.error_mean = mean(grid_res{grid_i}.results.xyz_sep.error);
        grid_res{grid_i}.results.xyz_sep.error_dev  = std(grid_res{grid_i}.results.xyz_sep.error);
        
        fprintf('Obj= %s Class= xyz_sep :: Err(mean/dev)= %5.5f / %5.5f \n', ...
            grid_res{grid_i}.name, ...
            grid_res{grid_i}.results.xyz_sep.error_mean, ...
            grid_res{grid_i}.results.xyz_sep.error_dev );
    end
    
    if isfield( grid_res{grid_i}.cross{1}.results, 'xy_sep' )
        % Special calculations
        grid_res{grid_i}.results.xy_sep.error = [];
        for cross_i=1:grid_res{grid_i}.crossval_num
            % Errors
            grid_res{grid_i}.results.xy_sep.error = ...
                [grid_res{grid_i}.results.xy_sep.error ; ...
                grid_res{grid_i}.cross{cross_i}.results.xy_sep.error];
        end
        grid_res{grid_i}.results.xy_sep.error_mean = mean(grid_res{grid_i}.results.xy_sep.error);
        grid_res{grid_i}.results.xy_sep.error_dev  = std(grid_res{grid_i}.results.xy_sep.error);
        
        fprintf('Obj= %s Class= xy_sep :: Err(mean/dev)= %5.5f / %5.5f \n', ...
            grid_res{grid_i}.name, ...
            grid_res{grid_i}.results.xy_sep.error_mean, ...
            grid_res{grid_i}.results.xy_sep.error_dev );
    end
    
    fprintf('------------------------------------------------\n');
    
end

%% Printing and plotting
res_str_i = 0;
for grid_i=1:numel(grid_res)
    res_str_i = res_str_i + 1;
    res_str{res_str_i} = sprintf('\n------------------------------------------------\n');
    res_str_i = res_str_i + 1;
    res_str{res_str_i} = sprintf('Results: Object = %s : \n', grid_res{grid_i}.name );
    for class_i=1:numel(classifiers)
        res_str_i = res_str_i + 1;
        res_str{res_str_i} = sprintf('Class= %s :: Accuracy= %5.2f Err(mean/dev)= %5.5f / %5.5f \n', ...
            classifiers{class_i}, ...
            grid_res{grid_i}.results.(classifiers{class_i}).accuracy, ...
            grid_res{grid_i}.results.(classifiers{class_i}).error_mean, ...
            grid_res{grid_i}.results.(classifiers{class_i}).error_dev );
    end
    
    if isfield( grid_res{grid_i}.cross{1}.results, 'xyz_sep' )
        res_str_i = res_str_i + 1;
        res_str{res_str_i} = sprintf('Class= xyz_sep :: Err(mean/dev)= %5.5f / %5.5f \n', ...
            grid_res{grid_i}.results.xyz_sep.error_mean, ...
            grid_res{grid_i}.results.xyz_sep.error_dev );
    end
    
    if isfield( grid_res{grid_i}.cross{1}.results, 'xy_sep' )
        res_str_i = res_str_i + 1;
        res_str{res_str_i} = sprintf('Class= xy_sep :: Err(mean/dev)= %5.5f / %5.5f \n', ...
            grid_res{grid_i}.results.xy_sep.error_mean, ...
            grid_res{grid_i}.results.xy_sep.error_dev );
    end
    
    res_str_i = res_str_i + 1;
    res_str{res_str_i} = sprintf('------------------------------------------------\n');
end

% Printing
res_str_all = '';
for str_i=1:numel(res_str)
    res_str_all = [res_str_all res_str{str_i}];
    fprintf('%s', res_str{str_i});
end

%% Saving results
save( [results_run_dir '/results.mat' ], 'grid_res', 'classifiers' );
% Saving a txt file
txt_file_id = fopen([results_run_dir '/results.txt' ],'w');
fprintf(txt_file_id, '%s', res_str_all);
fclose(txt_file_id);



