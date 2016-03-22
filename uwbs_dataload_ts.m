%% Parameters
ext = 'csv';

params.train_val_test_prop = [0.8, 0.08, 0.12]; %Training/Validation/Testing
params.gt_grid_step = 0.04 * [1 1 1]; %XYZ

params.feat_names = {'dist0','dist1','dist2','dist3','dist4','dist5'};
params.gt_prob_names  = {'x_prob','y_prob'};
params.gt_regr_names  = {'x_prob','y_prob'};
params.label_names = {'x','y'};

params.gt_limits = [-2 2; -1 1];
params.precision = 'single';
params.feat_timelen = 20; %number of time steps used as feature vector
% params.gt_prob_variance = 0.5 * params.gt_grid_step;
% params.gt_prob_variance = params.gt_grid_step;
params.gt_prob_variance =  2 * params.gt_grid_step;
% params.gt_prob_variance = 16 * params.gt_grid_step;
params.subsamp_num = 2000;
% params.gt_prob_variance = 64 * params.gt_grid_step; 
fig_offset = 0;

%output filenames MUST NOT contain extension. Only the path and the name
%itself
input_filenames = 'data_dir/';
input_directory = fileparts(input_filenames);
% filenames.output_dir = [input_directory '/train_val_grid040mm_sigma640mm'];
filenames.output_dir = [input_directory '/train_val'];
filenames.crossval_prefix = 'cross';
filenames.h5_train = 'data_train';
filenames.h5_val   = 'data_val';
filenames.h5_test  = 'data_test';
filenames.data_all = 'data_all';
filenames.label_values = [filenames.output_dir '/label_values'];


%Plotting options
plot_traj = 0;
plot_prob = 0;

%% Execution: preprocessing
tic
% --- Some sanity checking and preprocessing
%Renormalize train/val/test proportions
params.train_val_test_prop = params.train_val_test_prop ./ sum(params.train_val_test_prop);


%% Execution: data loading
%Getting filelist
file_list = get_file_list(input_filenames, ext);

%Read the data
file_i = 1;
[val_names , datasrc] = csvread_names(file_list{file_i});
datasrc = datasrc'; %s.t. time would be horizontal
params.label_dim_num = numel(params.label_names);

%Some minor corrections
val_names{1} = strrep(val_names{1}, '#', '');

%Separating units
[val_names, val_units] = extract_units(val_names);

%% Execution: getting relevant data points
%Get feature names and corresponding indices
[params.feat_name_indx params.dataname2indx] = featnames2indx(val_names, params.feat_names);

%Sample subset of points (which will correspond to our data samples)
subsamp_start_timeindx = params.feat_timelen + 1;
params.datasrc_sampnum = size(datasrc, 2);
timeindx_population = subsamp_start_timeindx:params.datasrc_sampnum;
params.feat_timeindx = sort(randsample( timeindx_population, params.subsamp_num), 'ascend');

%Get relevant data (extract samples from time series)
samp_num = numel( params.feat_timeindx );
samples{samp_num} = 0;
data_temp.x = zeros(1, samp_num);
data_temp.y = zeros(1, samp_num);
data_temp.z = zeros(1, samp_num);

for samp_i=1:samp_num
    samp_timeinterval = ...
        (params.feat_timeindx(samp_i) - params.feat_timelen + 1):params.feat_timeindx(samp_i);
    samples{samp_i}.data_ts = datasrc( :, samp_timeinterval );
    samples{samp_i}.data.x = samples{samp_i}.data_ts( params.dataname2indx.x, end );    
    samples{samp_i}.data.y = samples{samp_i}.data_ts( params.dataname2indx.y, end );
    samples{samp_i}.data.z = samples{samp_i}.data_ts( params.dataname2indx.z, end );
    
    data_temp.x(samp_i) = samples{samp_i}.data.x;
    data_temp.y(samp_i) = samples{samp_i}.data.y;
    data_temp.z(samp_i) = samples{samp_i}.data.z;
end

%% Execution: GT extraction
%Disretize ground truth
for dim_i=1:params.label_dim_num
    params.gt_grid{dim_i} = params.gt_limits(dim_i,1):params.gt_grid_step(dim_i):params.gt_limits(dim_i,2);
end


%Get values for labels and assign labels for ground truth
params.label = zeros( samp_num, params.label_dim_num );
for dim_i=1:params.label_dim_num
    
    params.label_val{dim_i} = zeros(1, numel(params.gt_grid{dim_i}) - 1);
    for bin_i=1:(numel(params.gt_grid{dim_i})-1)
        params.label_val{dim_i}(bin_i) = (params.gt_grid{dim_i}(bin_i) + params.gt_grid{dim_i}(bin_i+1))/2;
    end
    
    %Assign labels
    [params.label_count{dim_i}, params.label(:, dim_i)] = ...
        histc( data_temp.(params.label_names{dim_i}), params.gt_grid{dim_i} );
    params.label(:, dim_i) = params.label(:, dim_i) - 1; %For caffe
    
    fprintf('%s : Labels in for %s = %d \n', mfilename, params.label_names{dim_i}, numel(params.label_val{dim_i}) );
end

%Copy labels for every sample
for samp_i=1:samp_num 
   samples{samp_i}.label = params.label(samp_i, :);
end

%Get probabilities for ground truth from labels
for dim_i=1:params.label_dim_num    
    for samp_i=1:samp_num
        prob_distrib = ...
            gaussmf(params.label_val{dim_i}, [params.gt_prob_variance(dim_i) samples{samp_i}.data.(params.label_names{dim_i}) ]);
        
        %Renormalize
        prob_distrib = prob_distrib - min(prob_distrib);
        samples{samp_i}.gt.(params.gt_prob_names{dim_i}) = ...
            prob_distrib / sum( prob_distrib ) ;
    end
end

%Extracting indices for train/val/test
params.crossval_indx = uwbs_save_trainvaltest2(params, samples, filenames );


%% Plotting
cur_fig = fig_offset;
samples_train = samples(params.crossval_indx{1}.train);
samples_val = samples(params.crossval_indx{1}.val);
samples_test = samples(params.crossval_indx{1}.test);

if plot_traj
    cur_fig = uwbs_plot_xyz( samples_train, cur_fig);
    title(sprintf('Training trajectory. Samples_num = %d', numel(samples_train) ));
    cur_fig = uwbs_plot_xyz( samples_val, cur_fig);
    title(sprintf('Val trajectory. Samples_num = %d', numel(samples_val) ));
    cur_fig = uwbs_plot_xyz( samples_test, cur_fig);
    title(sprintf('Test trajectory. Samples_num = %d', numel(samples_test) ));
end

if plot_prob    
    cur_fig = uwbs_plot_prob2(params, samples, 100, cur_fig);
end

%Printing processing time
time_total = toc;
print_proctime(mfilename, time_total);