%% Parameters
ext = 'csv';
gt.train_prop = [0.8, 0.08, 0.12]; %Training/Validation/Testing
gt.step = 0.05 * [1 1 1]; %XYZ
gt.feat_names = {'dist0','dist1','dist2','dist3','dist4','dist5'};
gt.names  = {'x','y'};
gt.limits = [-2 2; -1 1];
precision = 'single';
gt.prob_variance = 2 * gt.step;
fig_offset = 0;

%output filenames MUST NOT contain extension. Only the path and the name
%itself
input_filenames = 'data_dir/';
input_directory = fileparts(input_filenames);
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

%% Execution
tic

%Getting filelist
file_list = get_file_list(input_filenames, ext);

%Read the data
file_i = 1;
[val_names , data] = csvread_names(file_list{file_i});
samp_num = size(data,1);
gt.dim_num = numel(gt.names);

%Some minor corrections
val_names{1} = strrep(val_names{1}, '#', '');

%Separating units
[val_names, val_units] = extract_units(val_names);

%Renormalize train/val/test proportions
gt.train_prop = gt.train_prop ./ sum(gt.train_prop);

%Get feature names and corresponding indices
gt.feat_name_indx = featnames2indx(val_names, gt.feat_names);
gt.name_indx   = featnames2indx(val_names, gt.names);

%Get relevant data
gt.feat = data(:, gt.feat_name_indx);
gt.data   = data(:, gt.name_indx);

%Disretize ground truth
for dim_i=1:gt.dim_num
    gt.grid{dim_i} = gt.limits(dim_i,1):gt.step(dim_i):gt.limits(dim_i,2);
end

%Get values for labels and assign labels for ground truth
gt.labels = zeros( size(gt.data) );
for dim_i=1:gt.dim_num
    gt.lbl_val{dim_i} = zeros(1, numel(gt.grid{dim_i}) - 1);
    for bin_i=1:(numel(gt.grid{dim_i})-1)
        gt.lbl_val{dim_i}(bin_i) = (gt.grid{dim_i}(bin_i) + gt.grid{dim_i}(bin_i+1))/2;
    end
    %Assign labels
    [gt.lbl_count{dim_i}, gt.labels(:, dim_i)] = ...
        histc( gt.data(:, dim_i), gt.grid{dim_i} );
    
    fprintf('%s : Labels in for %s = %d \n', mfilename, gt.names{dim_i}, numel(gt.lbl_val{dim_i}) );
end

%Get probabilities from labels
for dim_i=1:gt.dim_num    
    for samp_i=1:samp_num
%         gt.prob{samp_i, dim_i} = zeros( 1, numel(gt.lbl_val{dim_i}) );
        gt.prob{samp_i, dim_i} = ...
            gaussmf(gt.lbl_val{dim_i}, [gt.prob_variance(dim_i) gt.data(samp_i, dim_i) ]);
        %Renormalize
        gt.prob{samp_i, dim_i} = gt.prob{samp_i, dim_i} / sum( gt.prob{samp_i, dim_i} );
    end
end

%Extracting indices for train/val/test
crossval_indx = uwbs_save_trainvaltest(gt, filenames, precision);


%% Plotting
cur_fig = fig_offset;

if plot_traj
    cur_fig = cur_fig + 1;
    figure(cur_fig);
    plot( gt.data(:,1), gt.data(:,2) );
end

if plot_prob    
    cur_fig = uwbs_plot_prob(gt.prob, gt.data, 100, cur_fig);
end

%Printing processing time
time_total = toc;
print_proctime(mfilename, time_total);