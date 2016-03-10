%% Parameters
input = 'data_dir';
ext = 'csv';
train_prop = [0.8, 0.08, 0.12]; %Training/Validation/Testing
gt.step = 0.05 * [1 1 1]; %XYZ
feature_names = {'dist0','dist1','dist2','dist3','dist4','dist5'};
gt.names  = {'x','y'};
gt.limits = [-2 2; -1 1];
precision = 'single';
prob_variance = gt.step;
fig_offset = 0;

%Plotting options
plot_traj = 0;
plot_prob = 1;

%% Execution
tic

%Getting filelist
file_list = get_file_list(input, ext);

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
train_prop = train_prop ./ sum(train_prop);

%Get feature names and corresponding indices
feat_indx = featnames2indx(val_names, feature_names);
gt.indx   = featnames2indx(val_names, gt.names);

%Get relevant
feat_data = data(:, feat_indx);
gt.data   = data(:, gt.indx);

%Disretize ground truth
for dim_i=1:gt.dim_num
    gt.grid{dim_i} = gt.limits(dim_i,1):gt.step(dim_i):gt.limits(dim_i,2);
end

%Get values for labels and assign labels for ground truth
gt.labels = zeros( size(gt.data), precision );
for dim_i=1:gt.dim_num
    gt.lbl_val{dim_i} = zeros(1, numel(gt.grid{dim_i}) - 1 , precision);
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
    gt.prob{dim_i} = zeros( samp_num, numel(gt.lbl_val{dim_i}) );
    
    for samp_i=1:samp_num
        gt.prob{dim_i}(samp_i, :) = ...
            gaussmf(gt.lbl_val{dim_i}, [prob_variance(dim_i) gt.data(samp_i, dim_i) ]);
        %Renormalize
        gt.prob{dim_i}(samp_i, :) = gt.prob{dim_i}(samp_i, :) / sum(gt.prob{dim_i}(samp_i, :));
    end
end

%Convert data to proper precison for training
data = single(data);

%% Plotting
cur_fig = fig_offset;

cur_fig = cur_fig + 1;
figure(cur_fig);
if plot_traj
    plot( gt.data(:,1), gt.data(:,2) );
end

if plot_prob    
    cur_fig = uwbs_plot_prob(gt.prob, gt.data, cur_fig);
end

time_total = toc;
fprintf('%s : Total processing time = %d (min) %d (sec) \n', mfilename, floor(time_total / 60), rem(time_total, 60) );