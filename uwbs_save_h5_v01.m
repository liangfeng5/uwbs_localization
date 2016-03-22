function [ samp_num ] = uwbs_save_h5_v01( params, samples, h5_filename )
%% Description:
% given an array of data cuts save them into an h5 file for caffe
% --- Arguments:
% data_cuts - array with data cuts
% h5_filename - base (including path) for the filename of the file

%% Parameters:

%% Computed parameters
samp_num = numel(samples);


fprintf( '%s: samples = %d \n', mfilename, samp_num);

%% Executions
%H x W x C x N
feat_num = numel(params.feat_names);
feat_4d = zeros([feat_num, params.feat_timelen, 1, samp_num], params.precision);

% Label probabilities preinitialization
label = zeros(samp_num, numel(params.label_names), 'int32');
for dim_i=1:numel(params.label_names)
    label_prob.(params.gt_prob_names{dim_i}) = zeros( [1, 1, numel(params.label_val{dim_i}), samp_num], params.precision);
end

for samp_i=1:samp_num
    %Pack features
    for feat_i=1:feat_num
        feat_4d( feat_i, :, 1, samp_i) = ...
            samples{samp_i}.data_ts(params.feat_name_indx(feat_i), :);
    end
    
    %Pack label probabilities and labels
    for dim_i=1:numel(params.label_names)
       label_prob.(params.gt_prob_names{dim_i})(1, 1, :, samp_i) = ...
           (samples{samp_i}.gt.( params.gt_prob_names{dim_i} ))';
       label(samp_i, dim_i) = samples{samp_i}.label(dim_i);
    end
    
end

for dim_i=1:numel(params.label_names)
    hdf5write( [h5_filename, '_', params.label_names{dim_i} ,'.h5'], '/data', feat_4d, '/label', label(:, dim_i)' );
    hdf5write( [h5_filename, '_', params.gt_prob_names{dim_i} ,'.h5'], '/data', feat_4d, '/label', label_prob.(params.gt_prob_names{dim_i}) );
end

%Saving all ground truth data in a mat file 
save(h5_filename, 'samples');

%Creating a soft link to one of the files s.t. everything would work by
%default
% [h5_p h5_n h5_e] = fileparts(h5_filename);
% system(['rm -f ' h5_filename '.h5']);
% system(['ln -s ' './' h5_n '_yaw.h5 ' h5_filename '.h5']);

% el_size = fliplr(size(electrodes_train)); %reversing order for caffe
% electrodes_permuted = zeros( [ el_size(1:(end-1)), 1, el_size(end) ], 'single'  ); %Insert a singleton dimension since you must have n x c x h x w in caffe, and c = 1
% electrodes_permuted(:,:,1,:) = single( permute(electrodes_train, [3,2,1]) ); %copying with converting to single params.precision
% hdf5write([out_dir, '/', out_name_base,'_train.h5'], '/data', electrodes_permuted, '/label', single(labels_train) );
end

