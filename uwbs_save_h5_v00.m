function [ samples_num ] = uwbs_save_h5_v00( gt, indx, h5_filename, precision )
%% Description:
% given an array of data cuts save them into an h5 file for caffe
% --- Arguments:
% data_cuts - array with data cuts
% h5_filename - base (including path) for the filename of the file

%% Parameters:

%% Computed parameters
samples_num = numel(indx);
feat_size = [1, size(gt.feat, 2)];

fprintf( '%s: samples = %d \n', mfilename, samples_num);

%% Executions
feat_4d = zeros([feat_size(2), feat_size(1), 1, samples_num], precision);
%Save all: feat, data, labels, prob
gt_cur.feat    = gt.feat(indx, :);
gt_cur.xyz     = gt.data(indx, :);
gt_cur.labels  = gt.labels(indx, :);
gt_cur.prob    = gt.prob(indx, :);

for dim_i=1:gt.dim_num
    label_prob.(gt.names{dim_i}) = zeros([1, 1, numel(gt.lbl_val{dim_i}), samples_num], precision);
end

for samp_i=1:numel(indx)
    for feat_i=1:feat_size(2)
        feat_4d( feat_i, 1, 1, samp_i) = gt.feat(indx(samp_i), feat_i);
    end
    
    for dim_i=1:gt.dim_num
       label_prob.(gt.names{dim_i})(1,1,:, samp_i) = gt.prob{indx(samp_i), dim_i}'; 
    end   
end

for dim_i=1:gt.dim_num
    labels = int32(gt.labels(indx, dim_i)');
    hdf5write( [h5_filename, '_', gt.names{dim_i} ,'.h5'], '/data', feat_4d, '/label', labels);
    hdf5write( [h5_filename, '_', gt.names{dim_i} ,'_prob.h5'], '/data', feat_4d, '/label', label_prob.(gt.names{dim_i}) );
end

%Saving all ground truth data in a mat file 
save(h5_filename, 'gt_cur');

%Creating a soft link to one of the files s.t. everything would work by
%default
% [h5_p h5_n h5_e] = fileparts(h5_filename);
% system(['rm -f ' h5_filename '.h5']);
% system(['ln -s ' './' h5_n '_yaw.h5 ' h5_filename '.h5']);

% el_size = fliplr(size(electrodes_train)); %reversing order for caffe
% electrodes_permuted = zeros( [ el_size(1:(end-1)), 1, el_size(end) ], 'single'  ); %Insert a singleton dimension since you must have n x c x h x w in caffe, and c = 1
% electrodes_permuted(:,:,1,:) = single( permute(electrodes_train, [3,2,1]) ); %copying with converting to single precision
% hdf5write([out_dir, '/', out_name_base,'_train.h5'], '/data', electrodes_permuted, '/label', single(labels_train) );
end

