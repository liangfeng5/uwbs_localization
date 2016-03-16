function [ results ] = uwbs_caffe_test_classif( net_model, net_weigths, mode, files, varargin )
%% Description:
% The function tests a snapshot for classification
% [ results ] = poke_caffe_test_classif( net_model, net_weigths, mode, files, [plot_mode] )
% --- INPUT:
% net_model = prototxt file with model description
% net_weigths = snapshot file
% mode = which coordinate it evaluates:
%   'x' / 'y' / 'z' = individual coordinates (1d grid)
% files = struct containing names of files for loading
%   .h5_val = h5 file with data for validation
%   .all_val_data = mat file with the rest of data
%   .gt_meta = special file with meta data for particular labels, in
%   particular maps label indices to particular values of the dimension
%   predicted, for example, if we predict 'x' discretized into 5 indices,
%   it gives us what value should x get for every every index value. Need
%   this file to calculate errors given prediction accuracies
%
% --- OPTIONAL:
% [plot_mode] = options for plotting:
%   0 = don't plot 
%   1 = plot everything at once
%   2 = plot sequentially
%
% --- OUTPUT:
% results = structure containing results of testing
% 
%% Parameters
phase = 'test'; % run with phase test (so that dropout isn't applied)

%% Arguments
varar_id = 1;
plot_mode = 0;
if length(varargin) > (varar_id - 1)
   plot_mode = varargin{varar_id}; 
end

%% Initialize results
fprintf('%s : Deploy model = %s Best_snapshot = %s \n', ...
    mfilename, ...
    net_model, ...
    net_weigths);

% --- Loading data
% HDF5
fprintf('%s: Loading file for testing = %s \n', mfilename, files.h5_val );
h5_data_val  = h5read(files.h5_val, '/data');
h5_label_val = h5read(files.h5_val, '/label');

% Mat file with all data
load(files.all_val_data);

% Load
load(files.gt_meta); %It will provide label values and other

% gt = ground truth:
% - get rid of singletons (since gt is represented as 1x1xhxw, where
%   h = variables (electrode val, etc.), w = time axes
% - rearrange axes: 1st dim = sample index, 2nd dim = x,y,z ... i.e. values
%   of outputs

results.gt = squeeze(h5_label_val);

if min( results.gt ) ~= 1
   results.gt = permute(results.gt,[2 1]);
   fprintf('h5_label_val size =');
%     display(size(results.gt));
   [results.gt_prob results.gt_label] = max( results.gt, [], 2 );
    %VERY IMPORTANT, because caffe converts label indx starting with 0
    results.gt_label_shift = int32(results.gt_label); 
    results.label_pred = results.gt_label_shift - 1; 
else
    results.gt_label = results.gt;    
end

%preallocate memory for predictions
results.samples_num    = size(results.gt, 1);
results.samples_num_h5 = size(h5_data_val, 4);
if results.samples_num ~= results.samples_num_h5
   msgID = sprintf('%s:SampNumDontMatch', mfilename);
   msg = sprintf('ERROR: %s : number of samples(%d) in h5 file (%s) does not correspond to number of samples (%d) in mat file (%s) \n', mfilename, results.samples_num_h5, files.h5_val, results.samples_num, files.all_val_data ); 
   display(msg);
   baseException = MException(msgID,msg); 
   throw(baseException);
end

results.ff_time = zeros([results.samples_num, 1]);


if strcmp('x', mode)
    classes_num = length(gt_meta.lbl_val{1} ); %Only for classification
elseif strcmp('y', mode)
    classes_num = length(gt_meta.lbl_val{2} ); %Only for classification
elseif strcmp('z', mode)
    classes_num = length(gt_meta.lbl_val{3} ); %Only for classification
end

results.pred = zeros([ results.samples_num, classes_num]);


% Initialize a network
% IMPORTANT:
% If matlab crashes, there might be a problem with reading net description prototxt file
% In order to see the problem I recommend executing matlab from command
% line, since all matcaffe problems are reported only there.
% Some examples of problems:
% - the name of the network is set up twice 
% - loss layer is kept for the deployment network, although data layer with
%   labels has been removed
caffe.set_mode_gpu();
caffe.set_device(0);

net = caffe.Net(net_model, net_weigths, phase);

for samp_i=1:results.samples_num
    tic
    out = net.forward( {h5_data_val(:,:,:,samp_i)} );
%     size(results.pred(samp_i, :))
%     size(out{1})
    results.ff_time(samp_i) = toc; 
%     fprintf('Classes num = %d out_dim = %d \n', classes_num, size(out{1}(:)',2))
    results.pred(samp_i, :) = out{1}(:)';
    
end


%% Postprocess the results
% accuracy
[results.label_pred_prob results.label_pred] = max( results.pred, [], 2 );
%VERY IMPORTANT, because caffe converts label indx starting with 0
results.label_pred = int32(results.label_pred); 
results.label_pred_shift = results.label_pred;
results.label_pred = results.label_pred - 1;
%Shift back because matlab indexing is from 1 (not 0 as in C++)
%Need this variable to access values of labels in arrays

results.pred_miscl =  abs( int32(results.gt_label) - results.label_pred );
display(results.pred_miscl)
results.accuracy  = 1 - length( find( results.pred_miscl ) ) / length( results.pred_miscl );

% Getting values for labels associated with samples
distance_metric = 0;
if strcmp('x', mode)
    results.gt_value = zeros(results.samples_num, 1);
    results.label_pred_value = gt_meta.lbl_val{1}( results.label_pred_shift );
    results.label_pred_value = results.label_pred_value(:);
    for samp_i=1:results.samples_num 
        results.gt_value(samp_i) = gt_cur.xyz(samp_i, 1);
    end
elseif strcmp('y', mode)
    results.gt_value = zeros(results.samples_num, 1);
    results.label_pred_value = gt_meta.lbl_val{2}( results.label_pred_shift );
    results.label_pred_value = results.label_pred_value(:);
    for samp_i=1:results.samples_num 
        results.gt_value(samp_i) = gt_cur.xyz(samp_i, 2);
    end
elseif strcmp('z', mode)
    results.gt_value = zeros(results.samples_num, 1);
    results.label_pred_value = gt_meta.lbl_val{3}( results.label_pred_shift );
    results.label_pred_value = results.label_pred_value(:);
    for samp_i=1:results.samples_num 
        results.gt_value(samp_i) = gt_cur.xyz(samp_i, 3);
    end
end

results.error_vec = results.gt_value - results.label_pred_value;

switch distance_metric
    case 1 %Euclidean metric
       results.error = sqrt( sum( abs(results.error_vec).^2, 2  ) );  
    otherwise
       results.error = abs(results.error_vec);
end

results.error_mean = mean( results.error );
results.error_dev  = std( results.error );

%% Plotting/Printing results (if needed)

fprintf('-----------------');
fprintf('%s: Class = %s Accuracy = %f Mean_(abs)error = %f Total_pts = %d Miscl_points = %d \n', ...
    mfilename, ...
    mode, ...
    results.accuracy, ...
    results.error_mean, ...
    results.samples_num, ...
    length(find( results.pred_miscl )) ...
    );

%% Close everything
caffe.reset_all();

end

