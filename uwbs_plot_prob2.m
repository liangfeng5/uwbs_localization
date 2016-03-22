function [ cur_fig ] = uwbs_plot_prob2( params, samples, varargin )
%% Desription
% Plot probabilities
% [ cur_fig ] = uwbs_plot_prob( prob, data, [samp_step, fig_offset] )

%% Arguments
var_i = 1;
samp_step = 1;
if length(varargin) >= var_i
    samp_step =  varargin{var_i};
end

var_i = 2;
fig_offset = 0;
if length(varargin) >= var_i
    fig_offset =  varargin{var_i};
end

%% Execution
samp_num = numel(samples);
cur_fig  = fig_offset;
dim_num  = numel(params.label_names);

for samp_i=1:samp_step:samp_num
    for dim_i=1:dim_num
        figure(cur_fig + dim_i);
        clf(cur_fig + dim_i);
%         plot( samples{samp_i}.gt.( params.gt_prob_names{dim_i} ) );
        bar( samples{samp_i}.gt.( params.gt_prob_names{dim_i} ) );
        title(sprintf('Probabilities for sample = %d. Mean = %f', ...
            samp_i, ...
            samples{samp_i}.data.( params.label_names{dim_i} ) ) );
    end
    pause;
end
cur_fig = cur_fig + dim_num;

end

