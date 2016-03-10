function [ cur_fig ] = uwbs_plot_prob( prob, data, varargin )
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
samp_num = size(data,1);
cur_fig = fig_offset;
dim_num = numel(prob);

    for samp_i=1:samp_step:samp_num              
        for dim_i=1:dim_num
            figure(cur_fig + dim_i);
            clf(cur_fig + dim_i);
            plot( gt.prob{samp_i, dim_i} );
            title(sprintf('Probabilities for sample = %d. Mean = %f', ...
                samp_i, ...
                data(samp_i, dim_i)) );
        end
        pause;
    end
    cur_fig = cur_fig + gt.dim_num;

end

