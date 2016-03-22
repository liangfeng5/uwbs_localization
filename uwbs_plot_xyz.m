function [ fig_cur ] = uwbs_plot_xyz( samples, fig_cur)
%% Description:
% Plot trajectory given samples
% [ fig_cur ] = uwbs_plot_xyz( samples, fig_cur)
samp_num = numel(samples);
xyz = zeros(samp_num, 3);

for samp_i=1:samp_num
    xyz(samp_i, :) = [samples{samp_i}.data.x samples{samp_i}.data.y 0];
end

fig_cur = fig_cur + 1;
figure(fig_cur);
plot( xyz(:, 1), xyz(:, 2) );

end

