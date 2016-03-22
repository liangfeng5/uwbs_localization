function [ feat_indx, name_indx_str ] = featnames2indx( names, feature_names )
%% Description:
% Convert cell arraty of feature names, extract indices which they occupy
% in the names array
% [ feat_indx ] = valname2indx( names, feature_names )

%% Execution
feat_indx = zeros(1, numel(feature_names));
for name_i=1:numel(names)
    name_indx_str.(names{name_i}) = name_i;
end

for feat_i=1:numel(feat_indx)
   feat_indx(feat_i) = name_indx_str.(feature_names{feat_i});
end

end

