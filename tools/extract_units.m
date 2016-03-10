function [ names, units ] = extract_units( namesunits )
%% Description:
% Extracts units from names
% [ names, units ] = extract_units( namesunits )
%% Execution:

names = namesunits;
units = namesunits;
for name_i=1:numel(namesunits)
    unit_start_all = strfind(names{name_i}, '[');
    unit_end_all = strfind(names{name_i}, ']');
    if ~isempty(unit_start_all) && ~isempty(unit_end_all)
        unit_start = unit_start_all(end) + 1;
        unit_end = unit_end_all(end) - 1;
        cur_name = names{name_i};
        if unit_start <= unit_end && (unit_start -2) > 0
            names{name_i} = cur_name(1:(unit_start-2));
            units{name_i} = cur_name(unit_start:unit_end);
        end
    end
end

end

