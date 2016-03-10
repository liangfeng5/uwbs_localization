function [ names, data ] = csvread_names( file )
%% Read csv file where the first line denotes variables used
% [ names, data ] = csvread_names( file )
% ---INPUT:
% file = filename
% --- OUTPUT:
% names = cell array containing variable names
% data = matrix containing data

%% Execution
lines = textread(file, '%s', 'delimiter', '\n');
names = strsplit(lines{1},',');
%Cleaning up spaces at the beginning and end
for name_i=1:numel(names)
    names{name_i} = strtrim(names{name_i});
end

data = zeros(numel(lines)-1, numel(names));

for line_i=2:numel(lines)
    values = strsplit(lines{line_i}, ',' );
    if numel(values) ~= numel(names)
        msgID = sprintf('%s:WrongNumberOfValues',mfilename);
        msg = sprintf('%s : Number of values read dones not match the header. Header = %d Values = %d Line = %d', mfilename, numel(names), numel(values), line_i + 1 );
        baseException = MException(msgID,msg);
        throw(baseException);
    end
    
    for val_i=1:numel(values)
        data(line_i-1, val_i) = str2num( values{val_i} );
    end
end



end

