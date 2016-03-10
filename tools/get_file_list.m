function [ filelist ] = get_file_list( input, ext_in )
%% Description:
% the function consumes either a directory name or 
% a *.txt file with a list of files to load and produces a filelist
%% --- INPUTS:
% input = either a *.txt file with a list of files or a directory to load
%       files from
% ext_in = extension of files to load if necessary, like 'mat', 'clmc' (can not be 'txt') 

    ext_filter = ['*.' ext_in];
    filelist = {};
    
    %In case we entered just a filename 
    if strcmp(['.' ext_in], input((end-length(ext_in)):end))
        filelist{1} = input;
    else
        read_list = strcmp('.txt', input(end-3:end));
        % 0 = read from a directory mask provided
        % 1 = read from a filelist

        %% Execution
        if read_list
            fprintf('%s : filelist provided \n', mfilename);
            %The old method - uses predefined filelist
            filelist = importdata(input);
        else
            %Loading filenames according to the directory filter
            filelist_str = dir([input '/' ext_filter]);
            filelist{length(filelist_str)} = '';
            for file_i=1:length(filelist_str)
               filelist{file_i} = filelist_str(file_i).name; 
            end
        end
    end
end

