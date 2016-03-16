function print_proctime( name_in, time_total )
    fprintf('%s : Total processing time = %d (min) %d (sec) \n', name_in, floor(time_total / 60), int32(rem(time_total, 60)) );
end

