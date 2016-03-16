#!/bin/bash
#EXAMPLE: matlab_cmd.sh "functionmat(var1,var2)"
#printf '%s' "$1"
matlab -nosplash -nodesktop -logfile ${2}.log -r "$1"
