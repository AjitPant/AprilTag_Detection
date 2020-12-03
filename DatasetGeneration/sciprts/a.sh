#!/bin/bash

# User configuratoin
#===================
files=(*.jpg)           # Set the file pattern to be used, e.g. (*.txt) or (*)
num_files_per_tar=300 # Number of files per tar
num_procs=4         # Number of tar processes to start
tar_file_dir='./' # Tar files dir
tar_file_name_prefix='tar' # prefix for tar file names
tar_file_name="$tar_file_dir/$tar_file_name_prefix"

# Main algorithm
#===============
num_tars=$((${#files[@]}/num_files_per_tar))  # the number of tar files to create
tar_files=()  # will hold the names of files for each tar

tar_start=0 # gets update where each tar starts
# Loop over the files adding their names to be tared
for i in `seq 0 $((num_tars-1))`
do
  tar_files[$i]="$tar_file_name$i.tar.bz2 ${files[@]:tar_start:num_files_per_tar}"
  tar_start=$((tar_start+num_files_per_tar))
done

# Start tar in parallel for each of the strings we just constructed
printf '%s\n' "${tar_files[@]}" | xargs -n$((num_files_per_tar+1)) -P$num_procs tar cjvf
