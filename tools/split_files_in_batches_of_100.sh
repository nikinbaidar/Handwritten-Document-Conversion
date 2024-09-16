#!/bin/bash

# Set the source directory containing the files
source_dir="${1}"

if [ -d "${1}" ]; then
#
    # Get the total number of files
    total_files=$(ls -1 "$source_dir" | wc -l)

    # Calculate the number of directories needed
    num_dirs=$(( (total_files + 99) / 100 ))
    for i in $(seq 1 $num_dirs); do
      mkdir -p "dir$i"
    done

    # Get the list of files
    files=($(ls -1 "$source_dir"))

    # Move files to the corresponding directories
    for i in "${!files[@]}"; do
      dir_num=$((i / 100 + 1))
      mv "$source_dir/${files[$i]}" "dir$dir_num/"
    done
else
    echo "${1}" not found or is not a directory
    exit 1;
fi
