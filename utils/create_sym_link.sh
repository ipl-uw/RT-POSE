#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <src_root_dir> <dst_root_dir>"
    exit 1
fi

# Assign arguments to variables
src_root_dir="$1"
dst_root_dir="$2"

# Check if the source directory exists
if [ ! -d "$src_root_dir" ]; then
    echo "Source directory '$src_root_dir' does not exist."
    exit 1
fi

# Check if the destination directory exists, create if not
if [ ! -d "$dst_root_dir" ]; then
    echo "Destination directory '$dst_root_dir' does not exist. Creating it..."
    mkdir -p "$dst_root_dir"
fi

# Loop through each directory in the source root directory
for dir in "$src_root_dir"/*; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        ln -s "$dir" "$dst_root_dir/$dir_name"
        echo "Symlinked '$dir' to '$dst_root_dir/$dir_name'"
    fi
done

echo "Symlinking complete."
