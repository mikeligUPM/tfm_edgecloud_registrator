#!/bin/bash

# Must be executed in data/3DMatch folder
# Iterates over all subfolders in the current directory and renames the files in the following way:
#   Original: 'frame-000000.color.png', 'frame-000000.depth.png' and 'frame-000000.pose.txt' 
#   Renamed: 'frame-000000_color.png', 'frame-000000_depth.png' and 'frame-000000_pose.txt'
# Reason of implementation: Otherwise python can't read the files

# Specify the parent folder as the current directory
parent_folder="."

# Iterate through each subfolder
for folder in "$parent_folder"/*/; do
    echo "Processing folder: $folder"

    # Change directory to the current subfolder
    cd "$folder" || continue

    # Rename .color.png files
    rename 's/\.color\.png/_color\.png/' frame-*.color.png

    # Rename .depth.png files
    rename 's/\.depth\.png/_depth\.png/' frame-*.depth.png

    # Rename .pose.txt files
    rename 's/\.pose\.txt/_pose.txt/' frame-*.pose.txt

    # Change directory back to parent
    cd - >/dev/null || exit
done
