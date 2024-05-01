#!/bin/bash

# Loop through all .ulg files in the current directory
for file in *.ulg; do
    # Skip if no .ulg files are found
    [ -e "$file" ] || continue

    # Create a directory name by removing the .ulg extension
    dir="${file%.ulg}"

    # Create a directory with that name
    mkdir -p "$dir"

    # Move the .ulg file into the newly created directory
    mv "$file" "$dir/"

    # Go to the directory
    cd "$dir"

    # Call ulog2csv on the .ulg file
    ulog2csv "$file"

    # Go back to the original directory
    cd ..
done

