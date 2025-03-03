#!/bin/bash
# Safer reset script that preserves important directories but cleans their contents

echo "Resetting newspaper image analysis system..."

# Remove the database directory
if [ -d "db" ]; then
    echo "Removing database directory..."
    rm -rf db
fi

# Clean output directory but preserve the base directory
if [ -d "output" ]; then
    echo "Cleaning output directory..."
    find output -mindepth 1 -delete
else
    mkdir -p output
fi

# Clean cross_compare directory
if [ -d "cross_compare" ]; then
    echo "Cleaning cross_compare directory..."
    rm -rf cross_compare
fi

# Clean testout directory
if [ -d "testout" ]; then
    echo "Cleaning testout directory..."
    rm -rf testout
fi

# Remove progress tracking files
echo "Removing progress tracking files..."
rm -f output/*.json

echo "Reset completed successfully!"