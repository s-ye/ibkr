#!/bin/bash

# Define the output directory
OUTPUT_DIR="output"

# Check if the output directory exists
if [ -d "$OUTPUT_DIR" ]; then
  echo "Clearing contents of $OUTPUT_DIR..."
  rm -rf "$OUTPUT_DIR"/*
else
  echo "Creating output directory: $OUTPUT_DIR"
  mkdir "$OUTPUT_DIR"
fi

# Run main.py
echo "Running real_time_data.py..."
python3 real_time_data.py