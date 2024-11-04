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

# clear the cache as well
OUTPUT_DIR="cache"

# Check if the output directory exists
if [ -d "$OUTPUT_DIR" ]; then
  echo "Clearing contents of $OUTPUT_DIR..."
  rm -rf "$OUTPUT_DIR"/*
else
  echo "Creating output directory: $OUTPUT_DIR"
  mkdir "$OUTPUT_DIR"
fi


# Run main.py
echo "Running mass_testing.py..."
python3 mass_testing.py