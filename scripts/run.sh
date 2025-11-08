#!/bin/bash

# Activate conda environment (if using conda)
# conda activate transformer-env

# Set PYTHONPATH to include the project root directory
export PYTHONPATH="$PYTHONPATH:$(dirname "$(dirname "$0")")"

# Run the training script
python ../src/train.py