#!/bin/bash

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Create necessary directories if they don't exist
mkdir -p data/Takeout
mkdir -p config

# Check for required files
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please run ./setup.sh first"
    exit 1
fi

if [ ! -f "config/params.env" ]; then
    echo "Error: config/params.env not found. Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "api_venv_spacy" ]; then
    source api_venv_spacy/bin/activate
else
    echo "No virtual environment found. Please run ./setup.sh first"
    exit 1
fi

# Run the script with src in PYTHONPATH
PYTHONPATH=$PYTHONPATH:$(pwd)/src python src/finetune.py "$@" 