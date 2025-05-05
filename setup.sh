#!/bin/bash

# Exit on error
set -e

# Check Python version
echo "Checking Python version..."
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo "Error: Python 3.8 or higher is required. Found version $python_version"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/Takeout
mkdir -p config

# Copy example files if they don't exist
if [ ! -f ".env" ]; then
    echo "Creating .env from example..."
    cp env.example .env
    echo "Please edit .env and add your OpenAI API key"
fi

if [ ! -f "config/params.env" ]; then
    echo "Creating config/params.env from example..."
    cp config/params.env.example config/params.env
fi

# Verify setup
echo "Verifying setup..."
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not created"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "Error: .env file not created"
    exit 1
fi

if [ ! -f "config/params.env" ]; then
    echo "Error: config/params.env not created"
    exit 1
fi

echo "Setup complete! Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Place your Gmail Takeout data in data/Takeout/"
echo "3. Run './run.sh --test' to verify everything works" 