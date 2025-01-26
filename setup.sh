#!/bin/bash

echo "Starting setup..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt
pip install tqdm
pip install jupyterlab
pip install jupytext

# Run tests if requested
if [ "$1" = "test" ]; then
    echo -e "\nRunning tests..."
    python tests/run_tests.py
fi

echo -e "\nSetup complete! Virtual environment is activated and packages are installed."
echo -e "\nTo run in JupyterLab:"
echo "1. jupyter lab"
echo "2. Either:"
echo "   a) Convert Python script: jupytext --to notebook quantum_percolation_simulation.py"
echo "   b) Create new notebook and copy code from quantum_percolation_simulation.py"
echo -e "\nOr to run as Python script:"
echo "python quantum_percolation_simulation.py"
echo -e "\nIf you see any missing package errors, run:"
echo "pip install <package_name>"
echo "pip freeze > requirements.txt"