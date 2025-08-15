#!/bin/bash

echo "Setting up RED Data Exporter Python Environment"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Using conda to create environment..."
    
    # Create conda environment
    conda create -n red_exporter python=3.10 -y
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate red_exporter
    
    # Install packages via conda (faster for some packages)
    echo "Installing packages via conda..."
    conda install -y numpy opencv
    conda install -y -c conda-forge pyyaml
    
    # Install remaining packages via pip
    echo "Installing remaining packages via pip..."
    pip install -r requirements.txt
    
    echo ""
    echo "✓ Conda environment 'red_exporter' created successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate red_exporter"
    
elif command -v python3 -m venv &> /dev/null; then
    echo "Using Python venv to create environment..."
    
    # Create virtual environment
    python3 -m venv red_yolo_exporter_env
    
    # Activate environment
    source red_yolo_exporter_env/bin/activate

    # Upgrade pip
    pip install --upgrade pip
    
    # Install packages
    echo "Installing packages via pip..."
    pip install -r requirements.txt
    
    echo ""
    echo "✓ Virtual environment 'red_yolo_exporter_env' created successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "  source red_yolo_exporter_env/bin/activate"

else
    echo "Neither conda nor python3 venv is available."
    echo "Please install conda or ensure python3-venv is installed."
    echo ""
    echo "You can still install packages globally with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "Setup complete! You can now run the export scripts."
