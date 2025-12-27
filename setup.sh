#!/bin/bash

# Setup script for Streamlit Cloud deployment
# This script is automatically run by Streamlit Cloud during deployment

# Create necessary directories
mkdir -p .streamlit
mkdir -p temp

# Set environment variables
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "Setup complete!"
