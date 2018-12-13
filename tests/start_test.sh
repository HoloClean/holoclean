#!/usr/bin/env bash
# Set & move to home directory
source ../set_env.sh

# Launch tests.
echo "Launching tests..."
# python -m unittest discover .
python test_holoclean_repair.py

