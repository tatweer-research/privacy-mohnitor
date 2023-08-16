#!/bin/bash

sudo apt-get update

# Clone the privacy-mohnitor repository
# Replace **** with the appropriate username and access token
git clone -b 6-improve-training-evaluation-code https://github.com/tatweer-research/privacy-mohnitor.git

# Install package venv on Debian
sudo apt install python3.8-venv

# Create a new virtual env
python3 -m venv venvs/pm

# Activate the virtual env
source venvs/pm/bin/activate

# Install requirements
pip install -r privacy-mohnitor/instruction_finetuning/requirements.txt