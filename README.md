# Drone Detection

[![PythonSupported](https://img.shields.io/badge/python-3.12-brightgreen.svg)](https://python3statement.org/#sections50-why)
[![poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

- [Overview](#overview)
- [Repository contents](#repository-contents)
- [System Requirements](#system-requirements)
- [Deploy Locally](#deploy-locally)
- [Code Launch](#code-launch)
- [Deploy Docker](#deploy-docker)
- [Data Links](#data-links)

## Overview
The project includes data processing, neural network training and inference for drone detection.


## System Requirements
- Python version: >= 3.12


## Repository Contents
- [demo](demo) - demonstrate the neural network
- [drone_detection](drone_detection) - source files of the project
- [notebooks](notebooks) - directory for `ipynb` notebooks to solve problems
- [models](models) - models for this project
- [config](config) - configuration files for model
- [data](data) - data for model training, validation and testing


## Deploy Locally
> Not for development!

1. Clone existing repository:
    ```bash
    git clone https://github.com/DanilSvechnikar/drone-detection.git && cd drone-detection
    ```

2. Activate the virtual environment!
<br></br>

3. Install packages (about *+-5GB !*)
    ```bash
    pip install --no-cache-dir -r requirements.txt
    ```


## Code Launch

Check cuda support if you want:
   ```bash
   python drone_detection/cuda_utils.py
   ```

Run [inference.py](./demo/inference.py):

- name_data - filename at path ./data/demo_data/`your_file`. By `default` is test_5.mp4

   ```bash
   python ./demo/inference.py
   ```

Or run [inference_jupyter.ipynb](./demo/inference_jupyter.ipynb) \
On the plus side: no need to re-import libraries and model every time


## Deploy Docker
I'll write later

## Data Links
  - https://app.roboflow.com/dronedetection-osldo/drone-detection-ch74g/1 (Self-Labeled Data)
