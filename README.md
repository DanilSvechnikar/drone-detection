# Drone Detection

[![PythonSupported](https://img.shields.io/badge/python-3.12-brightgreen.svg)](https://python3statement.org/#sections50-why)
[![poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

## Overview
The project includes data processing, neural network training and inference for drone detection.

## Repository Contents

- [demo](demo) - demonstrate the neural network
- [drone_detection](drone_detection) - source files of the project
- [notebooks](notebooks) - directory for `ipynb` notebooks to solve problems
- [models](models) - models for this project
- [config](config) - configuration files for model
- [data](data) - data for model training, validation and testing

## Deploying the environment (not for development)
### Preparations

1. Install make
    - Windows:

        Install [chocolatey](https://chocolatey.org/install) and install `make` with command:

    ```bash
    choco install make
    ```

    - Linux:

    ```bash
    sudo apt install build-essential
    ```

2. Install python 3.12
    - Windows

        Install with [official executable](https://www.python.org/downloads/)

    - Linux

    ```bash
    sudo apt install python3.12
    ```

3. Install poetry

   - Windows

        Use [official instructions](https://python-poetry.org/docs/#windows-powershell-install-instructions) or use `powershell` command:

    ```bash
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

   - Linux

        Use [official instructions](https://python-poetry.org/docs/#installing-with-the-official-installer) or bash command:

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

### Deployment
1. Clone project

    ```bash
    git clone https://github.com/DanilSvechnikar/drone-detection.git -b develop
    cd drone-detection
   ```

2. Packet initialization (about +-5GB!)
   ```bash
   make project-init
   ```

> **Note**: It is important that poetry sees the correct version of python on the system!

## Using a neural network
> **Note**: Importing torch, ultralytics packages and initializing the model takes a lot of time!

### Check cuda support
   ```bash
   poetry run python drone_detection/check_cuda.py
   ```

### Startup
There are two choices:
1. Through a [inference.py](./demo/inference.py):
   ```bash
   poetry run python ./demo/inference.py
   ```

    Or
   - name_data - filename at path ./data/demo_data/*your_file*. By **default** is test_img.jpg

   ```bash
   poetry run python ./demo/inference.py name_data=your_img_or_video
   ```

2. Through a [inference_jupyter.ipynb](./demo/inference_jupyter.ipynb) if you have the opportunity:
- On the plus side: no need to re-import libraries and model every time


## Data Links
  - https://universe.roboflow.com/dronedetection-osldo/drone-detection-nugaj/dataset/5 (Self-Labeled Data)
