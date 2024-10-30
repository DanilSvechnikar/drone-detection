# Development Guide

Preparing the development environment

## Table of contents

- [Table of contents](#table-of-contents)
- [Preparations](#preparations)
- [Project Initialization](#project-initialization)


## Preparations

1. Install make
    - Windows:

        Install [chocolatey](https://chocolatey.org/install) and install `make` with command:

    ```bash
    choco install make
    ```

    - Linux:

    ```bash
    sudo apt-get install build-essential
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

        Use [official instructions](https://python-poetry.org/docs/#windows-powershell-install-instructions) or use powershell command:

    ```bash
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

   - Linux

        Use [official instructions](https://python-poetry.org/docs/#installing-with-the-official-installer) or bash command:

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```


## Project Initialization

1. Clone existing repository:

```bash
git clone https://github.com/DanilSvechnikar/drone-detection.git && cd drone-detection
```

2. Install packages and `pre-commit` hooks:

```bash
make project-init-dev
```


[Table of contents](#table-of-contents)
