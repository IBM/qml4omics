# Tutorial on Quantum Machine Learning (QML) for Healthcare and Life Science (HCLS) data.

 [![Minimum Python Version](https://img.shields.io/badge/Python-%3E=%203.9-blue)](https://www.python.org/downloads/) [![Maximum Python Version Tested](https://img.shields.io/badge/Python-%3C=%203.11-blueviolet)](https://www.python.org/downloads/) [![Supported Python Versions](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/downloads/) 
 
 This project has been tested and is compatible with Python versions 3.9, 3.10, and 3.11. While it might work on other versions, these are the officially supported and tested ones. 

## Getting Started

### Prerequisites

Before you can run this project, you need to have python installed on your system

#### Option 1: Setting up a Python Virtual Environment (venv)

This is the standard way to create an isolated Python Environment.

**Steps:**

1. **Install pip (if you don't have it):**
  ```bash
   python -m ensurepip --default-pip
  ```
  or on some systems:
 ```bash
  sudo apt update
  sudo apt install python3-pip
  ```
2. **Create a virtual environment:**
```bash
   python -m venv venv
  ```
This command creates a new directory named `venv` (you can choose a different name if you prefer) containing a copy of the Python interpreter and necessary supporting files.
3. **Activate the virtual environment:**
* **On macOS and Linux:**
```bash
  source venv/bin/activate
  ```
* **On Windows (command prompt):**
```bash
  venv\Scripts\activate
  ```
* **On Windows (PowerShell):**
```bash
  .\venv\Scripts\Activate.ps1
  ```
Once activated, you'll see `(venv)` at the beginning of your terminal prompt.

4. **Install project dependencies:**
   Once the virtual environment is activated, you can install the required packages listed in the `requirements.txt` file:
  ```bash
  pip install  -r requirements.txt
  ```
5. **Deactivate the virtual environment (when you are done):**
   ```bash
    deactivate
   ```
   This will return you to your base Python environment.

#### Option 2: Setting up a Conda Environment

1. Create the environment from the `requirements.txt` file.  This can be done using Anaconda, miniconda, miniforge, or any other environment manager.
```
conda create -n qml4omics python==3.11

```
* Note: if you receive the error `bash: conda: command not found...`, you need to install some form of anaconda in your development environment.
2. Activate the new environment:
```
conda activate qml4omics
pip install -r requirements.txt
```
3. Verify that the new environment and packages were installed correctly:
```
conda env list
pip list
```
<!-- * Additional resources:
   * [Connect to computing cluster](http://ccc.pok.ibm.com:1313/gettingstarted/newusers/connecting/)
   * [Set up / install Anaconda on remote linux server](https://kengchichang.com/post/conda-linux/)
   * [Set up remote development environment using VSCode](https://code.visualstudio.com/docs/remote/ssh) -->

<a name="running_comical"></a>
<!-- ### Running qml4omics -->

<!-- [![Notebook Template][notebook]](#running_comical) -->

<!-- 1. Request resources from computing cluster:
```
jbsub -cores 2+1 -q x86_1h -mem 5g -interactive bash
```
OR
Submit your job without the interactive session (shown later).  -->

<!-- 2. Activate the new environment:
```
conda activate qml4omics
``` -->
### qml4omics-profiler

The profiler relies on a `config.yaml` file for this setting and paremeters. **To run the project via the command prompt, you must have a correctly formatted `config.yaml`.

#### `config.yaml` Structure

The `config.yaml` file should be structured in YAML format. Please refer to the [`config.yaml`](./configs/config.yaml) file in this repository

**Important:**
* Ensure that the keys in your `config.yaml`  file match the expected configuration parameters.
* The project will likely fail or behave unexpectedly if the `config.yaml` file is missing, incorrectly formatted, or contains incorrect values.

#### Running via command prompt 
```
python qml4omics-profiler.py --config-name=config.yaml
```


### Help
```
python qml4omics-profiler.py --help
```

## Authors

Contributors and contact info:

* Bryan Raubenolt (raubenb@ccf.org)
* Aritra Bose (a.bose@ibm.com)
* Kahn Rhrissorrakrai (krhriss@us.ibm.com)
* Filippo Utro (futro@us.ibm.com)
* Akhil Mohan (mohana2@ccf.org)
* Daniel Blankenberg (blanked2@ccf.org)
