
 [![Minimum Python Version](https://img.shields.io/badge/Python-%3E=%203.9-blue)](https://www.python.org/downloads/) [![Maximum Python Version Tested](https://img.shields.io/badge/Python-%3C=%203.11-blueviolet)](https://www.python.org/downloads/) [![Supported Python Versions](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/downloads/) 
 
 This project has been tested and is compatible with Python versions 3.9, 3.10, and 3.11. While it might work on other versions, these are the officially supported and tested ones. 

# Getting Started


Before you can run this project, you need to have python installed on your system

## Option 1: Setting up a Python Virtual Enviroment (venv)

This is the standard way to create an isolated Python enviroment.

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
2. **Create a virtual enviroment:**
```bash
   python -m venv venv
  ```
This command creatas a new directory named `venv` (you can choose a different name if you prefer) containing a copy of the Python interpreter and necessary supporting files.

3. **Activate the virtual enviroment:**
* **On macOS and Linux:**
```bash
  source venv/bin/activate
  ```
* **On Windows (command promt):**
```bash
  venv\Scripts\activate
  ```
* **On Windows (PowerShell):**
```bash
  .\venv\Scripts\Activate.ps1
  ```
Once the activated, you'll see `(venv)` at the beginning of your terminal promt.

4. **Install project dependencies:**
   Once the virtual enviroment is activated, you can install the required packages listed in the `requirements.txt` file:
  ```bash
  pip install  .
  ```
5. **Deactivate the virtual enviroment (when you are done):**
   ```bash
    deactivate
   ```
   This will return you to your base Python enviroment.

## Option 2: Setting up a Conda Enviroment

1. Create the environment from the `requirements.txt` file.  This can be done using anaconda, miniconda, miniforge, or any other environment manager.
```
conda create -n qbc python==3.11

```
* Note: if you receive the error `bash: conda: command not found...`, you need to install some form of anaconda to your development environment.
2. Activate the new environment:
```
conda activate qbc
pip install .
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

<a name="running_qbiocode"></a>
<!-- ### Running QBioCode -->

<!-- [![Notebook Template][notebook]](#running_comical) -->

<!-- 1. Request resources from computing cluster:
```
jbsub -cores 2+1 -q x86_1h -mem 5g -interactive bash
```
OR
Submit your job without the interactive session (shown later).  -->

<!-- 2. Activate the new environment:
```
conda activate qbiocode
``` -->
