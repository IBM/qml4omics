# Tutorial on Quantum Machine Learning (QML) for Healthcare and Life Science (HCLS) data.

## Getting Started

### Prerequisites
1. Create the environment from the `requirements.txt` file.  This can be done using anaconda, miniconda, miniforge, or any other environment manager.
```
conda create -n qml4omics python==3.11

```
* Note: if you receive the error `bash: conda: command not found...`, you need to install some form of anaconda to your development environment.
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
4. Run qml4omics pipeline:
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
