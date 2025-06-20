# Configure the YAML file

In order to run mulitple experiments and for reprudicibility of the experiments, the use of the terminal `xxx`  is suggested.
To do so, a `yaml` file should be set. An example of a config file can be found in [config.yaml]()

In what follows the major component of the file are explained.

### Define the inputs

Inputs files can be placed in a single folder (e.g. `test_data`).
In the case all files in the folder need to be analized, one can set:

```yaml
config_file_name: 'basic_config'

# specify output directory where input datasets are located
folder_path: 'test_data/'
file_dataset: 'ALL'
```

or use a list as below, to only select specif files

```yaml
file_dataset: ['file1', 'file2', 'file3']
```

### Quantum backend

Define which quantum backend to be used for quantum computation.
For instance, in case of simulator:

```yaml
# choose a backend for the QML methods
backend: 'simulator'
```

in case one has access to IBM quantum device, one can specify the name. For instance to use the least busy:

```yaml
# choose a backend for the QML methods
backend: 'ibm_least'
```

IBM runtime credential should be stored n your device, in case of json path, it can be specified for instance as:

```yaml
qiskit_json_path: '~/.qiskit/qiskit-ibm.json'
```

### Embedding

To specify the embedding method for reducing dimensionality (number of features) in your data set, the following list 

```yaml
embeddings: ['pca', 'nmf']
```

in case of no embedding need to be applied please specify it as:

```yaml
embeddings: ['none']
```

###  Model selection
ML model can be selected as a list