# Configure the YAML file

In order to run mulitple experiments and for reprudicibility of the experiments, the use of the terminal `xxx`  is suggested.
To do so, a `yaml` file should be set. An example of a config file can be found in [config.yaml](../../../apps/profiler/configs/config.yaml)

In what follows the major component of the file are explained.

## Define the inputs

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

To ensure deterministic and reproducible outcome a seed for should be fixed as:


```yaml
seed: 42
q_seed: 42
```

`q_seed` refers to the seed used by quantum algoriths.

## Quantum backend

Define which quantum backend to be used for quantum computation.
For instance, in case of simulator:

```yaml
# choose a backend for the QML methods
backend: 'simulator'
```

In case one has access to IBM quantum device, one can specify the name. For instance to use the least busy:

```yaml
# choose a backend for the QML methods
backend: 'ibm_least'
```

as well as the number of shoots:

```yaml
# number of shoots
shots: 1024
```

Level of error mitigation for the QML methods (ranges from 1-3) can be set as:

```yaml
resil_level: 1
```

IBM runtime credential should be stored n your device, in case of json path, it can be specified for instance as:

```yaml
qiskit_json_path: '~/.qiskit/qiskit-ibm.json'
```

## Embedding

To specify the embedding method for reducing dimensionality (number of features) in your data set, the following list 

```yaml
embeddings: ['pca', 'nmf']
```

while the number of dimensions (features) to embed (reduce) your data down need to be set as:

```yaml
n_components: 3
```

In case of no embedding need to be applied please specify it as:

```yaml
embeddings: ['none']
```

## Train/Test parameters

Ratio of train:test the data is split into via the `test_size` option.
User can also specify if data need to be scaled and stratified. For instace, in this case 70:30 train:test ragio, with stratification and scaling:

```yaml
test_size: 0.3
stratify: ['y']
scaling: ['True']
```

## Model selection
ML model can be selected as a list, between the following, options:
`svc` (Support Vector Machine), `dt` (Decision Tree), `lr` (Logistic Regression), `nb` (Naive Bayes), `rf` (Random Forest), `mlp` (Multi-layer Perceptron), `qsvc` (Quantum svc), `vqc` (Variational Quantum Classifier), `qnn` (Quantum Neural Network), `pqk` (Projection Quantum Kernel). For instace if one would want to run all models, would set:

```yaml
model: ['svc', 'dt', 'lr', 'nb', 'rf', 'mlp', 'qsvc', 'vqc', 'qnn', 'pqk']
```

parameters to use for each ML method could be set.
Each CML method will have a grid search set of arguments as well as a standard set of arguments to pass, which will usually be the same as the grid search arguments except not as a list of parameters to iterate over.

For instance, for `svc`:

```yaml
svc_args: { 'C': 0.01, 'gamma': 0.1, kernel: 'linear' }

gridsearch_svc_args: {'C': [0.1, 1, 10, 100], 
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'kernel': ['linear', 'rbf', 'poly','sigmoid']
                     }
```

Parameter name and range values description can be found in the corresponding sklearn page for CML. 

```{Note}
For QML grid search, one must use generate_experiments.ipynb in tutorial_notebooks/analyses/qml_experiment_generators/,
which will generate individual config.yaml files for each combination of parameters.
```
