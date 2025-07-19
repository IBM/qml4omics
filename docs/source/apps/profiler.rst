
#######################################
QProfiler
#######################################


QProfiler helps analize quickly your data via a series basic models (classical and quantum) allowing parameter tuning without much code.
Furthemore, it helps understand the data charactestics via the use of complexity measures.


.. note::
    Before you start, make sure that you have installed QBioCode correctly by following the  `Installation <https://ibm.github.io/QBioCode/installation.html>`_ guide.


The QProfiler relies on a `config.yaml` file for tis setting and paremeters. **To run the project via the command prompt, you must have a correctly formatted `config.yaml`.**

`config.yaml` Structure
========================

The `config.yaml` file should be a structured in YAML format. Please refer to the `config.yaml <./configs/config.yaml>`_ file in this repository as an example.

.. note::
    Ensure that the keys in your `config.yaml`  file match the expected configuration parameters.
    The project will likely fail or behave unexpectedly if the `config.yaml` file is missing, incorrectly formatted, or contains incorrect values.

Please see also the documentation below for the details of each parameter in the `config.yaml` in:

.. toctree::
   :maxdepth: 2
   
   Configure YAML <config.md>

Running via command prompt 
===========================

To run QProfiler use the following command in your terminal  

.. code-block:: bash

    python qprofiler.py --config-name=config.yaml


Help
===========================

You can also get extra information using the following command in your terminal  

.. code-block:: bash

    python qprofiler.py --help

=======
    
