API Overview
============
Quick overview of the :ref:`methods` and :ref:`datasets` available in qbiocode.

.. _methods:

Methods
-------
Depending on the underlying foundations in qbiocode can be....

Embeddings
^^^^^^^^^^
Collection of common embeddings  (:mod:`qbiocode.embeddings`) functionalities.

.. autosummary::
    ~qbiocode.embeddings.embed.get_embeddings


Evaluation
^^^^^^^^^^
The :mod:`qbiocode.evaluation` submodule of qbiocode computes the evaluation metrics for the input dataset and the models.

Data Evaluation
""""""""""""""""
Depending on the underlying mathematical foundations, they can be classified into the following categories: (i)..

.. autosummary::
    ~qbiocode.evaluation.dataset_evaluation.evaluate

Model Evaluation
""""""""""""""""

.. autosummary::
    ~qbiocode.evaluation.model_evaluation.modeleval


Model Computation 
^^^^^^^^^^^^^^^^^
qbiocode brings together a number of established machine learning model both from classical and quantum (:mod:`qbiocode.learning`).
Multiple models can be run via the following 

.. autosummary::
    ~qbiocode.evaluation.model_run.model_run

Classical Models
""""""""""""""""

Classical model....

.. autosummary::
    ~qbiocode.learning.compute_dt.compute_dt  
    ~qbiocode.learning.compute_lr.compute_lr 
    ~qbiocode.learning.compute_mlp.compute_mlp
    ~qbiocode.learning.compute_nb.compute_nb 
    ~qbiocode.learning.compute_rf.compute_rf
    ~qbiocode.learning.compute_svc.compute_svc 

Each of them has an alternative function where grid search parameter can be given as input. Details can be found in the specific :mod:`qbiocode.learning` submodules.

Quantum Models
""""""""""""""

Quantum model....


.. autosummary::
    ~qbiocode.learning.compute_qnn.compute_qnn
    ~qbiocode.learning.compute_qsvc.compute_qsvc
    ~qbiocode.learning.compute_vqc.compute_vqc
    ~qbiocode.learning.compute_pqk.compute_pqk



Visualisation
^^^^^^^^^^^^^
The plotting module (:mod:`qbiocode.visualization`) enables the user to visualise the data and provides out-of-the-box plots for some
of the metrics.

.. autosummary::
    ~qbiocode.visualization.visualize_correlation.compute_results_correlation
    ~qbiocode.visualization.visualize_correlation.plot_results_correlation
    
.. _datasets:

Datasets
-------- 
qbiocode provides... 

References
^^^^^^^^^^

