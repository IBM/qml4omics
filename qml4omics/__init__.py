
# ====== Import our CML functions ======
from qml4omics.classical.supervised.compute_svc import compute_svc, compute_svc_opt
from qml4omics.classical.supervised.compute_dt import compute_dt, compute_dt_opt
from qml4omics.classical.supervised.compute_nb import compute_nb, compute_nb_opt
from qml4omics.classical.supervised.compute_lr import compute_lr, compute_lr_opt
from qml4omics.classical.supervised.compute_rf import compute_rf, compute_rf_opt
from qml4omics.classical.supervised.compute_mlp import compute_mlp, compute_mlp_opt

# ====== Import our QML functions ======
from qml4omics.quantum.supervised.compute_qnn import compute_qnn
from qml4omics.quantum.supervised.compute_qsvc import compute_qsvc
from qml4omics.quantum.supervised.compute_vqc import compute_vqc
from qml4omics.quantum.supervised.compute_pqk import compute_pqk


# ====== Import our embedding functions ======
from qml4omics.embeddings.embed import get_embeddings


# ====== Import helper functions ======
from qml4omics.utils.helper_fn import scaler_fn, feature_encoding
from qml4omics.utils.qc_winner_finder import qml_winner
from qml4omics.utils.dataset_checkpoint import checkpoint_restart

# ====== Import evaluation functions ======
from qml4omics.evaluation.model_evaluation import modeleval
from qml4omics.evaluation.dataset_evaluation import evaluate
from qml4omics.evaluation.model_run import model_run

# ====== Import visualization functions ======
from qml4omics.visualization.visualize_correlation import plot_results_correlation, compute_results_correlation 


