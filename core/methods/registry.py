from __future__ import annotations
from typing import Dict, Callable
from .isolation_forest import IForest
from .lof import LOF
from .ocsvm import OCSVM
from .pca_recon import PCAReconstruction
from .hbos import HBOS
from .ensemble import MeanPercentileEnsemble
from core.methods.ecod import ECOD
from core.methods.deep_svdd import DeepSVDD
from core.methods.copod import COPOD
from core.methods.autoencoder import AutoEncoder
from core.methods.knn import KNN
from core.methods.sdo import SDO



def ENSEMBLE_FACTORY(selected_names):
    def make():
        bases = [lambda n=n: REGISTRY[n]() for n in selected_names]  # no params version for simplicity
        return MeanPercentileEnsemble(bases)
    return make


# Factories with minimal kwargs; extend later (Optuna, presets, etc.)
REGISTRY: Dict[str, Callable[..., object]] = {
"IsolationForest": lambda **kw: IForest(**kw),
"LOF": lambda **kw: LOF(**kw),
"OneClassSVM": lambda **kw: OCSVM(**kw),
"PCA(Reconstruction)": lambda **kw: PCAReconstruction(**kw),
"HBOS": lambda **kw: HBOS(**kw),
"ECOD": lambda **kw: ECOD(**kw),
"DeepSVDD": lambda **kw: DeepSVDD(**kw),
"COPOD": lambda **kw: COPOD(**kw),
"AutoEncoder": lambda **kw: AutoEncoder(**kw),
"KNN": lambda **kw: KNN(**kw),
"SDO": lambda **kw: SDO(**kw),
}



DEFAULT_KWARGS = {
"IsolationForest": {"n_estimators": 200, "contamination": "auto"},
"LOF": {"n_neighbors": 20, "contamination": "auto"},
"OneClassSVM": {"kernel": "rbf", "nu": 0.5, "gamma": "scale"},
"PCA(Reconstruction)": {"n_components": None, "scale": True},
"HBOS": {"n_bins": 10, "alpha": 0.1, "tol": 0.5},
"ECOD": {"contamination": 0.1},
"DeepSVDD": {"contamination": 0.1, "epochs": 10, "batch_size": 32},
"COPOD": {"contamination": 0.1},
"AutoEncoder": {"contamination": 0.1, "epochs": 50, "batch_size": 32},
"KNN": {"contamination": 0.1, "n_neighbors": 5, "method": "largest"},
"SDO": {"contamination": 0.1, "n_observers": 50},
}

