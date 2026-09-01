"""
Métricas de avaliação para AES.

Inclui Quadratic Weighted Kappa (QWK), principal métrica do campo.
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error


def discretize_scores(scores: np.ndarray, n_classes: int = 6) -> np.ndarray:
    """Discretiza scores contínuos [0,1] em classes {0,1,...,n_classes-1}."""
    bins = np.linspace(0, 1, n_classes + 1)
    # np.digitize retorna 1-indexed; subtrair 1 e clipar
    discrete = np.digitize(scores, bins[1:-1])
    return np.clip(discrete, 0, n_classes - 1)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray,
                              n_classes: int = 6) -> float:
    """
    Calcula o Quadratic Weighted Kappa entre notas reais e preditas.

    Scores são primeiro discretizados para a escala ENEM (0-5 → 0,40,...,200).
    """
    y_true_d = discretize_scores(np.array(y_true), n_classes)
    y_pred_d = discretize_scores(np.array(y_pred), n_classes)
    return cohen_kappa_score(y_true_d, y_pred_d, weights="quadratic")


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         n_classes: int = 6) -> dict:
    """Calcula todas as métricas de avaliação."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "qwk": quadratic_weighted_kappa(y_true, y_pred, n_classes),
    }
