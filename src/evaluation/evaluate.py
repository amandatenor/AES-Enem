"""
Script de avaliação — compara baseline vs BERTimbau.

Gera tabelas comparativas e gráficos de performance por competência.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import compute_all_metrics

ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def evaluate_baselines():
    """Carrega resultados do baseline e gera visualizações."""
    results_path = ROOT / "outputs" / "reports" / "baseline_results.csv"
    if not results_path.exists():
        print("[WARN] Resultados baseline não encontrados. Treine primeiro.")
        return None

    df = pd.read_csv(results_path)
    print("\n=== Resultados Baseline ===")
    print(df.to_string(index=False))
    return df


def plot_comparison(results_df: pd.DataFrame, save_path: Path = None):
    """Gráfico de barras comparando QWK por modelo e competência."""
    if results_df is None or results_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric in zip(axes, ["qwk", "mae", "rmse"]):
        pivot = results_df.pivot(index="target", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(metric.upper())
        ax.set_xlabel("Competência")
        ax.legend(title="Modelo")

    plt.suptitle("Comparação de Modelos Baseline", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Gráfico salvo em {save_path}")

    plt.close()


def main():
    figures_dir = ROOT / "outputs" / "figures"

    results = evaluate_baselines()
    if results is not None:
        plot_comparison(results, figures_dir / "baseline_comparison.png")

    print("\n[INFO] Avaliação concluída.")


if __name__ == "__main__":
    main()
