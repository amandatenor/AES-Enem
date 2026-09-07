"""
Script de avaliação — compara baseline vs BERTimbau.

Gera tabelas comparativas e gráficos de performance por competência.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.evaluation.metrics import compute_all_metrics
from src.models.train_bert import EssayDataset, MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def _get_target_cols(df: pd.DataFrame, config: dict):
    configured_competencias = config.get("competencias", ["C1", "C2", "C3", "C4", "C5"])
    configured_target_cols = [f"{comp}_norm" for comp in configured_competencias]
    target_cols = [col for col in configured_target_cols if col in df.columns]
    if target_cols:
        return target_cols

    if "total" in df.columns:
        df["total_norm"] = df["total"] / 1000.0
        return ["total_norm"]

    raise ValueError("Nenhuma coluna-alvo encontrada para avaliação.")


def evaluate_baselines():
    """Carrega resultados do baseline."""
    results_path = ROOT / "outputs" / "reports" / "baseline_results.csv"
    if not results_path.exists():
        print("[WARN] Resultados baseline não encontrados. Treine primeiro.")
        return None

    df = pd.read_csv(results_path)
    print("\n=== Resultados Baseline ===")
    print(df.to_string(index=False))
    return df


def evaluate_bert():
    """Avalia o BERT treinado e salva métricas por competência."""
    config = load_config()
    seed = config["project"]["seed"]
    test_size = config["project"]["test_size"]
    bert_cfg = config["bert"]
    n_classes = config.get("n_classes", 6)

    model_dir = ROOT / "outputs" / "models" / "bertimbau" / "best"
    data_path = ROOT / "data" / "processed" / "essays_processed.csv"
    report_path = ROOT / "outputs" / "reports" / "bert_results.csv"

    if not model_dir.exists():
        print(f"[WARN] Modelo BERT não encontrado em {model_dir}. Treine primeiro.")
        return None
    if not data_path.exists():
        print(f"[WARN] Dados processados não encontrados em {data_path}.")
        return None

    df = pd.read_csv(data_path)
    target_cols = _get_target_cols(df, config)
    num_outputs = len(target_cols)
    texts = df["essay_clean"].astype(str).tolist()
    labels = df[target_cols].values.astype(np.float32)

    _, val_texts, _, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    dataset = EssayDataset(val_texts, val_labels, tokenizer, bert_cfg["max_length"])
    dataloader = DataLoader(dataset, batch_size=bert_cfg["batch_size"], shuffle=False)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = MultiOutputRegressor(bert_cfg["model_name"], num_outputs=num_outputs)

    safetensors_path = model_dir / "model.safetensors"
    pytorch_bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
    elif pytorch_bin_path.exists():
        state_dict = torch.load(str(pytorch_bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"Pesos do modelo não encontrados em {model_dir} "
            "(esperado: model.safetensors ou pytorch_model.bin)."
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"].to(device)
            logits = model(**inputs)["logits"]
            predictions.append(logits.cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0)
    y_pred = np.clip(y_pred, 0.0, 1.0)
    y_true = val_labels

    results = []
    for idx, target_col in enumerate(target_cols):
        metrics = compute_all_metrics(y_true[:, idx], y_pred[:, idx], n_classes=n_classes)
        results.append({
            "model": "bertimbau",
            "target": target_col,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "qwk": metrics["qwk"],
        })

    results_df = pd.DataFrame(results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(report_path, index=False)
    print(f"\n=== Resultados BERT ===")
    print(results_df.to_string(index=False))
    print(f"[INFO] Resultados BERT salvos em {report_path}")
    return results_df


def plot_comparison(results_df: pd.DataFrame, save_path: Path = None):
    """Gráfico de barras comparando QWK/MAE/RMSE por modelo e competência."""
    if results_df is None or results_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric in zip(axes, ["qwk", "mae", "rmse"]):
        pivot = results_df.pivot(index="target", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(metric.upper())
        ax.set_xlabel("Competência")
        ax.legend(title="Modelo")

    plt.suptitle("Comparação de Modelos por Competência", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Gráfico salvo em {save_path}")

    plt.close()


def plot_qwk_heatmap(results_df: pd.DataFrame, save_path: Path):
    """Heatmap de QWK por modelo e competência."""
    if results_df is None or results_df.empty:
        return

    qwk_df = results_df.copy()
    qwk_df["target"] = qwk_df["target"].str.replace("_norm", "", regex=False)
    pivot = qwk_df.pivot(index="model", columns="target", values="qwk")

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("QWK por Modelo e Competência")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("QWK")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Gráfico salvo em {save_path}")
    plt.close()


def plot_mae_vs_qwk(results_df: pd.DataFrame, save_path: Path):
    """Dispersão MAE vs QWK para evidenciar erro contínuo vs concordância ordinal."""
    if results_df is None or results_df.empty:
        return

    plot_df = results_df.copy()
    plot_df["target"] = plot_df["target"].str.replace("_norm", "", regex=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, group in plot_df.groupby("model"):
        ax.scatter(group["mae"], group["qwk"], label=model_name, s=60)
        for _, row in group.iterrows():
            ax.annotate(row["target"], (row["mae"], row["qwk"]), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Relação entre MAE e QWK por Competência")
    ax.set_xlabel("MAE (menor é melhor)")
    ax.set_ylabel("QWK (maior é melhor)")
    ax.legend(title="Modelo")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Gráfico salvo em {save_path}")
    plt.close()


def main():
    reports_dir = ROOT / "outputs" / "reports"
    figures_dir = ROOT / "outputs" / "figures"

    baseline_results = evaluate_baselines()
    bert_results = evaluate_bert()

    available_results = [df for df in [baseline_results, bert_results] if df is not None and not df.empty]
    if available_results:
        merged_results = pd.concat(available_results, ignore_index=True)
        merged_path = reports_dir / "model_comparison_results.csv"
        reports_dir.mkdir(parents=True, exist_ok=True)
        merged_results.to_csv(merged_path, index=False)
        print(f"[INFO] Resultados combinados salvos em {merged_path}")

        plot_comparison(merged_results, figures_dir / "model_comparison.png")
        plot_qwk_heatmap(merged_results, figures_dir / "qwk_heatmap.png")
        plot_mae_vs_qwk(merged_results, figures_dir / "mae_vs_qwk.png")
    else:
        print("[WARN] Nenhum resultado disponível para comparação.")

    print("\n[INFO] Avaliação concluída.")


if __name__ == "__main__":
    main()
