"""
Script de avaliação — compara baseline vs BERTimbau.

Gera tabelas comparativas e gráficos de performance por competência.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from transformers import AutoTokenizer

from src.evaluation.metrics import compute_all_metrics
from src.models.train_bert import MultiOutputRegressor, get_train_val_split

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

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
    return df


def evaluate_bert():
    """Avalia o BERTimbau salvo no conjunto de validação reproduzindo o split do treino."""
    config = load_config()
    model_dir = ROOT / "outputs" / "models" / "bertimbau" / "best"
    model_path = model_dir / "model.safetensors"

    if not model_dir.exists() or not model_path.exists():
        print("[WARN] Modelo BERTimbau não encontrado em outputs/models/bertimbau/best.")
        return None

    data_path = ROOT / "data" / "processed" / "essays_processed.csv"
    if not data_path.exists():
        print("[WARN] Dataset processado não encontrado para avaliação do BERT.")
        return None

    df = pd.read_csv(data_path)
    _, val_texts, _, val_labels, target_cols = get_train_val_split(
        df=df,
        seed=config["project"]["seed"],
        test_size=config["project"]["test_size"],
    )

    if load_safetensors is None:
        raise ImportError(
            "Dependência 'safetensors' não encontrada para carregar o modelo BERT salvo."
        )

    state_dict = load_safetensors(str(model_path))
    num_outputs = state_dict["regressor.4.bias"].shape[0]
    if num_outputs != len(target_cols):
        raise ValueError(
            "Checkpoint BERT incompatível com o dataset atual. "
            f"Saídas no checkpoint: {num_outputs}; esperadas: {len(target_cols)}. "
            "Retreine com `python -m src.models.train_bert` para gerar checkpoint com 5 competências."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = MultiOutputRegressor(config["bert"]["model_name"], num_outputs=num_outputs)
    model.load_state_dict(state_dict)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()

    batch_size = config["bert"]["batch_size"]
    predictions = []

    for start in range(0, len(val_texts), batch_size):
        batch_texts = val_texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=config["bert"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                token_type_ids=encoded.get("token_type_ids"),
            )
        predictions.append(outputs["logits"].cpu().numpy())

    y_pred = np.vstack(predictions)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_classes = config.get("n_classes", 6)
    rows = []
    for idx, target in enumerate(target_cols):
        metrics = compute_all_metrics(val_labels[:, idx], y_pred[:, idx], n_classes=n_classes)
        rows.append(
            {
                "model": "bertimbau",
                "target": target,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "qwk": metrics["qwk"],
            }
        )

    bert_df = pd.DataFrame(rows)
    report_path = ROOT / "outputs" / "reports" / "bert_results.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bert_df.to_csv(report_path, index=False)
    print(f"[INFO] Resultados BERT salvos em {report_path}")
    return bert_df


def format_target_label(target: str) -> str:
    return target.replace("_norm", "").upper()


def plot_all_models_comparison(results_df: pd.DataFrame, save_path: Path):
    """Compara baseline e BERT por métrica/competência."""
    if results_df is None or results_df.empty:
        return

    plot_df = results_df.copy()
    plot_df["target_label"] = plot_df["target"].map(format_target_label)
    long_df = plot_df.melt(
        id_vars=["model", "target_label"],
        value_vars=["qwk", "mae", "rmse"],
        var_name="metric",
        value_name="value",
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["qwk", "mae", "rmse"]):
        metric_df = long_df[long_df["metric"] == metric]
        sns.barplot(data=metric_df, x="target_label", y="value", hue="model", ax=ax)
        ax.set_title(metric.upper())
        ax.set_xlabel("Competência")
        ax.set_ylabel(metric.upper())
        ax.legend(title="Modelo", fontsize=8)

    plt.suptitle("Comparação Baselines vs BERTimbau", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"[INFO] Gráfico salvo em {save_path}")
    plt.close()


def plot_best_baseline_vs_bert(results_df: pd.DataFrame, save_path: Path):
    """Compara o melhor baseline (por QWK) contra BERT por competência."""
    if results_df is None or results_df.empty:
        return

    baselines = results_df[results_df["model"] != "bertimbau"].copy()
    bert = results_df[results_df["model"] == "bertimbau"].copy()
    if baselines.empty or bert.empty:
        return

    best_baseline = (
        baselines.sort_values(["target", "qwk"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()
    )
    best_baseline["model"] = "best_baseline"
    compare_df = pd.concat([best_baseline, bert], ignore_index=True)
    compare_df["target_label"] = compare_df["target"].map(format_target_label)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["qwk", "mae", "rmse"]):
        sns.barplot(data=compare_df, x="target_label", y=metric, hue="model", ax=ax)
        ax.set_title(f"{metric.upper()} — Melhor Baseline vs BERT")
        ax.set_xlabel("Competência")
        ax.set_ylabel(metric.upper())
        ax.legend(title="Modelo")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"[INFO] Gráfico salvo em {save_path}")
    plt.close()


def main():
    figures_dir = ROOT / "outputs" / "figures"
    reports_dir = ROOT / "outputs" / "reports"

    baseline_results = evaluate_baselines()
    bert_results = evaluate_bert()

    if baseline_results is None and bert_results is None:
        print("[WARN] Nenhum resultado disponível para comparação.")
        return

    frames = [df for df in [baseline_results, bert_results] if df is not None and not df.empty]
    combined = pd.concat(frames, ignore_index=True)
    combined_path = reports_dir / "model_comparison_results.csv"
    combined.to_csv(combined_path, index=False)
    print(f"[INFO] Tabela comparativa salva em {combined_path}")

    plot_all_models_comparison(combined, figures_dir / "baseline_vs_bert_comparison.png")
    plot_best_baseline_vs_bert(combined, figures_dir / "best_baseline_vs_bert.png")
    print("[INFO] Avaliação concluída.")


if __name__ == "__main__":
    main()
