"""
Treinamento dos modelos baseline (SVR, Random Forest, XGBoost).

Treina um modelo por competência e salva resultados.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

# Colunas que NÃO são features
NON_FEATURE_COLS = {"id", "id_prompt", "essay", "essay_clean", "essay_text", "essay_title",
                    "prompt", "reference", "supporting_text", "source", "split",
                    "essay_year",
                    "C1", "C2", "C3", "C4", "C5",
                    "total", "C1_norm", "C2_norm", "C3_norm", "C4_norm", "C5_norm"}


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_model(name: str, cfg: dict):
    """Instancia modelo pelo nome."""
    if name == "svr":
        return SVR(**cfg.get("svr", {}))
    elif name == "random_forest":
        return RandomForestRegressor(**cfg.get("random_forest", {}), random_state=42)
    elif name == "xgboost":
        return XGBRegressor(**cfg.get("xgboost", {}), random_state=42)
    else:
        raise ValueError(f"Modelo desconhecido: {name}")


def train_baseline():
    config = load_config()
    seed = config["project"]["seed"]
    test_size = config["project"]["test_size"]

    # Carregar dados com features
    data_path = ROOT / "data" / "processed" / "essays_with_features.csv"
    df = pd.read_csv(data_path)

    # Identificar colunas de features (apenas numéricas, excluindo alvos)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in NON_FEATURE_COLS]
    # Identificar colunas-alvo disponíveis
    target_cols = [c for c in ["C1_norm", "C2_norm", "C3_norm", "C4_norm", "C5_norm"]
                   if c in df.columns]

    if not target_cols:
        print("[WARN] Colunas de notas não encontradas. Usando 'total' se disponível.")
        if "total" in df.columns:
            target_cols = ["total"]
        else:
            raise ValueError("Nenhuma coluna-alvo encontrada no dataset.")

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models_dir = ROOT / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for model_name in config["baseline"]["models"]:
        print(f"\n{'='*50}")
        print(f"Modelo: {model_name.upper()}")
        print(f"{'='*50}")

        for target_col in target_cols:
            y = df[target_col].values
            mask = ~np.isnan(y)
            X_valid, y_valid = X_scaled[mask], y[mask]

            n_samples = len(y_valid)
            val_size = max(int(round(n_samples * test_size)), 8)
            val_size = min(val_size, n_samples - 2)
            val_size = max(2, val_size)

            bins = np.linspace(0, 1, config.get("n_classes", 6) + 1)
            stratify_labels = np.digitize(y_valid, bins[1:-1])
            class_counts = pd.Series(stratify_labels).value_counts()
            can_stratify = (class_counts >= 2).all() and len(class_counts) > 1

            X_train, X_test, y_train, y_test = train_test_split(
                X_valid,
                y_valid,
                test_size=val_size,
                random_state=seed,
                stratify=stratify_labels if can_stratify else None,
            )

            model = get_model(model_name, config["baseline"])
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Métricas
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # QWK (discretizando predições)
            from src.evaluation.metrics import quadratic_weighted_kappa
            qwk = quadratic_weighted_kappa(y_test, y_pred, n_classes=6)

            print(f"  {target_col}: MAE={mae:.4f}, RMSE={rmse:.4f}, QWK={qwk:.4f}")

            results.append({
                "model": model_name,
                "target": target_col,
                "mae": mae,
                "rmse": rmse,
                "qwk": qwk,
            })

            # Salvar modelo
            model_path = models_dir / f"baseline_{model_name}_{target_col}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump({"model": model, "scaler": scaler}, f)

    # Salvar resultados
    results_df = pd.DataFrame(results)
    results_path = ROOT / "outputs" / "reports" / "baseline_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Resultados salvos em {results_path}")

    return results_df


def main():
    train_baseline()


if __name__ == "__main__":
    main()
