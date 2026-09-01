"""
Download do dataset AES-ENEM do HuggingFace.

Dataset: kamel-usp/aes_enem_dataset
Configs disponíveis: JBCS2025, PROPOR2024, gradesThousand
Referência: https://huggingface.co/datasets/kamel-usp/aes_enem_dataset
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DATASET_ID = "kamel-usp/aes_enem_dataset"
# JBCS2025 é a config mais completa (500 train + 132 val + 138 test)
DEFAULT_CONFIG = "JBCS2025"


def download_aes_enem(config: str = DEFAULT_CONFIG) -> pd.DataFrame:
    """Baixa o dataset AES-ENEM do HuggingFace e salva como CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / f"aes_enem_{config}.csv"

    if output_path.exists():
        print(f"[INFO] Dataset já existe em {output_path}")
        return pd.read_csv(output_path)

    print(f"[INFO] Baixando {DATASET_ID} (config={config})...")
    ds = load_dataset(DATASET_ID, config)

    # Combinar todos os splits
    dfs = []
    for split_name, split_ds in ds.items():
        df_split = split_ds.to_pandas()
        df_split["split"] = split_name
        dfs.append(df_split)

    df = pd.concat(dfs, ignore_index=True)

    # Expandir coluna 'grades' (lista) em C1..C5 + total
    grades_expanded = pd.DataFrame(
        df["grades"].tolist(),
        columns=["C1", "C2", "C3", "C4", "C5", "total"],
    )
    df = pd.concat([df.drop(columns=["grades"]), grades_expanded], axis=1)

    df.to_csv(output_path, index=False)
    print(f"[INFO] Salvo {len(df)} redações em {output_path}")
    return df


def main():
    download_aes_enem()


if __name__ == "__main__":
    main()
