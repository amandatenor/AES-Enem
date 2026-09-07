"""
Pré-processamento das redações do Essay-BR.

Lê os arquivos brutos, limpa o texto e salva um DataFrame unificado
com colunas: [id, essay, C1, C2, C3, C4, C5, total].
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_essays(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Carrega redações do dataset AES-ENEM (HuggingFace)."""
    # Procura CSVs gerados pelo download
    csv_files = list(raw_dir.glob("aes_enem_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum dado encontrado em {raw_dir}. "
            "Execute: python -m src.data.download_essay_br"
        )

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    # Remove apenas duplicatas exatas de linha para evitar colapsar
    # redações válidas que compartilham o mesmo `id`.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = n_before - len(df)
    if removed > 0:
        print(f"[INFO] Removidas {removed} linhas duplicadas exatas.")

    # Renomear coluna de texto para padronizar
    if "essay_text" in df.columns:
        df = df.rename(columns={"essay_text": "essay"})

    print(f"[INFO] Carregadas {len(df)} redações.")
    return df


def clean_text(text: str) -> str:
    """Limpeza básica do texto da redação."""
    # Normalizar Unicode
    text = unicodedata.normalize("NFKC", text)
    # Remover múltiplos espaços em branco
    text = re.sub(r"\s+", " ", text)
    # Remover espaços no início/fim
    text = text.strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de pré-processamento."""
    df = df.copy()

    # Limpar texto
    if "essay" in df.columns:
        df["essay_clean"] = df["essay"].astype(str).apply(clean_text)

    # Remover redações vazias ou muito curtas (< 50 caracteres)
    if "essay_clean" in df.columns:
        mask = df["essay_clean"].str.len() >= 50
        removed = (~mask).sum()
        df = df[mask].reset_index(drop=True)
        if removed > 0:
            print(f"[INFO] Removidas {removed} redações muito curtas.")

    # Normalizar notas para [0, 1] se colunas de competência existem
    comp_cols = [c for c in df.columns if c in ["C1", "C2", "C3", "C4", "C5"]]
    for col in comp_cols:
        df[f"{col}_norm"] = df[col] / 200.0

    print(f"[INFO] Dataset pré-processado: {len(df)} redações.")
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_essays()
    df = preprocess(df)

    output_path = PROCESSED_DIR / "essays_processed.csv"
    df.to_csv(output_path, index=False)
    print(f"[INFO] Salvo em {output_path}")


if __name__ == "__main__":
    main()
