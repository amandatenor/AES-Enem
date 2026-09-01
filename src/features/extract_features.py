"""
Engenharia de features textuais para o modelo baseline.

Extrai features léxicas, sintáticas, de coesão e de legibilidade.
"""

import re
from typing import Dict

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    print("[WARN] Modelo spaCy pt_core_news_sm não encontrado. "
          "Execute: python -m spacy download pt_core_news_sm")
    nlp = None


# --- Conectivos comuns em português ---
CONECTIVOS = {
    "aditivos": ["e", "nem", "também", "além disso", "ademais", "bem como",
                 "não só", "como também"],
    "adversativos": ["mas", "porém", "contudo", "todavia", "entretanto",
                     "no entanto", "não obstante"],
    "conclusivos": ["portanto", "logo", "assim", "por conseguinte",
                    "em suma", "enfim", "dessa forma", "desse modo"],
    "causais": ["porque", "pois", "visto que", "já que", "uma vez que",
                "dado que"],
    "temporais": ["quando", "enquanto", "antes que", "depois que",
                  "assim que", "logo que"],
}

# Palavras-chave de proposta de intervenção (C5)
PROPOSTA_KEYWORDS = [
    "governo", "estado", "ministério", "políticas públicas",
    "educação", "conscientização", "campanha", "lei", "legislação",
    "mídia", "escola", "sociedade", "ong", "família",
    "a fim de", "com o intuito de", "para que", "de modo que",
]


def extract_basic_features(text: str) -> Dict[str, float]:
    """Features léxicas básicas."""
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    n_chars = len(text)
    n_words = len(words)
    n_sentences = max(len(sentences), 1)
    n_paragraphs = max(len(paragraphs), 1)
    n_unique_words = len(set(w.lower() for w in words))

    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_sentences": n_sentences,
        "n_paragraphs": n_paragraphs,
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "avg_sentence_length": n_words / n_sentences,
        "avg_paragraph_length": n_words / n_paragraphs,
        "lexical_diversity": n_unique_words / n_words if n_words else 0,
        "long_words_ratio": sum(1 for w in words if len(w) > 6) / n_words if n_words else 0,
    }


def extract_connectivity_features(text: str) -> Dict[str, float]:
    """Contagem de conectivos por tipo (proxy para coesão — C4)."""
    text_lower = text.lower()
    features = {}
    total = 0
    for tipo, conectivos in CONECTIVOS.items():
        count = sum(text_lower.count(c) for c in conectivos)
        features[f"conectivos_{tipo}"] = count
        total += count
    features["conectivos_total"] = total
    return features


def extract_proposal_features(text: str) -> Dict[str, float]:
    """Features de proposta de intervenção (C5)."""
    text_lower = text.lower()
    count = sum(1 for kw in PROPOSTA_KEYWORDS if kw in text_lower)
    return {
        "proposta_keywords": count,
        "has_proposta": int(count >= 2),
    }


def extract_spacy_features(text: str) -> Dict[str, float]:
    """Features sintáticas via spaCy (POS tags, entidades)."""
    if nlp is None:
        return {}

    # Limitar tamanho para performance
    doc = nlp(text[:5000])

    pos_counts: Dict[str, int] = {}
    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

    n_tokens = max(len(doc), 1)
    features = {
        f"pos_{pos}_ratio": count / n_tokens
        for pos, count in pos_counts.items()
    }
    features["n_entities"] = len(doc.ents)
    features["n_unique_entities"] = len(set(ent.text for ent in doc.ents))

    return features


def extract_all_features(text: str) -> Dict[str, float]:
    """Extrai todas as features de um texto."""
    features = {}
    features.update(extract_basic_features(text))
    features.update(extract_connectivity_features(text))
    features.update(extract_proposal_features(text))
    features.update(extract_spacy_features(text))
    return features


def process_dataset(df: pd.DataFrame, text_col: str = "essay_clean") -> pd.DataFrame:
    """Extrai features para todo o DataFrame."""
    print("[INFO] Extraindo features textuais...")
    feature_records = []
    for text in tqdm(df[text_col], desc="Extraindo features"):
        feature_records.append(extract_all_features(str(text)))

    features_df = pd.DataFrame(feature_records)
    # Preencher NaN (algumas features spaCy podem não existir em todos os textos)
    features_df = features_df.fillna(0.0)

    result = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    print(f"[INFO] {len(features_df.columns)} features extraídas.")
    return result


def main():
    from pathlib import Path

    processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    input_path = processed_dir / "essays_processed.csv"
    output_path = processed_dir / "essays_with_features.csv"

    df = pd.read_csv(input_path)
    df = process_dataset(df)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Salvo em {output_path}")


if __name__ == "__main__":
    main()
