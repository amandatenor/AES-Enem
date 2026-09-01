"""
Interface Streamlit para demonstração do AES-ENEM.

Permite ao usuário colar uma redação e receber notas preditas por competência.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.extract_features import extract_all_features

st.set_page_config(page_title="AES-ENEM", page_icon="📝", layout="wide")

st.title("📝 AES-ENEM — Correção Automática de Redações")
st.markdown(
    "Sistema de Automatic Essay Scoring para redações do ENEM. "
    "Cole sua redação abaixo para receber uma avaliação por competência."
)

COMPETENCIAS = {
    "C1": "Domínio da norma culta da língua escrita",
    "C2": "Compreensão da proposta e aplicação de conceitos",
    "C3": "Seleção e organização de informações e argumentos",
    "C4": "Conhecimento dos mecanismos linguísticos de coesão",
    "C5": "Elaboração de proposta de intervenção",
}

# --- Input ---
essay = st.text_area(
    "Cole sua redação aqui:",
    height=300,
    placeholder="Digite ou cole o texto da redação...",
)

if st.button("🔍 Avaliar Redação", type="primary"):
    if len(essay.strip()) < 50:
        st.error("A redação deve ter pelo menos 50 caracteres.")
    else:
        with st.spinner("Analisando redação..."):
            # Extrair features
            features = extract_all_features(essay)

            st.subheader("📊 Features Extraídas")
            col1, col2, col3 = st.columns(3)
            col1.metric("Palavras", int(features.get("n_words", 0)))
            col2.metric("Frases", int(features.get("n_sentences", 0)))
            col3.metric("Parágrafos", int(features.get("n_paragraphs", 0)))

            col4, col5, col6 = st.columns(3)
            col4.metric("Diversidade Léxica", f"{features.get('lexical_diversity', 0):.2%}")
            col5.metric("Conectivos", int(features.get("conectivos_total", 0)))
            col6.metric("Palavras de Proposta", int(features.get("proposta_keywords", 0)))

            # Tentar carregar modelo treinado
            import pickle
            model_path = ROOT / "outputs" / "models"
            baseline_files = list(model_path.glob("baseline_*.pkl"))

            if baseline_files:
                st.subheader("📋 Notas Preditas")
                import numpy as np

                for pkl_file in sorted(baseline_files):
                    comp = pkl_file.stem.split("_")[-1]
                    with open(pkl_file, "rb") as f:
                        saved = pickle.load(f)

                    model = saved["model"]
                    scaler = saved["scaler"]

                    feature_values = np.array(
                        [features.get(k, 0) for k in features.keys()]
                    ).reshape(1, -1)

                    try:
                        feature_scaled = scaler.transform(feature_values)
                        pred = model.predict(feature_scaled)[0]
                        # Converter [0,1] → [0,200]
                        nota = int(round(pred * 200 / 40) * 40)
                        nota = max(0, min(200, nota))

                        label = COMPETENCIAS.get(comp.replace("_norm", ""), comp)
                        st.progress(nota / 200, text=f"**{comp}**: {nota}/200 — {label}")
                    except Exception as e:
                        st.warning(f"Erro ao predizer {comp}: {e}")
            else:
                st.info(
                    "⚠️ Nenhum modelo treinado encontrado. "
                    "Execute `python -m src.models.train_baseline` para treinar."
                )

st.markdown("---")
st.caption("TCC — Pós-graduação em Ciência dos Dados e Big Data")
