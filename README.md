# AES-ENEM: Correção Automática de Redações do ENEM

Trabalho de Conclusão de Curso — Pós-graduação em Ciência dos Dados e Big Data.

## Objetivo

Desenvolver um sistema de Automatic Essay Scoring (AES) para redações do ENEM,
predizendo notas nas 5 competências utilizando técnicas de NLP e Deep Learning.

## Estrutura do Projeto

```
TCC/
├── configs/             # Configurações e hiperparâmetros
├── data/
│   ├── raw/             # Dados brutos (Essay-BR, etc.)
│   └── processed/       # Dados pré-processados
├── notebooks/           # Análise exploratória e experimentos
├── outputs/
│   ├── models/          # Modelos treinados
│   ├── figures/         # Gráficos e visualizações
│   └── reports/         # Relatórios de avaliação
├── src/
│   ├── data/            # Coleta e pré-processamento
│   ├── features/        # Engenharia de features
│   ├── models/          # Treinamento e inferência
│   ├── evaluation/      # Métricas e análise de erros
│   └── app/             # Interface Streamlit
└── requirements.txt
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

## Uso Rápido

```bash
# 1. Baixar e preparar dados
python -m src.data.download_essay_br
python -m src.data.preprocessing

# 2. Extrair features e treinar baseline
python -m src.features.extract_features
python -m src.models.train_baseline

# 3. Fine-tuning BERTimbau
python -m src.models.train_bert

# 4. Avaliar resultados
python -m src.evaluation.evaluate
# Gera:
# - outputs/reports/bert_results.csv
# - outputs/reports/model_comparison_results.csv
# - outputs/figures/baseline_vs_bert_comparison.png
# - outputs/figures/best_baseline_vs_bert.png

# 5. Interface demo
streamlit run src/app/app.py
```

## Dataset

Utiliza o [AES ENEM Dataset](https://huggingface.co/datasets/kamel-usp/aes_enem_dataset)
do HuggingFace (kamel-usp). Configs disponíveis:

- **JBCS2025**: 770 redações com texto de apoio e prompt (padrão)
- **PROPOR2024**: 1155 redações
- **gradesThousand**: 179 redações nota 1000

Cada redação possui notas nas 5 competências (0-200) + nota total.
