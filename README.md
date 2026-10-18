# Sistema NLP de Apoio à Avaliação de Redações do ENEM

Projeto de TCC para apoio à avaliação de redações do ENEM com NLP e aprendizado supervisionado, usando o dataset AES-ENEM e modelagem híbrida (**BERTimbau + XGBoost**).

## Objetivo

Construir um pipeline que receba **tema + redação** e retorne:

- estimativas para as competências **C1, C2, C3, C4 e C5**;
- nota total estimada (0–1000);
- modelo usado em cada competência;
- evidências linguísticas e semânticas para apoio à interpretação humana.

> Este sistema é de apoio e **não substitui avaliação humana**.

## Dataset

Base principal: [AES ENEM Dataset](https://huggingface.co/datasets/kamel-usp/aes_enem_dataset) (configuração **JBCS2025** no notebook).

Cada redação possui:

- texto da redação;
- tema/proposta;
- notas por competência (C1–C5, escala 0–200);
- nota total.

## Pipeline do projeto (resumo)

1. **Carga e validação dos dados** (splits oficiais train/validation/test).
2. **Análise exploratória** (distribuições de notas, tamanho de redação, parágrafos, tokens BERT).
3. **Preprocessing conservador** (normalização sem destruir estrutura textual).
4. **Pipeline NLP base** (parágrafos, sentenças, tokens, lemas, POS, entidades).
5. **Módulos de evidência por competência (C1–C5)**.
6. **Consolidação de features numéricas**.
7. **Baseline XGBoost** (um modelo por competência).
8. **BERTimbau multi-output** (predição conjunta de C1–C5).
9. **Comparação experimental** (MAE, RMSE, QWK e análises complementares).
10. **Inferência híbrida final**:
   - C1 → **BERTimbau**
   - C2 → **XGBoost**
   - C3 → **XGBoost**
   - C4 → **XGBoost**
   - C5 → **XGBoost**
11. **Interface Gradio** para teste com nova redação.

## Estrutura geral do repositório

```text
TCC/
├── configs/
├── data/
├── notebooks/
├── outputs/
├── src/
└── requirements.txt
```

## Requisitos

- Python 3.10+
- Dependências em [requirements.txt](src/requirements.txt)
- Modelo spaCy para português: `pt_core_news_sm`

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

## Execução (via notebook disponível em /notebooks/Laboratorio-TCC.ipynb)

O fluxo completo está documentado no notebook principal (12 blocos, do ambiente até a interface final).

### Principais saídas geradas no experimento

- **Features**: `data/features_train.csv`, `data/features_validation.csv`, `data/features_test.csv`
- **Modelos XGBoost**: `models/xgboost_C1.joblib` ... `models/xgboost_C5.joblib`
- **Modelo BERTimbau**: `models/bertimbau_multioutput.pt`
- **Relatórios**:
  - `reports/xgboost_validation_metrics.csv`
  - `reports/xgboost_test_metrics.csv`
  - `reports/bertimbau_validation_metrics.csv`
  - `reports/bertimbau_test_metrics.csv`
  - `reports/comparacao_xgboost_bertimbau.csv`

## Interface (Gradio)

A interface recebe:

- tema da redação;
- texto da redação.

E exibe:

- tabela com C1–C5 (modelo, predição contínua e nota estimada);
- nota total estimada;
- evidências linguísticas (com destaque para proposta de intervenção em C5).

## Observações metodológicas

- Splits oficiais são preservados (sem reamostragem aleatória para avaliação final).
- Features de treino e inferência usam o mesmo pipeline de extração.
- Evidências NLP são apresentadas como **apoio interpretativo**, não como explicação causal das notas previstas.
