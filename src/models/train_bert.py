"""
Fine-tuning do BERTimbau para scoring de redações do ENEM.

Treina um modelo de regressão (ou multi-output) sobre as 5 competências.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_target_cols(df: pd.DataFrame) -> list[str]:
    target_cols = [c for c in ["C1_norm", "C2_norm", "C3_norm", "C4_norm", "C5_norm"] if c in df.columns]
    if len(target_cols) != 5:
        raise ValueError("É necessário ter as 5 competências normalizadas (C1_norm...C5_norm).")
    return target_cols


def get_train_val_split(df: pd.DataFrame, seed: int, test_size: float, min_val_samples: int = 8):
    """
    Gera split de treino/validação robusto para datasets pequenos.

    - Usa todas as 5 competências;
    - Remove linhas com NA em texto/alvos;
    - Garante um tamanho mínimo de validação;
    - Estratifica pela classe da média das competências quando possível.
    """
    target_cols = get_target_cols(df)

    clean_df = df.dropna(subset=["essay_clean", *target_cols]).copy()
    n_samples = len(clean_df)
    if n_samples < 10:
        raise ValueError(f"Poucas amostras válidas para treino BERT: {n_samples}.")

    val_size = max(int(round(n_samples * test_size)), min_val_samples)
    val_size = min(val_size, n_samples - 2)
    val_size = max(2, val_size)

    y_mean = clean_df[target_cols].mean(axis=1).values
    bins = np.linspace(0, 1, 7)  # n_classes=6 -> 7 pontos de corte
    stratify_labels = np.digitize(y_mean, bins[1:-1])
    class_counts = pd.Series(stratify_labels).value_counts()
    can_stratify = (class_counts >= 2).all() and len(class_counts) > 1

    train_df, val_df = train_test_split(
        clean_df,
        test_size=val_size,
        random_state=seed,
        stratify=stratify_labels if can_stratify else None,
    )

    train_texts = train_df["essay_clean"].astype(str).tolist()
    val_texts = val_df["essay_clean"].astype(str).tolist()
    train_labels = train_df[target_cols].values.astype(np.float32)
    val_labels = val_df[target_cols].values.astype(np.float32)

    return train_texts, val_texts, train_labels, val_labels, target_cols


class EssayDataset(Dataset):
    """Dataset de redações para fine-tuning."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class MultiOutputRegressor(torch.nn.Module):
    """BERTimbau com cabeça de regressão multi-output (5 competências)."""

    def __init__(self, model_name: str, num_outputs: int = 5):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.bert.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_outputs),
            torch.nn.Sigmoid(),  # Output [0, 1] → escalar para [0, 200]
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # [CLS] token
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.regressor(pooled)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def train_bert():
    config = load_config()
    bert_cfg = config["bert"]
    seed = config["project"]["seed"]

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    # Carregar dados
    data_path = ROOT / "data" / "processed" / "essays_processed.csv"
    df = pd.read_csv(data_path)

    train_texts, val_texts, train_labels, val_labels, target_cols = get_train_val_split(
        df=df,
        seed=seed,
        test_size=config["project"]["test_size"],
    )
    num_outputs = len(target_cols)
    print(f"[INFO] Split treino/val: {len(train_texts)} / {len(val_texts)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_cfg["model_name"])

    train_dataset = EssayDataset(train_texts, train_labels, tokenizer, bert_cfg["max_length"])
    val_dataset = EssayDataset(val_texts, val_labels, tokenizer, bert_cfg["max_length"])

    # Modelo
    model = MultiOutputRegressor(bert_cfg["model_name"], num_outputs=num_outputs)
    model.to(device)

    # Training
    output_dir = ROOT / "outputs" / "models" / "bertimbau"
    steps_per_epoch = int(np.ceil(len(train_dataset) / bert_cfg["batch_size"]))
    total_steps = max(1, steps_per_epoch * int(bert_cfg["epochs"]))
    warmup_steps = int(total_steps * bert_cfg["warmup_ratio"])
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=bert_cfg["epochs"],
        per_device_train_batch_size=bert_cfg["batch_size"],
        per_device_eval_batch_size=bert_cfg["batch_size"],
        warmup_steps=warmup_steps,
        weight_decay=bert_cfg["weight_decay"],
        learning_rate=bert_cfg["learning_rate"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("[INFO] Iniciando fine-tuning do BERTimbau...")
    trainer.train()

    # Salvar
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))
    metadata_path = output_dir / "best" / "metadata.json"
    metadata = {
        "target_cols": target_cols,
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "seed": seed,
        "test_size": config["project"]["test_size"],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)
    print(f"[INFO] Modelo salvo em {output_dir / 'best'}")


def main():
    train_bert()


if __name__ == "__main__":
    main()
