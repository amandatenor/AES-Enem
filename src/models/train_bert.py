"""
Fine-tuning do BERTimbau para scoring de redações do ENEM.

Treina um modelo de regressão (ou multi-output) sobre as 5 competências.
"""

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

    # Verificar colunas-alvo
    target_cols = [c for c in ["C1_norm", "C2_norm", "C3_norm", "C4_norm", "C5_norm"]
                   if c in df.columns]
    num_outputs = len(target_cols) if target_cols else 1

    if not target_cols:
        print("[WARN] Usando nota total normalizada.")
        if "total" in df.columns:
            df["total_norm"] = df["total"] / 1000.0
            target_cols = ["total_norm"]
        else:
            raise ValueError("Nenhuma coluna-alvo encontrada.")

    texts = df["essay_clean"].astype(str).tolist()
    labels = df[target_cols].values.astype(np.float32)

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=config["project"]["test_size"], random_state=seed
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_cfg["model_name"])

    train_dataset = EssayDataset(train_texts, train_labels, tokenizer, bert_cfg["max_length"])
    val_dataset = EssayDataset(val_texts, val_labels, tokenizer, bert_cfg["max_length"])

    # Modelo
    model = MultiOutputRegressor(bert_cfg["model_name"], num_outputs=num_outputs)
    model.to(device)

    # Training
    output_dir = ROOT / "outputs" / "models" / "bertimbau"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=bert_cfg["epochs"],
        per_device_train_batch_size=bert_cfg["batch_size"],
        per_device_eval_batch_size=bert_cfg["batch_size"],
        warmup_ratio=bert_cfg["warmup_ratio"],
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
    print(f"[INFO] Modelo salvo em {output_dir / 'best'}")


def main():
    train_bert()


if __name__ == "__main__":
    main()
