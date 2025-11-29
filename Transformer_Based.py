#!/usr/bin/env python3
"""
transformer_train_for_app.py

Improved transformer training script that saves a HuggingFace-compatible
DistilBERT sequence-classification model into the Streamlit app folder.

By default it will save to:
  /home/izen-abbas/venv/ResuSight/NLP_Project/transformer_model

So your app.py (which calls DistilBertForSequenceClassification.from_pretrained(transformer_path))
will be able to load the model directly.
"""
import os
import argparse
import random
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm

# -------------------------- DEFAULT CONFIG --------------------------
DEFAULT_BACKBONE = "distilbert-base-uncased"
DEFAULT_MAX_LEN = 512
DEFAULT_BATCH = 16
DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 6
WEIGHT_DECAY = 1e-2
WARMUP_PROPORTION = 0.06
SEED = 42
# Path your app expects by default:
DEFAULT_APP_MODEL_DIR = "/home/izen-abbas/venv/ResuSight/NLP_Project/transformer_model"

# -------------------------- UTIL --------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_resume(text):
    import re
    if pd.isna(text):
        return ""
    s = str(text)
    s = re.sub(r'\S+@\S+', ' ', s)
    s = re.sub(r'http\S+', ' ', s)
    replacements = {
        "C++": "CPLUSPLUS", "c++": "CPLUSPLUS",
        "C#": "CSHARP", "c#": "CSHARP",
        ".NET": "DOTNET", ".net": "DOTNET",
        "Node.js": "NODEJS", "node.js": "NODEJS"
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = re.sub(r'[^A-Za-z0-9+\#\./\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    inv = {v: k.lower() for k, v in replacements.items()}
    for k, v in inv.items():
        s = s.replace(k, v)
    return s.lower().strip()

# -------------------------- DATASET --------------------------
class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=DEFAULT_MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        enc = self.tokenizer(
            txt,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = int(self.labels[idx])
        return enc

def collate_fn(batch, tokenizer):
    # tokenizer.pad will convert list[dict] -> batched tensors
    return tokenizer.pad(batch, padding="longest", return_tensors="pt")

# -------------------------- TRAIN / EVAL HELPERS --------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, epoch, class_weights_tensor=None, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    preds_all = []
    labels_all = []

    pbar = tqdm(dataloader, desc=f"Train E{epoch+1}", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            logits = outputs.logits  # (B, num_labels)
            if class_weights_tensor is not None:
                loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
                loss = loss_fct(logits, labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        preds_all.extend(preds.tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro")
    return avg_loss, acc, f1

def eval_model(model, dataloader, device):
    model.eval()
    preds_all = []
    labels_all = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            losses.append(loss.item() * input_ids.size(0))

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds_all.extend(preds.tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = sum(losses) / len(dataloader.dataset)
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro")
    report = classification_report(labels_all, preds_all, digits=4)
    return avg_loss, acc, f1, report

# -------------------------- MAIN --------------------------
def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CSV
    df = pd.read_csv(args.data)
    assert "Resume" in df.columns and "Category" in df.columns, "CSV must have Resume and Category columns"

    df["Resume"] = df["Resume"].apply(clean_resume).fillna("").astype(str)

    # Label encode
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Category"].astype(str))
    num_labels = len(le.classes_)
    print(f"Classes: {num_labels}")

    # Tokenizer & model (HuggingFace classification model)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    config = AutoConfig.from_pretrained(args.backbone, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.backbone, config=config)
    model.to(device)

    # Splits
    X = df["Resume"].tolist()
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.val_size, stratify=y_train, random_state=args.seed)

    train_ds = ResumeDataset(X_train, y_train, tokenizer, max_len=args.max_len)
    val_ds = ResumeDataset(X_val, y_val, tokenizer, max_len=args.max_len)
    test_ds = ResumeDataset(X_test, y_test, tokenizer, max_len=args.max_len)

    # Class weights
    class_weights_np = compute_class_weight(class_weight="balanced", classes=np.arange(num_labels), y=y_train)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
    print("Class weights:", class_weights_np.tolist())

    # Sampler (optional)
    if args.use_sampler:
        class_counts = Counter(y_train)
        samples_weight = np.array([1.0 / class_counts[lbl] for lbl in y_train])
        sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=lambda b: collate_fn(b, tokenizer), drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer), drop_last=False)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    # Optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_proportion))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device, scaler, epoch, class_weights_tensor=class_weights, grad_clip=args.grad_clip)
        val_loss, val_acc, val_f1, val_report = eval_model(model, val_loader, device)

        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc); history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1); history["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1}/{args.epochs} | Train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}")

        # Save best (full checkpoint + HF-friendly saved model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "label_classes": list(le.classes_)
            }
            ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pth")
            torch.save(ckpt, ckpt_path)

            # Save HF-compatible model and tokenizer to output dir (so app can load via from_pretrained)
            os.makedirs(args.output_dir, exist_ok=True)
            tokenizer.save_pretrained(args.output_dir)
            # save model using HF API
            model.save_pretrained(args.output_dir)
            # also save state_dict separately
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_state_dict.pt"))

            # label classes for app decoding
            with open(os.path.join(args.output_dir, "label_classes.json"), "w") as f:
                json.dump(list(le.classes_), f)

            print(f"Saved best checkpoint and HF model to {args.output_dir}")

    print(f"Training complete. Best epoch {best_epoch+1} val_loss {best_val_loss:.4f}")

    # Final evaluation on test set (load best if available)
    print("Evaluating on test set...")
    if os.path.exists(os.path.join(args.output_dir, "best_checkpoint.pth")):
        ckpt = torch.load(os.path.join(args.output_dir, "best_checkpoint.pth"), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc, test_f1, test_report = eval_model(model, test_loader, device)
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f} f1 {test_f1:.4f}")
    print("Classification Report (test):\n", test_report)

    # Save history and label_classes
    with open(os.path.join(args.output_dir, "transformer_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.output_dir, "label_classes.json"), "w") as f:
        json.dump(list(le.classes_), f)

# -------------------------- ARGPARSE --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Resume Classifier (save HF-compatible model for app)")
    parser.add_argument("--data", type=str, required=True, help="Path to Final_Categorized.csv")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, help="Transformer backbone (HuggingFace name)")
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN, help="Max token length (<=512)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision (recommended on GPU)")
    parser.add_argument("--use_sampler", action="store_true", help="Use WeightedRandomSampler for imbalanced classes")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of train->val split")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion test split")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_proportion", type=float, default=WARMUP_PROPORTION)
    parser.add_argument("--seed", type=int, default=SEED)
    # default output dir points to the app path so the app can load the saved model directly
    parser.add_argument("--output_dir", type=str, default=DEFAULT_APP_MODEL_DIR, help="Where to save tokenizer & HF model (app will load from here)")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
