# ========================= IMPORTS =========================
import pandas as pd
import numpy as np
import re
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, top_k_accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ========================= CLEANER =========================
def clean_resume(text):
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

# ========================= LOAD DATA =========================
df = pd.read_csv("/home/izen-abbas/venv/LSTMs/Final_Categorized.csv")
df["Resume"] = df["Resume"].apply(clean_resume)

le = LabelEncoder()
df["label"] = le.fit_transform(df["Category"])
num_classes = len(le.classes_)

X = df["Resume"].values
y = df["label"].values

# Split: train / test (final) and within train create train/val for early stopping
x_train_text_full, x_test_text, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
x_train_text, x_val_text, y_train, y_val = train_test_split(
    x_train_text_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
)

# ========================= TOKENIZATION / VOCAB =========================
VOCAB_SIZE = 15000   # reduced vocab to improve generalization
MAX_LEN = 500        # reduced sequence length
EMBED_DIM = 128

def build_vocab(texts, vocab_size=VOCAB_SIZE):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(vocab_size - 2)
    word2idx = {word: idx+2 for idx, (word, _) in enumerate(most_common)}
    word2idx["<PAD>"] = 0
    word2idx["<OOV>"] = 1
    return word2idx

word2idx = build_vocab(x_train_text, vocab_size=VOCAB_SIZE)
# idx2word maps index -> word
idx2word = {idx: word for word, idx in word2idx.items()}

def text_to_seq(text, word2idx, max_len=MAX_LEN):
    seq = [word2idx.get(w, 1) for w in text.split()]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

x_train = np.array([text_to_seq(t, word2idx) for t in x_train_text])
x_val = np.array([text_to_seq(t, word2idx) for t in x_val_text])
x_test = np.array([text_to_seq(t, word2idx) for t in x_test_text])

# ========================= CLASS WEIGHTS (on train set) =========================
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# ========================= DATASET =========================
class ResumeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ResumeDataset(x_train, y_train)
val_dataset = ResumeDataset(x_val, y_val)
test_dataset = ResumeDataset(x_test, y_test)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================= HYBRID MODEL (BiLSTM + CNN) =========================
class HybridBiLSTM_CNN_NoAttention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        padding_idx=0,
        lstm_layers=2,
        cnn_filters=128,
        kernel_sizes=(2,3,4),
        embed_dropout=0.2,
        fc_dropout=0.5
    ):
        super().__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)

        # ----- BiLSTM -----
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # ----- CNN (after LSTM output) -----
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_dim * 2,   # BiLSTM output dim
                out_channels=cnn_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        # ----- FC Layers -----
        # LSTM last hidden state = hidden_dim*2
        lstm_vec_dim = hidden_dim * 2

        # CNN output = cnn_filters * number_of_kernels
        cnn_vec_dim = cnn_filters * len(kernel_sizes)

        fusion_dim = lstm_vec_dim + cnn_vec_dim

        self.fc1 = nn.Linear(fusion_dim, 256)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)

        # ----- LSTM -----
        lstm_out, (h_n, _) = self.lstm(emb)
        # Take final hidden state of both directions → (batch, hidden_dim*2)
        lstm_vec = torch.cat((h_n[-2], h_n[-1]), dim=1)

        # ----- CNN -----
        # CNN expects: (batch, channels, seq_len)
        cnn_input = lstm_out.permute(0, 2, 1)

        cnn_feats = [
            torch.max(F.relu(conv(cnn_input)), dim=2)[0]  # global max pooling
            for conv in self.convs
        ]

        cnn_vec = torch.cat(cnn_feats, dim=1)

        # ----- Concatenate LSTM + CNN -----
        fused = torch.cat([lstm_vec, cnn_vec], dim=1)

        # ----- Classification -----
        x = self.relu(self.fc1(fused))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Use actual vocab size (len(word2idx))
vocab_size_actual = len(word2idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridBiLSTM_CNN_NoAttention(
    vocab_size=vocab_size_actual,
    embed_dim=EMBED_DIM,
    hidden_dim=128,
    num_classes=num_classes,
    cnn_filters=128,
    kernel_sizes=(2,3,4)   # best-performing usually
).to(device)


criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# ========================= OPTIMIZER / SCHEDULER =========================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # L2 regularization
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

# ========================= TRAINING LOOP (with gradient clipping & early stopping) =========================
EPOCHS = 13
best_val_loss = float('inf')
patience = 5
counter = 0

# Record training history
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    all_train_preds = []
    all_train_labels = []

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

        preds = out.softmax(dim=1).detach().cpu().numpy()
        all_train_preds.append(preds)
        all_train_labels.append(yb.cpu().numpy())

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item()
            all_preds.append(out.softmax(dim=1).cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    # normalize losses by number of batches
    val_loss /= max(1, len(val_loader))
    train_loss_epoch = train_loss / max(1, len(train_loader))

    # compute accuracies
    if all_train_labels:
        y_train_pred = np.concatenate([np.argmax(p, axis=1) for p in all_train_preds])
        y_train_true = np.concatenate(all_train_labels)
        train_acc = (y_train_pred == y_train_true).mean()
    else:
        train_acc = 0.0

    y_val_pred = np.concatenate([np.argmax(p, axis=1) for p in all_preds]) if all_preds else np.array([])
    y_val_true = np.concatenate(all_labels) if all_labels else np.array([])
    val_acc = (y_val_pred == y_val_true).mean() if y_val_true.size else 0.0

    print(f"Epoch {epoch+1}: Train Loss={train_loss_epoch:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    history['train_loss'].append(train_loss_epoch)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    # scheduler step (ReduceLROnPlateau uses validation loss)
    scheduler.step(val_loss)

    # Early stopping / save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "BiLSTM+CNN.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# ========================= LOAD BEST MODEL =========================
model.load_state_dict(torch.load("BiLSTM+CNN.pt", map_location=device))

# ========================= EVALUATION ON TEST SET =========================
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        all_preds.append(out.softmax(dim=1).cpu().numpy())
        all_labels.append(yb.cpu().numpy())

y_pred_prob = np.vstack(all_preds)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_flat = np.hstack(all_labels)

acc = accuracy_score(y_test_flat, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test_flat, y_pred, average="weighted")
top3 = top_k_accuracy_score(y_test_flat, y_pred_prob, k=3)
top5 = top_k_accuracy_score(y_test_flat, y_pred_prob, k=5)
cm = confusion_matrix(y_test_flat, y_pred)

print("\n===== TEST METRICS =====")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Top-3 Accuracy: {top3:.4f}")
print(f"Top-5 Accuracy: {top5:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test_flat, y_pred))

# ========================= SAMPLE PREDICTION (WITH TOP-5 OUTPUT) =========================
sample_resume = """Skills * Programming Languages: Python (pandas, numpy, scipy, scikit-learn, matplotlib), R, Sql, Spark, Scala. 
Machine learning: Deep Learning, CNN, RNN, Transformers, Regression, SVM, Random Forest, Ensemble Methods, NLP, Time Series.
Databases: MySQL, MongoDB, PowerBI, AWS, GCP.
Education: MS Data Science, Stanford University.
Experience: Senior Data Scientist at Tech Innovations."""

sample_clean = clean_resume(sample_resume)
sample_seq = torch.tensor([text_to_seq(sample_clean, word2idx)], dtype=torch.long).to(device)

model.eval()
with torch.no_grad():
    pred = model(sample_seq).softmax(dim=1).cpu().numpy()[0]  # shape: (num_classes,)

# Top-1
top1_idx = np.argmax(pred)
top1_label = le.inverse_transform([top1_idx])[0]

# Top-5
top5_idx = pred.argsort()[-5:][::-1]   # highest → lowest
top5_labels = le.inverse_transform(top5_idx)
top5_probs = pred[top5_idx]

print("\n===== SAMPLE RESUME PREDICTION =====")
print("Top-1 Predicted Category:", top1_label)

print("\nTop-5 Predictions:")
for label, prob in zip(top5_labels, top5_probs):
    print(f"{label}: {prob:.4f}")

# ========================= SAVE TOKENIZER & LABEL ENCODER =========================
pickle.dump(word2idx, open("word2idx.pkl", "wb"))
pickle.dump(idx2word, open("idx2word.pkl", "wb"))
pickle.dump(le, open("le.pkl", "wb"))

# ========================= SAVE HISTORY =========================
import json
with open("history_model2.json", "w") as f:
    json.dump(history, f)