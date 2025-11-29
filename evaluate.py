import streamlit as st
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, top_k_accuracy_score, classification_report, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import re
import io
import base64
import json
from PyPDF2 import PdfReader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ========================= MODEL ARCHITECTURES =========================

class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_dropout=0.3):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, lstm_out):
        attn_scores = self.attn(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, attn_weights

class BiLSTMClassifier(nn.Module):
    """BiLSTM + Attention"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, padding_idx=0,
                 lstm_layers=2, embed_dropout=0.2, fc_dropout1=0.5, fc_dropout2=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=lstm_layers, dropout=0.3,
            batch_first=True, bidirectional=True
        )
        self.attention = Attention(hidden_dim, attn_dropout=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, 192)
        self.dropout1 = nn.Dropout(fc_dropout1)
        self.bn2 = nn.BatchNorm1d(192)
        self.fc2 = nn.Linear(192, 96)
        self.dropout2 = nn.Dropout(fc_dropout2)
        self.fc3 = nn.Linear(96, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)
        lstm_out, _ = self.lstm(emb)
        context, attn_weights = self.attention(lstm_out)
        x = self.bn1(context)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.bn2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class HybridBiLSTM_CNN_NoAttention(nn.Module):
    """BiLSTM + CNN (No Attention)"""
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
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        # BiLSTM
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=lstm_layers, batch_first=True,
            dropout=0.3, bidirectional=True
        )
        # CNN Blocks with multiple kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_dim * 2, out_channels=cnn_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        # Output dims
        lstm_vec_dim = hidden_dim * 2  # last hidden state
        cnn_vec_dim = cnn_filters * len(kernel_sizes)
        fusion_dim = lstm_vec_dim + cnn_vec_dim
        # Fully Connected Layers
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)
        # LSTM
        lstm_out, (h_n, _) = self.lstm(emb)
        # Take final hidden state of both directions
        lstm_vec = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch, hidden_dim*2)
        # CNN
        cnn_input = lstm_out.permute(0, 2, 1)  # (batch, channels, seq_len)
        cnn_feats = [
            torch.max(F.relu(conv(cnn_input)), dim=2)[0]
            for conv in self.convs
        ]
        cnn_vec = torch.cat(cnn_feats, dim=1)  # (batch, filters*num_kernels)
        # Concatenate LSTM + CNN
        fused = torch.cat([lstm_vec, cnn_vec], dim=1)
        # Classification
        x = self.relu(self.fc1(fused))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AttentionModel3(nn.Module):
    """Attention layer for Model 3 (no dropout, returns context only)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        attn_scores = self.attn(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context

class HybridBiLSTM_CNN(nn.Module):
    """BiLSTM + CNN + Attention (Hybrid Model)"""
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
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        # BiLSTM
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=lstm_layers, dropout=0.3,
            batch_first=True, bidirectional=True
        )
        # Attention (uses separate class to match state_dict keys: attention.attn.weight)
        self.attention = AttentionModel3(hidden_dim)
        
        # CNN Blocks with multiple kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_dim * 2, out_channels=cnn_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        # Output dims
        lstm_vec_dim = hidden_dim * 2
        cnn_vec_dim = cnn_filters * len(kernel_sizes)
        fusion_dim = lstm_vec_dim + cnn_vec_dim
        # Fully Connected Layers
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Embedding
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)
        # LSTM
        lstm_out, _ = self.lstm(emb)  # (batch, seq_len, hidden*2)
        
        # Attention
        lstm_vec = self.attention(lstm_out)  # (batch, hidden*2)
        
        # CNN
        cnn_input = lstm_out.permute(0, 2, 1)  # (batch, channels, seq_len)
        cnn_feats = [
            torch.max(F.relu(conv(cnn_input)), dim=2)[0]
            for conv in self.convs
        ]
        cnn_vec = torch.cat(cnn_feats, dim=1)  # (batch, filters*num_kernels)
        # Fusion: LSTM + CNN
        fused = torch.cat([lstm_vec, cnn_vec], dim=1)
        # Fully Connected
        x = self.relu(self.fc1(fused))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ========================= LOAD RESOURCES =========================
BASE_PATH = "/home/izen-abbas/venv/ResuSight/NLP_Project"

@st.cache_resource
def load_resources_v2():
    word2idx = pickle.load(open(f"{BASE_PATH}/word2idx.pkl", "rb"))
    idx2word = pickle.load(open(f"{BASE_PATH}/idx2word.pkl", "rb"))
    le = pickle.load(open(f"{BASE_PATH}/le.pkl", "rb"))
    tfidf = pickle.load(open(f"{BASE_PATH}/tfidf.pkl", "rb"))
    
    # Model hyperparameters (must match training scripts)
    vocab_size = len(word2idx)
    num_classes = len(le.classes_)
    embed_dim = 128
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DL Models (3 BiLSTM models from .pt files)
    # Model 1: BiLSTM + Attention
    model1 = BiLSTMClassifier(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model1.load_state_dict(torch.load(f"{BASE_PATH}/BiLSTM+Attention.pt", map_location=device))
    model1 = model1.to(device)
    model1.eval()
    
    # Model 2: BiLSTM + CNN (No Attention)
    model2 = HybridBiLSTM_CNN_NoAttention(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model2.load_state_dict(torch.load(f"{BASE_PATH}/BiLSTM+CNN.pt", map_location=device))
    model2 = model2.to(device)
    model2.eval()
    
    # Model 3: BiLSTM + CNN + Attention (Hybrid)
    model3 = HybridBiLSTM_CNN(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model3.load_state_dict(torch.load(f"{BASE_PATH}/BiLSTM+CNN+Attention.pt", map_location=device))
    model3 = model3.to(device)
    model3.eval()

    # Load ML Models (3 sklearn models from .pkl files)
    clf1 = pickle.load(open(f"{BASE_PATH}/clf1.pkl", "rb"))
    clf2 = pickle.load(open(f"{BASE_PATH}/clf2.pkl", "rb"))
    clf3 = pickle.load(open(f"{BASE_PATH}/clf3_rf.pkl", "rb"))

    
    
    # Load History (DL Models)
    dl_history = {}
    try:
        dl_history["BiLSTM+Attention"] = json.load(open(f"{BASE_PATH}/history_model1.json", "r"))
        dl_history["BiLSTM+CNN"] = json.load(open(f"{BASE_PATH}/history_model2.json", "r"))
        dl_history["BiLSTM+CNN+Attention"] = json.load(open(f"{BASE_PATH}/history_model3.json", "r"))
    except FileNotFoundError:
        pass
    
    # Load History (ML Models) - No history for ML models
    ml_history = {}
    
    # Load History (Transformer Model)
    transformer_history = {}
    try:
        transformer_history["Transformer (DistilBERT)"] = json.load(open(f"{BASE_PATH}/transformer_model/transformer_history.json", "r"))
    except FileNotFoundError:
        pass

    # Load transformer tokenizer + model if available
    transformer_tokenizer = None
    transformer_model = None
    transformer_path = os.path.join(BASE_PATH, "transformer_model")
    try:
        transformer_tokenizer = DistilBertTokenizerFast.from_pretrained(transformer_path)
        transformer_model = DistilBertForSequenceClassification.from_pretrained(transformer_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        transformer_model.eval()
    except Exception as e:
        # ignore if transformer not present
        transformer_tokenizer = None
        transformer_model = None

    return word2idx, idx2word, le, tfidf, model1, model2, model3, clf1, clf2, clf3, dl_history, ml_history, transformer_history, transformer_tokenizer, transformer_model



# unpack
word2idx, idx2word, le, tfidf, model1, model2, model3, clf1, clf2, clf3, dl_history, ml_history, transformer_history, tokenizer_trans, transformer_model = load_resources_v2()

MAX_LEN = 500
EMBED_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================= CLEANER =========================
def clean_resume(text):
    if not text:
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


def text_to_seq(text):
    seq = [word2idx.get(w, 1) for w in text.split()]
    if len(seq) < MAX_LEN:
        seq += [0] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    return torch.tensor([seq], dtype=torch.long).to(device)

# ========================= DATASET & DATALOADER =========================
class ResumeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

@st.cache_resource
def load_test_data():
    df = pd.read_csv("/home/izen-abbas/venv/LSTMs/Final_Categorized.csv")
    df["Resume"] = df["Resume"].apply(clean_resume)
    
    # Re-encode labels to ensure consistency
    df["label"] = le.transform(df["Category"])
    
    X = df["Resume"].values
    y = df["label"].values
    
    # Replicate split
    x_train_text_full, x_test_text, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Convert text to seq for DL models
    x_test_seq = np.array([
        [word2idx.get(w, 1) for w in t.split()][:MAX_LEN] + [0]*max(0, MAX_LEN-len(t.split())) 
        for t in x_test_text
    ])
    x_test_seq = np.array([s[:MAX_LEN] for s in x_test_seq])
    
    test_dataset = ResumeDataset(x_test_seq, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Convert text to TF-IDF for ML models
    x_test_tfidf = tfidf.transform(x_test_text)
    
    return test_loader, x_test_tfidf, y_test, x_test_text

# ========================= MODEL DEFINITIONS =========================
# (DL models code unchanged - omitted here for brevity in the document but present in the actual app file)
# ... (keep your existing BiLSTM/CNN classes and model loading code) - already present above in original file

# ========================= TRANSFORMER PREDICT / EVAL HELPERS =========================

def transformer_predict(text, tokenizer, model):
    if tokenizer is None or model is None:
        return None
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    return probs



# ========================= LOAD DL MODELS =========================
# (loading DL models same as your original app - ensure these lines are present in the file)
vocab_size_actual = len(word2idx)
num_classes = len(le.classes_)

# instantiate DL models and load state_dicts (same as original file)
# ... (omitted here for brevity in canvas)

# ========================= STREAMLIT APP (modified to include transformer) =========================
st.title("Resume Job Category Prediction")

# ========================= MODEL WRAPPERS FOR EVALUATION =========================

# ---- Machine Learning models dictionary ----
ml_models = {
    "Logistic Regression": clf1,
    "Linear SVM": clf2,
    "Random Forest": clf3
}

# ---- Deep Learning models dictionary ----
dl_models = {
    "BiLSTM+Attention": model1,
    "BiLSTM+CNN": model2,
    "BiLSTM+CNN+Attention": model3
}

# ---- Metrics calculator ----
def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    top3 = top_k_accuracy_score(y_true, y_prob, k=3, labels=list(range(len(le.classes_))))
    top5 = top_k_accuracy_score(y_true, y_prob, k=5, labels=list(range(len(le.classes_))))
    cm = confusion_matrix(y_true, y_pred)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Top-3 Accuracy": top3,
        "Top-5 Accuracy": top5,
        "Confusion Matrix": cm
    }

# ---- Evaluate DL models ----
def evaluate_dl_model(model, test_loader, y_test):
    model.eval()
    preds = []
    probs = []

    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            out = model(X)
            p = F.softmax(out, dim=1).cpu().numpy()
            pred = np.argmax(p, axis=1)

            preds.extend(pred)
            probs.extend(p)

    return np.array(preds), np.array(probs)

# ---- Evaluate ML models ----
def evaluate_ml_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    return preds, probs


# --- Sidebar for Navigation ---
page = st.sidebar.selectbox("Choose a Page", ["Prediction", "Model Evaluation"]) 

if page == "Prediction":
    st.header("Upload Resume for Prediction")
    uploaded_file = st.file_uploader("Upload Resume:", type=["txt", "pdf"])
    
    resume_text = ""
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            try:
                resume_text = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                resume_text = uploaded_file.read().decode("latin-1")
            st.subheader("Uploaded Resume (Text)")
            st.text_area("File Content", resume_text, height=400)
        
        elif uploaded_file.type == "application/pdf":
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()
            try:
                pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
                for page in pdf_reader.pages:
                    resume_text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Failed to read PDF file: {e}")

            st.subheader("Uploaded Resume (PDF Viewer)")
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

            st.subheader("Extracted Text")
            st.text_area("Content", resume_text, height=300)

            #printing cleaned text by using re
            

    
    # Default sample if no file
    if not resume_text:
        st.info("Using sample resume for demonstration.")
        resume_text = """Skills * Programming Languages: Python (pandas, numpy, scipy, scikit-learn, matplotlib), R, Sql, Spark, Scala. 
        Machine learning: Deep Learning, CNN, RNN, Transformers, Regression, SVM, Random Forest, Ensemble Methods, NLP, Time Series.
        Databases: MySQL, MongoDB, PowerBI, AWS, GCP.
        Education: MS Data Science, Stanford University.
        Experience: Senior Data Scientist at Tech Innovations."""
        st.text_area("Sample Resume", resume_text, height=200)

    if resume_text:
        cleaned_text = clean_resume(resume_text)
        
        st.subheader("Cleaned Text")
        st.text_area("Cleaned Content", cleaned_text, height=200)
        
        # Prepare inputs
        # DL Input
        seq = text_to_seq(cleaned_text)
        # ML Input
        tfidf_vec = tfidf.transform([cleaned_text])
        
        st.subheader("Predictions")
        
        # --- DL Models ---
        st.markdown("### Deep Learning Models")
        cols = st.columns(len(dl_models))
        for i, (name, model) in enumerate(dl_models.items()):
            with cols[i]:
                with torch.no_grad():
                    logits = model(seq)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                top1_idx = np.argmax(probs)
                top1_label = le.inverse_transform([top1_idx])[0]
                st.info(f"**{name}**\n\n{top1_label}\n({probs[top1_idx]:.4f})")
                
                with st.expander(f"Top-5 ({name})"):
                    top5_idx = probs.argsort()[-5:][::-1]
                    top5_labels = le.inverse_transform(top5_idx)
                    top5_probs = probs[top5_idx]
                    for l, p in zip(top5_labels, top5_probs):
                        st.write(f"{l}: {p:.4f}")

        # --- ML Models ---
        st.markdown("### Machine Learning Models")
        cols_ml = st.columns(len(ml_models))
        for i, (name, model) in enumerate(ml_models.items()):
            with cols_ml[i]:
                probs = model.predict_proba(tfidf_vec)[0]
                top1_idx = np.argmax(probs)
                top1_label = le.inverse_transform([top1_idx])[0]
                st.success(f"**{name}**\n\n{top1_label}\n({probs[top1_idx]:.4f})")
                
                with st.expander(f"Top-5 ({name})"):
                    top5_idx = probs.argsort()[-5:][::-1]
                    top5_labels = le.inverse_transform(top5_idx)
                    top5_probs = probs[top5_idx]
                    for l, p in zip(top5_labels, top5_probs):
                        st.write(f"{l}: {p:.4f}")

        # --- Transformer Model ---
        if tokenizer_trans is not None and transformer_model is not None:
            st.markdown("### Transformer Model")
            probs_t = transformer_predict(cleaned_text, tokenizer_trans, transformer_model)
            if probs_t is not None:
                top1_idx = np.argmax(probs_t)
                top1_label = le.inverse_transform([top1_idx])[0]
                st.warning(f"**DistilBERT Transformer**\n\n{top1_label}\n({probs_t[top1_idx]:.4f})")
                with st.expander("Top-5 (Transformer)"):
                    top5_idx = probs_t.argsort()[-5:][::-1]
                    top5_labels = le.inverse_transform(top5_idx)
                    top5_probs = probs_t[top5_idx]
                    for l, p in zip(top5_labels, top5_probs):
                        st.write(f"{l}: {p:.4f}")
        else:
            st.info("Transformer model not available in the resources folder.")

elif page == "Model Evaluation":
    st.header("Model Evaluation on Test Set")
    
    # Path to cached evaluation results
    eval_cache_path = f"{BASE_PATH}/evaluation_results.pkl"
    
    # Check if cached results exist
    cache_exists = os.path.exists(eval_cache_path)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        run_eval = st.button("Run Evaluation", disabled=cache_exists and not st.session_state.get('force_rerun', False))
    with col2:
        if cache_exists:
            force_rerun = st.button("Re-run Evaluation (Force)")
            if force_rerun:
                st.session_state['force_rerun'] = True
                st.rerun()
    
    # Load or compute results
    results = None
    
    if run_eval or st.session_state.get('force_rerun', False):
        with st.spinner("Loading test data and evaluating models..."):
            test_loader, x_test_tfidf, y_test, x_test_text = load_test_data()
            
            results = {}
            
            # Evaluate DL Models
            st.info("Evaluating Deep Learning Models...")
            for name, model in dl_models.items():
                y_pred, y_pred_prob = evaluate_dl_model(model, test_loader, y_test)
                results[name] = calculate_metrics(y_test, y_pred, y_pred_prob)
                
            # Evaluate ML Models
            st.info("Evaluating Machine Learning Models...")
            for name, model in ml_models.items():
                y_pred, y_pred_prob = evaluate_ml_model(model, x_test_tfidf, y_test)
                results[name] = calculate_metrics(y_test, y_pred, y_pred_prob)

            # Evaluate Transformer Model
            if tokenizer_trans is not None and transformer_model is not None:
                st.info("Evaluating Transformer Model...")
                transformer_probs = []
                transformer_preds = []
                for txt in x_test_text:
                    p = transformer_predict(txt, tokenizer_trans, transformer_model)
                    transformer_probs.append(p)
                    transformer_preds.append(np.argmax(p))
                transformer_probs = np.array(transformer_probs)
                transformer_preds = np.array(transformer_preds)
                results["Transformer (DistilBERT)"] = calculate_metrics(y_test, transformer_preds, transformer_probs)

            # Save results to cache
            with open(eval_cache_path, 'wb') as f:
                pickle.dump(results, f)
            
            st.success("Evaluation Complete! Results saved to cache.")
            st.session_state['force_rerun'] = False
            
    elif cache_exists:
        # Load from cache
        with st.spinner("Loading cached evaluation results..."):
            with open(eval_cache_path, 'rb') as f:
                results = pickle.load(f)
        st.info("Loaded evaluation results from cache.")
    
    # Display results if available
    if results is not None:
        # Display Metrics
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({
            name: {k: v for k, v in res.items() if k != "Confusion Matrix"}
            for name, res in results.items()
        })
        st.dataframe(metrics_df.style.highlight_max(axis=1))
        
        # Display Confusion Matrices
        st.subheader("Confusion Matrices")
        
        # Create tabs for models
        tabs = st.tabs(list(results.keys()))
        for i, (name, res) in enumerate(results.items()):
            with tabs[i]:
                st.write(f"**{name}**")
                cm = res["Confusion Matrix"]
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                st.pyplot(fig)
        
        # Display Training Graphs for DL Models
        if dl_history:
            st.subheader("Training History - Deep Learning Models")
            for name, h in dl_history.items():
                st.markdown(f"**{name}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots()
                    ax.plot(h['train_loss'], label='Train Loss')
                    ax.plot(h['val_loss'], label='Val Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'{name} - Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                with col2:
                    fig, ax = plt.subplots()
                    ax.plot(h['train_acc'], label='Train Acc')
                    ax.plot(h['val_acc'], label='Val Acc')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.set_title(f'{name} - Accuracy')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        # Display Training Graphs for Transformer Model
        if transformer_history:
            st.subheader("Training History - Transformer Model")
            for name, h in transformer_history.items():
                st.markdown(f"**{name}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots()
                    ax.plot(h['train_loss'], label='Train Loss')
                    ax.plot(h['val_loss'], label='Val Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'{name} - Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                with col2:
                    fig, ax = plt.subplots()
                    ax.plot(h['train_acc'], label='Train Acc')
                    ax.plot(h['val_acc'], label='Val Acc')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.set_title(f'{name} - Accuracy')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        if not dl_history and not ml_history and not transformer_history:
            st.info("Note: Training and validation loss graphs are not available. Run the training scripts to generate history files.")