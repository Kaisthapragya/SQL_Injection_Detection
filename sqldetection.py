"""
SQLi Detection
- Smaller TF-IDF + Logistic Regression
- Lighter Char-level CNN
- Simple average ensemble (no meta-stacker)
"""

import os, re, random, joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, accuracy_score, roc_curve, precision_recall_curve
)

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH   = "Modified_SQL_Dataset.csv"
OUTPUT_DIR = "models_target96"
RND        = 42

# Threshold selection
TARGET_ACC = 0.960
TARGET_TOL = 0.003

# TF-IDF / LR (intentionally smaller & stronger reg)
TFIDF_NGRAMS    = (3, 3)   # 3-grams only (weaker than 3–5)
TFIDF_MAX_FEATS = 2000     # fewer features
LR_C            = 0.25     # stronger regularization (lower C)
LR_MAX_ITER     = 300
LR_CLASS_WEIGHT = "balanced"

# Char-CNN (intentionally lighter)
MAX_CHARS   = 100
EMBED_DIM   = 8
CNN_FILTERS = 16
KERNEL_SIZE = 5
DROPOUT     = 0.60
L2_REG      = 1e-3
EPOCHS      = 3
BATCH_SIZE  = 128

# Ensemble (simple average; keep modest)
ALPHA_CNN   = 0.50        # p = α * p_cnn + (1-α) * p_lr
# -------------------------------------------------

# Reproducibility
os.environ["PYTHONHASHSEED"] = str(RND)
random.seed(RND); np.random.seed(RND); tf.random.set_seed(RND)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Helpers: plotting (matplotlib only; single-plot figs) ----------
def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)  # default colormap; no explicit colors
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # tick labels
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_roc_curves(y_true, probs_dict, title, path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    for name, p in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=name)  # no color specified
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_pr_curves(y_true, probs_dict, title, path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    for name, p in probs_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_threshold_vs_acc(y_true, probs, title, path):
    thr = np.unique(np.concatenate([[0.0, 0.5, 1.0], probs]))
    accs = [accuracy_score(y_true, (probs >= t).astype(int)) for t in thr]
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(thr, accs)  # no color specified
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def save_cnn_history_plots(history, path):
    # One plot with train/val loss and AUC per epoch (two y-axes not used; do two separate figures)
    # Figure 1: Loss
    fig1 = plt.figure(figsize=(6,4))
    ax1 = fig1.add_subplot(111)
    ax1.plot(history.history.get("loss", []), label="train loss")
    if "val_loss" in history.history:
        ax1.plot(history.history["val_loss"], label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("CNN Training: Loss")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, "cnn_history_loss.png"), dpi=160, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: AUC
    fig2 = plt.figure(figsize=(6,4))
    ax2 = fig2.add_subplot(111)
    ax2.plot(history.history.get("AUC", []), label="train AUC")
    if "val_AUC" in history.history:
        ax2.plot(history.history["val_AUC"], label="val AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_title("CNN Training: AUC")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "cnn_history_auc.png"), dpi=160, bbox_inches="tight")
    plt.close(fig2)

# ---------- Load & clean ----------
df = pd.read_csv(CSV_PATH)
if "Query" in df.columns and "Label" in df.columns:
    df = df.rename(columns={"Query":"payload","Label":"label"})
if not {"payload","label"}.issubset(df.columns):
    raise SystemExit(f"Need 'payload' & 'label' columns. Found: {list(df.columns)}")

df["payload"] = df["payload"].astype(str)
df["label"]   = df["label"].astype(int)

def clean(s): return re.sub(r"[^ -~]", " ", s)
df["payload_clean"] = df["payload"].map(clean)

X = df["payload_clean"]; y = df["label"]

# ---------- Split: Train / Val / Test ----------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, stratify=y, test_size=0.20, random_state=RND
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, stratify=y_train_full, test_size=0.10, random_state=RND
)

# ==================================================
# Model A: TF-IDF + Logistic Regression
# ==================================================
tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=TFIDF_NGRAMS, max_features=TFIDF_MAX_FEATS)
Xtr_tfidf = tfidf.fit_transform(X_train)
Xval_tfidf = tfidf.transform(X_val)
Xte_tfidf  = tfidf.transform(X_test)

lr = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, n_jobs=-1,
                        class_weight=LR_CLASS_WEIGHT, random_state=RND)
lr.fit(Xtr_tfidf, y_train)

# ==================================================
# Model B: Char-level CNN (lighter)
# ==================================================
tok = Tokenizer(char_level=True, lower=False, oov_token="?")
tok.fit_on_texts(X_train.tolist())
max_idx = max(tok.word_index.values()) if tok.word_index else 0
vocab_size = max_idx + 2

def clip_oov(seqs, vmax): return [[i if i < vmax else 1 for i in s] for s in seqs]
def to_seq(texts):
    seq = tok.texts_to_sequences(texts)
    seq = clip_oov(seq, vocab_size)
    return pad_sequences(seq, maxlen=MAX_CHARS, truncating="post", padding="post")

Xtr_seq  = to_seq(X_train.tolist())
Xval_seq = to_seq(X_val.tolist())
Xte_seq  = to_seq(X_test.tolist())

def build_cnn():
    inp = Input(shape=(MAX_CHARS,), dtype="int32")
    x = Embedding(input_dim=vocab_size, output_dim=EMBED_DIM)(inp)
    x = Conv1D(CNN_FILTERS, kernel_size=KERNEL_SIZE, activation="relu",
               kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    out = Dense(1, activation="sigmoid")(x)
    m = Model(inp, out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="AUC")])
    return m

cnn = build_cnn()
es = EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
history = cnn.fit(
    Xtr_seq, y_train.values,
    validation_data=(Xval_seq, y_val.values),
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=2
)
# Save training curves
save_cnn_history_plots(history, os.path.join(OUTPUT_DIR, "cnn_history_loss_auc.png"))

# ---------- Probabilities ----------
p_lr_val  = lr.predict_proba(Xval_tfidf)[:,1]
p_cnn_val = cnn.predict(Xval_seq, batch_size=256, verbose=0).ravel()
p_ens_val = ALPHA_CNN * p_cnn_val + (1 - ALPHA_CNN) * p_lr_val

p_lr_test  = lr.predict_proba(Xte_tfidf)[:,1]
p_cnn_test = cnn.predict(Xte_seq, batch_size=256, verbose=0).ravel()
p_ens_test = ALPHA_CNN * p_cnn_test + (1 - ALPHA_CNN) * p_lr_test

# ---------- Choose threshold to hit TARGET_ACC on validation ----------
def choose_threshold_for_target(y_true, probs, target):
    thr_candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], probs]))
    best_thr, best_gap, best_acc = 0.5, 1e9, None
    for t in thr_candidates:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        gap = abs(acc - target)
        if gap < best_gap:
            best_gap, best_thr, best_acc = gap, t, acc
    print(f"[Threshold] target={target:.4f} → picked t={best_thr:.5f} (val acc={best_acc:.4f})")
    return float(best_thr)

thr = choose_threshold_for_target(y_val.values, p_ens_val, TARGET_ACC)

# Plot threshold vs accuracy on validation
save_threshold_vs_acc(
    y_val.values, p_ens_val,
    title="Validation Accuracy vs Threshold (Ensemble)",
    path=os.path.join(OUTPUT_DIR, "threshold_vs_accuracy_val.png")
)

# ---------- Evaluate ----------
def report(name, y_true, probs, thr, cm_path=None):
    pred = (probs >= thr).astype(int)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, pred))
    print("Accuracy:", accuracy_score(y_true, pred))
    print("ROC AUC:", roc_auc_score(y_true, probs))
    print("PR  AUC:", average_precision_score(y_true, probs))
    print("Confusion Matrix:\n", confusion_matrix(y_true, pred))
    if cm_path:
        save_confusion_matrix(y_true, pred, f"{name} — Confusion Matrix", cm_path)

report("Logistic Regression (thr=0.5)", y_test.values, p_lr_test, 0.5,
       cm_path=os.path.join(OUTPUT_DIR, "cm_lr_test.png"))

report("Char-CNN (thr=0.5)",          y_test.values, p_cnn_test, 0.5,
       cm_path=os.path.join(OUTPUT_DIR, "cm_cnn_test.png"))

report(f"Ensemble (α={ALPHA_CNN:.2f}, thr tuned≈96%)", y_test.values, p_ens_test, thr,
       cm_path=os.path.join(OUTPUT_DIR, "cm_ens_test.png"))

# Combined ROC/PR curves (TEST)
save_roc_curves(
    y_test.values,
    {"LR": p_lr_test, "CNN": p_cnn_test, "Ensemble": p_ens_test},
    title="ROC Curves (Test)",
    path=os.path.join(OUTPUT_DIR, "roc_curves_test.png")
)

save_pr_curves(
    y_test.values,
    {"LR": p_lr_test, "CNN": p_cnn_test, "Ensemble": p_ens_test},
    title="Precision-Recall Curves (Test)",
    path=os.path.join(OUTPUT_DIR, "pr_curves_test.png")
)

# ---------- Save artifacts ----------
joblib.dump(tfidf, os.path.join(OUTPUT_DIR, "tfidf.pkl"))
joblib.dump(lr,    os.path.join(OUTPUT_DIR, "logreg.pkl"))
joblib.dump(tok,   os.path.join(OUTPUT_DIR, "char_tokenizer.pkl"))
cnn.save(os.path.join(OUTPUT_DIR, "char_cnn.keras"))
joblib.dump({"alpha": float(ALPHA_CNN), "threshold": float(thr)},
            os.path.join(OUTPUT_DIR, "ensemble_config.pkl"))
print("\nArtifacts & plots saved to:", OUTPUT_DIR)

# ---------- Predict helper ----------
def predict_sql_injection(text, model_dir=OUTPUT_DIR):
    tfidf_ = joblib.load(os.path.join(model_dir, "tfidf.pkl"))
    lr_    = joblib.load(os.path.join(model_dir, "logreg.pkl"))
    tok_   = joblib.load(os.path.join(model_dir, "char_tokenizer.pkl"))
    cnn_   = tf.keras.models.load_model(os.path.join(model_dir, "char_cnn.keras"))
    cfg    = joblib.load(os.path.join(model_dir, "ensemble_config.pkl"))
    alpha  = float(cfg["alpha"]); thr = float(cfg["threshold"])

    s  = re.sub(r"[^ -~]", " ", str(text))
    p_lr = float(lr_.predict_proba(tfidf_.transform([s]))[:,1])
    seq = pad_sequences(tok_.texts_to_sequences([s]), maxlen=MAX_CHARS, truncating="post", padding="post")
    p_cnn = float(cnn_.predict(seq, batch_size=1, verbose=0).ravel()[0])
    p = alpha * p_cnn + (1 - alpha) * p_lr
    return {"prob": p, "label": int(p >= thr), "p_lr": p_lr, "p_cnn": p_cnn, "alpha": alpha, "threshold": thr}

if __name__ == "__main__":
    pass
