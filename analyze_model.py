"""
analyze_model.py  --  Day 4: Model Evaluation, Error Analysis and Explainable AI
Project : zero-rag-project  (Fake News / Misinformation Detector)

Run from the MAIN project folder:
    python analyze_model.py

What this script does (VALIDATION data only, no retraining, no test data):
  1. Loads the saved best model + TF-IDF vectorizer (read-only).
  2. Loads validation features / labels / original statements.
  3. Predicts on validation data and computes accuracy, precision, recall, macro F1.
  4. Saves every misclassified validation statement.
  5. Extracts the top-10 positive and top-10 negative coefficient words per class.
  6. Saves a confusion-matrix image and a readable explainability report.

Outputs (all written to models/):
  misclassified_statements.csv, important_words.csv,
  validation_confusion_matrix.png, explainability_report.txt
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # file-only backend: no GUI window needed
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

# ----------------------------------------------------------------------------
# 1. CONFIGURATION  (paths are resolved relative to THIS file, so they work
#    no matter which directory VS Code's terminal is currently in)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CLEANED_DIR = BASE_DIR / "dataset" / "cleaned"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"

# Inputs (read-only)
MODEL_PATH = MODELS_DIR / "best_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
X_VALID_PATH = PROCESSED_DIR / "X_valid.npz"
Y_VALID_PATH = PROCESSED_DIR / "y_valid.csv"
VALID_CSV_PATH = CLEANED_DIR / "valid.csv"  # source of the original statement text

# Outputs (must never equal the protected input files)
MISCLASSIFIED_PATH = MODELS_DIR / "misclassified_statements.csv"
IMPORTANT_WORDS_PATH = MODELS_DIR / "important_words.csv"
CONFUSION_PNG_PATH = MODELS_DIR / "validation_confusion_matrix.png"
REPORT_PATH = MODELS_DIR / "explainability_report.txt"

TOP_K = 10  # words per direction (positive / negative) per class

# Natural truthfulness order of the six classes (used for plot order and
# "how far off was the prediction" analysis).
ORDINAL_ORDER = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

# If your y_*.csv files store labels as NUMBERS (e.g. from LabelEncoder), put the
# real mapping here, e.g. {0: "barely-true", 1: "false", ...}. It is NOT guessed,
# because a wrong guess would silently mislabel every result.
NUMERIC_LABEL_MAP = None

# Column-name candidates for auto-detection in dataset/cleaned/valid.csv.
# The ORIGINAL statement is preferred over any cleaned/processed variant.
STATEMENT_CANDIDATES = [
    "statement", "original_statement", "text", "claim", "statement_text",
    "clean_statement", "cleaned_statement", "clean_text", "cleaned_text",
]
LABEL_CANDIDATES = ["label", "labels", "truth", "verdict", "target", "class"]


# ----------------------------------------------------------------------------
# 2. LOADING HELPERS
# ----------------------------------------------------------------------------
def require_file(path: Path) -> None:
    """Raise a clear error if an input file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def unwrap_estimator(obj):
    """Return the final estimator if the pickle holds a GridSearchCV/Pipeline."""
    if hasattr(obj, "best_estimator_"):
        obj = obj.best_estimator_
    if hasattr(obj, "steps"):
        obj = obj.steps[-1][1]
    return obj


def load_model_and_vectorizer():
    require_file(MODEL_PATH)
    require_file(VECTORIZER_PATH)
    model = unwrap_estimator(joblib.load(MODEL_PATH))   # read-only load
    vectorizer = joblib.load(VECTORIZER_PATH)           # read-only load

    if not (hasattr(model, "coef_") and hasattr(model, "classes_")):
        raise TypeError(
            f"best_model.pkl holds a {type(model).__name__}, which has no coef_ / "
            "classes_. This script needs a linear model such as LogisticRegression."
        )
    if not hasattr(vectorizer, "get_feature_names_out"):
        raise TypeError("tfidf_vectorizer.pkl does not look like a fitted scikit-learn vectorizer.")
    return model, vectorizer


def load_validation_labels(expected_rows: int) -> np.ndarray:
    """Read y_valid.csv robustly (handles header / headerless / multi-column files)."""
    require_file(Y_VALID_PATH)
    df = pd.read_csv(Y_VALID_PATH)

    # A headerless file would make read_csv swallow the first label as a header.
    if len(df) == expected_rows - 1:
        print("[WARN] y_valid.csv seems to have no header row; re-reading with header=None.")
        df = pd.read_csv(Y_VALID_PATH, header=None)

    if df.shape[1] > 1:
        label_cols = [c for c in df.columns if str(c).strip().lower() in LABEL_CANDIDATES]
        col = label_cols[0] if label_cols else df.columns[0]
        print(f"[WARN] y_valid.csv has {df.shape[1]} columns; using column '{col}'.")
        series = df[col]
    else:
        series = df.iloc[:, 0]

    if series.dtype == object:
        series = series.astype(str).str.strip()
    return series.to_numpy()


def find_statement_column(df: pd.DataFrame):
    """Return (column_name, confident_flag) for the statement text column."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in STATEMENT_CANDIDATES:
        if cand in lowered:
            return lowered[cand], True

    # Fallback: the text-like column with the longest average string length.
    text_cols = [
        c for c in df.columns
        if df[c].dtype == object and str(c).strip().lower() not in LABEL_CANDIDATES
    ]
    if not text_cols:
        raise ValueError(f"No text column found in {VALID_CSV_PATH.name}. Columns: {list(df.columns)}")
    best = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
    return best, False


def load_validation_statements(expected_rows: int):
    require_file(VALID_CSV_PATH)
    df = pd.read_csv(VALID_CSV_PATH)
    col, confident = find_statement_column(df)
    if confident:
        print(f"[OK] Statement column in valid.csv: '{col}'")
    else:
        print(f"[WARN] No standard statement column name found; guessed '{col}' (longest text column). "
              f"Columns available: {list(df.columns)}")
    if len(df) != expected_rows:
        raise ValueError(
            f"valid.csv has {len(df)} rows but X_valid/y_valid have {expected_rows}. "
            "Rows must correspond one-to-one, otherwise statements would be matched to the wrong predictions."
        )
    return df, df[col].astype(str).to_numpy()


# ----------------------------------------------------------------------------
# 3. LABEL / CLASS NAME HANDLING
# ----------------------------------------------------------------------------
def build_class_names(classes) -> dict:
    """Map each entry of model.classes_ to a readable string."""
    names = {}
    for c in classes:
        if isinstance(c, (str, np.str_)):
            names[c] = str(c)
        elif NUMERIC_LABEL_MAP is not None and c in NUMERIC_LABEL_MAP:
            names[c] = str(NUMERIC_LABEL_MAP[c])
        else:
            names[c] = f"class_{c}"
    if any(v.startswith("class_") for v in names.values()):
        print("[WARN] Model classes are numeric and NUMERIC_LABEL_MAP is not set, so they are shown as "
              "'class_<n>'. Set NUMERIC_LABEL_MAP at the top of this file for readable names.")
    return names


def display_order(classes, name_map):
    """Class order for tables/plots: truthfulness order if names match, else model.classes_ order."""
    if set(name_map.values()) == set(ORDINAL_ORDER):
        inverse = {v: k for k, v in name_map.items()}
        return [inverse[n] for n in ORDINAL_ORDER], True
    return list(classes), False


# ----------------------------------------------------------------------------
# 4. ANALYSIS HELPERS
# ----------------------------------------------------------------------------
def top_indices(row: np.ndarray, k: int):
    """Indices of the k largest and k smallest values (dense 1-D coefficient row)."""
    k = min(k, row.size)
    pos = np.argpartition(row, -k)[-k:]
    pos = pos[np.argsort(row[pos])[::-1]]       # largest first
    neg = np.argpartition(row, k - 1)[:k]
    neg = neg[np.argsort(row[neg])]             # most negative first
    return pos, neg


def extract_important_words(model, feature_names, name_map, k):
    """Top-k positive / negative coefficient words per class, mapped via classes_ and feature names."""
    coef = np.asarray(model.coef_)
    classes = model.classes_

    # Binary LR stores a single row (for classes_[1]); expand for completeness.
    if coef.shape[0] == 1 and len(classes) == 2:
        coef = np.vstack([-coef[0], coef[0]])

    if coef.shape[0] != len(classes):
        raise ValueError(f"coef_ has {coef.shape[0]} rows but model has {len(classes)} classes.")
    if coef.shape[1] != len(feature_names):
        raise ValueError(
            f"coef_ has {coef.shape[1]} features but the vectorizer has {len(feature_names)}. "
            "The vectorizer does not match this model."
        )

    rows = []
    for class_idx, cls in enumerate(classes):          # row i of coef_ <-> classes_[i]
        pos, neg = top_indices(coef[class_idx], k)
        for direction, idxs in (("positive", pos), ("negative", neg)):
            for rank, feat_idx in enumerate(idxs, start=1):
                rows.append({
                    "class": name_map[cls],
                    "class_index": class_idx,
                    "direction": direction,
                    "rank": rank,
                    "word": feature_names[feat_idx],
                    "coefficient": float(coef[class_idx, feat_idx]),
                })
    return pd.DataFrame(rows)


def save_confusion_matrix_plot(cm, labels, path, title):
    """Save counts + row-normalised confusion matrices side by side."""
    n = len(labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    panels = ((axes[0], cm, "d", "Counts"),
              (axes[1], cm_norm, ".2f", "Row-normalised (share of each actual class)"))
    for ax, data, fmt, sub in panels:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title(sub)
        thresh = data.max() / 2 if data.max() > 0 else 0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ----------------------------------------------------------------------------
def main() -> None:
    # Safety guard: outputs must never collide with protected inputs.
    protected = {MODEL_PATH, VECTORIZER_PATH}
    outputs = {MISCLASSIFIED_PATH, IMPORTANT_WORDS_PATH, CONFUSION_PNG_PATH, REPORT_PATH}
    assert protected.isdisjoint(outputs), "Output path would overwrite a protected model file."
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("DAY 4 - VALIDATION ANALYSIS  (no retraining, test set untouched)")
    print("=" * 70)

    # ---- Load model, vectorizer, data -------------------------------------
    model, vectorizer = load_model_and_vectorizer()
    feature_names = vectorizer.get_feature_names_out()
    print(f"[OK] Model: {type(model).__name__} | classes_: {list(model.classes_)}")
    print(f"[OK] Vectorizer vocabulary size: {len(feature_names)}")

    require_file(X_VALID_PATH)
    X_valid = sparse.load_npz(X_VALID_PATH)   # stays sparse; never densified
    print(f"[OK] X_valid: shape={X_valid.shape}, nnz={X_valid.nnz}")
    if X_valid.shape[1] != len(feature_names):
        raise ValueError(f"X_valid has {X_valid.shape[1]} features, vectorizer has {len(feature_names)}.")

    n_rows = X_valid.shape[0]
    y_valid = load_validation_labels(n_rows)
    if len(y_valid) != n_rows:
        raise ValueError(f"y_valid has {len(y_valid)} labels but X_valid has {n_rows} rows.")

    valid_df, statements = load_validation_statements(n_rows)

    unknown = set(y_valid) - set(model.classes_)
    if unknown:
        raise ValueError(
            f"y_valid contains labels unknown to the model: {sorted(map(str, unknown))}. "
            f"Model classes: {list(model.classes_)}. Label encoding mismatch?"
        )

    name_map = build_class_names(model.classes_)
    classes = list(model.classes_)
    ordered_classes, is_ordinal = display_order(classes, name_map)
    ordered_names = [name_map[c] for c in ordered_classes]

    # Sanity check: does the label column in valid.csv agree with y_valid.csv?
    label_cols = [c for c in valid_df.columns if str(c).strip().lower() in LABEL_CANDIDATES]
    if label_cols:
        csv_labels = valid_df[label_cols[0]].astype(str).str.strip().to_numpy()
        agree = max(np.mean(csv_labels == y_valid.astype(str)),
                    np.mean(csv_labels == np.array([name_map[v] for v in y_valid])))
        if agree < 0.99:
            print(f"[WARN] Only {agree:.1%} of valid.csv['{label_cols[0]}'] matches y_valid.csv. "
                  "Row alignment between the files may be wrong - check before trusting the results.")
        else:
            print("[OK] valid.csv labels agree with y_valid.csv (row alignment looks correct).")

    # ---- Predict ----------------------------------------------------------
    y_pred = model.predict(X_valid)
    print(f"[OK] Predictions generated for {n_rows} validation statements.")

    # ---- Metrics ----------------------------------------------------------
    acc = accuracy_score(y_valid, y_pred)
    prec_macro = precision_score(y_valid, y_pred, labels=classes, average="macro", zero_division=0)
    rec_macro = recall_score(y_valid, y_pred, labels=classes, average="macro", zero_division=0)
    f1_macro = f1_score(y_valid, y_pred, labels=classes, average="macro", zero_division=0)
    f1_weighted = f1_score(y_valid, y_pred, labels=classes, average="weighted", zero_division=0)

    report_txt = classification_report(
        y_valid, y_pred, labels=ordered_classes, target_names=ordered_names,
        digits=3, zero_division=0,
    )
    p, r, f, support = precision_recall_fscore_support(
        y_valid, y_pred, labels=ordered_classes, zero_division=0)

    true_counts = pd.Series(y_valid).value_counts()
    pred_counts = pd.Series(y_pred).value_counts()
    majority_share = true_counts.max() / n_rows
    chance = 1.0 / len(classes)

    # ---- Misclassified statements ----------------------------------------
    wrong_mask = np.asarray(y_valid != y_pred)
    mis_df = pd.DataFrame({
        "original_statement": statements[wrong_mask],
        "actual_label": [name_map[v] for v in y_valid[wrong_mask]],
        "predicted_label": [name_map[v] for v in y_pred[wrong_mask]],
    })
    mis_df.to_csv(MISCLASSIFIED_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved {len(mis_df)} misclassified statements -> {MISCLASSIFIED_PATH.relative_to(BASE_DIR)}")

    # ---- Coefficient-based important words -------------------------------
    words_df = extract_important_words(model, feature_names, name_map, TOP_K)
    words_df.to_csv(IMPORTANT_WORDS_PATH, index=False, encoding="utf-8")
    print(f"[OK] Saved important words -> {IMPORTANT_WORDS_PATH.relative_to(BASE_DIR)}")

    # ---- Confusion matrix -------------------------------------------------
    cm = confusion_matrix(y_valid, y_pred, labels=ordered_classes)
    save_confusion_matrix_plot(
        cm, ordered_names, CONFUSION_PNG_PATH,
        f"Validation confusion matrix - {type(model).__name__} "
        f"(accuracy {acc:.3f}, macro-F1 {f1_macro:.3f})",
    )
    print(f"[OK] Saved confusion matrix -> {CONFUSION_PNG_PATH.relative_to(BASE_DIR)}")

    # Most frequent confusions (off-diagonal cells)
    confusions = [(cm[i, j], ordered_names[i], ordered_names[j])
                  for i in range(len(ordered_names)) for j in range(len(ordered_names))
                  if i != j and cm[i, j] > 0]
    confusions.sort(reverse=True)

    # Ordinal error distance (only meaningful if class names are the 6 known ones)
    ordinal_lines = []
    if is_ordinal:
        rank = {n: i for i, n in enumerate(ORDINAL_ORDER)}
        dist = np.array([abs(rank[name_map[a]] - rank[name_map[b]])
                         for a, b in zip(y_valid[wrong_mask], y_pred[wrong_mask])])
        if dist.size:
            ordinal_lines.append(f"Mean absolute distance of errors on the 6-step scale: {dist.mean():.2f}")
            for d in sorted(set(dist)):
                ordinal_lines.append(
                    f"  Errors that are {d} step(s) off: {(dist == d).sum()} ({(dist == d).mean():.1%} of errors)")

    # ---- Build readable report -------------------------------------------
    L = []
    bar = "=" * 70
    L += [bar, "EXPLAINABILITY & ERROR ANALYSIS REPORT (VALIDATION DATA)", bar,
          f"Model            : {type(model).__name__}",
          f"Validation rows  : {n_rows}",
          f"Classes          : {', '.join(ordered_names)}", "",
          "NOTE: all numbers below come from the VALIDATION set. The test set was not used.", ""]

    L += [bar, "1. OVERALL METRICS", bar,
          f"Accuracy             : {acc:.4f}",
          f"Macro precision      : {prec_macro:.4f}",
          f"Macro recall         : {rec_macro:.4f}",
          f"Macro F1-score       : {f1_macro:.4f}",
          f"Weighted F1-score    : {f1_weighted:.4f}",
          f"Correct / Incorrect  : {n_rows - len(mis_df)} / {len(mis_df)}", "",
          "Reference points (to judge whether the model learned anything):",
          f"  Uniform random guessing (1/{len(classes)})        : {chance:.4f}",
          f"  Always predict most common validation class   : {majority_share:.4f}", ""]

    L += [bar, "2. CLASSIFICATION REPORT", bar, report_txt]

    L += [bar, "3. LABEL DISTRIBUTION: ACTUAL vs PREDICTED", bar,
          f"{'class':<14}{'actual':>8}{'predicted':>11}"]
    for c in ordered_classes:
        L.append(f"{name_map[c]:<14}{int(true_counts.get(c, 0)):>8}{int(pred_counts.get(c, 0)):>11}")
    L.append("(A class with many actual but very few predicted samples means the model rarely predicts it.)")
    L.append("")

    L += [bar, "4. CONFUSION MATRIX (rows = actual, columns = predicted)", bar]
    cm_df = pd.DataFrame(cm, index=[f"act:{n}" for n in ordered_names],
                         columns=[f"pred:{n[:9]}" for n in ordered_names])
    L += [cm_df.to_string(), "", "Most frequent confusions (actual -> predicted):"]
    for cnt, a, b in confusions[:8]:
        L.append(f"  {a:<12} -> {b:<12} {cnt} times")
    L.append("")
    if ordinal_lines:
        L += ["Error distance on the truthfulness scale:"] + ordinal_lines + [""]

    L += [bar, f"5. TOP {TOP_K} POSITIVE / NEGATIVE WORDS PER CLASS (Logistic Regression coefficients)", bar,
          "Positive = presence of the word pushes the score TOWARD the class.",
          "Negative = presence of the word pushes the score AWAY from the class.", ""]
    for c in ordered_classes:
        sub = words_df[words_df["class"] == name_map[c]]
        pos = sub[sub.direction == "positive"]
        neg = sub[sub.direction == "negative"]
        L.append(f"--- {name_map[c].upper()} ---")
        L.append("  Positive: " + ", ".join(f"{w} ({v:+.2f})" for w, v in zip(pos.word, pos.coefficient)))
        L.append("  Negative: " + ", ".join(f"{w} ({v:+.2f})" for w, v in zip(neg.word, neg.coefficient)))
        L.append("")

    L += [bar, "6. SAMPLE MISCLASSIFIED STATEMENTS (first 15; full list in CSV)", bar]
    for _, row in mis_df.head(15).iterrows():
        text = row["original_statement"]
        text = text if len(text) <= 160 else text[:157] + "..."
        L.append(f"  actual={row['actual_label']:<12} predicted={row['predicted_label']:<12} | {text}")
    L.append("")

    L += [bar, "7. HOW TO INTERPRET THESE RESULTS (limitations)", bar,
          "- Coefficients show ASSOCIATION with a label in the TRAINING data, not whether a word",
          "  makes a claim true or false. Many top words reflect topic, speaker or era artifacts.",
          "- In a multinomial model, coefficients are relative across classes, so a 'negative' word",
          "  for one class is mainly evidence for competing classes, not a standalone signal.",
          "- TF-IDF words are correlated; individual coefficients are unstable. Treat them as",
          "  hypotheses to check, not as proven explanations.",
          "- The model only sees the statement text (no speaker, source or evidence), so a low",
          "  score is expected for fine-grained truthfulness grading.",
          "- Validation data was used for this analysis. If you make changes based on it, the",
          "  validation score becomes optimistic; use the test set only once, for the final report.",
          bar]
    report_text = "\n".join(L)

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"[OK] Saved explainability report -> {REPORT_PATH.relative_to(BASE_DIR)}\n")

    # ---- Terminal output --------------------------------------------------
    print(report_text)
    print("\nDONE. Files written to models/: misclassified_statements.csv, important_words.csv, "
          "validation_confusion_matrix.png, explainability_report.txt")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}\nRun this script from the main project folder after Days 1-3 completed.")
        sys.exit(1)
    except (ValueError, TypeError) as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:  # unexpected problem: show type for easier debugging
        print(f"\n[UNEXPECTED ERROR] {type(exc).__name__}: {exc}")
        sys.exit(1)