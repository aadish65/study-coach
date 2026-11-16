
import pandas as pd

# Define dataset column names
columns = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context"
]

# Load training dataset
train_df = pd.read_csv(
    "dataset/train.tsv",
    sep="\t",
    header=None,
    names=columns,
    quoting=3,
    keep_default_na=False
)

# Load validation dataset
valid_df = pd.read_csv(
    "dataset/valid.tsv",
    sep="\t",
    header=None,
    names=columns,
    quoting=3,
    keep_default_na=False
)

# Load testing dataset
test_df = pd.read_csv(
    "dataset/test.tsv",
    sep="\t",
    header=None,
    names=columns,
    quoting=3,
    keep_default_na=False
)

print("Training dataset:", train_df.shape)
print("Validation dataset:", valid_df.shape)
print("Testing dataset:", test_df.shape)

print("\nFirst five records:")
print(train_df[["label", "statement"]].head())

# Display dataset information
print("\nDataset information:")
train_df.info()

# Check missing values
print("\nMissing values:")
print(train_df.isnull().sum())

# Check label distribution
print("\nLabel distribution:")
print(train_df["label"].value_counts())

# Check duplicate statements
print("\nDuplicate statements:")
print(train_df["statement"].duplicated().sum())

# Check statement length
train_df["text_length"] = train_df["statement"].str.split().str.len()

print("\nStatement length statistics:")
print(train_df["text_length"].describe())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))

sns.countplot(
    data=train_df,
    x="label",
    order=train_df["label"].value_counts().index
)

plt.title("LIAR Dataset - Label Distribution")
plt.xlabel("Truthfulness Category")
plt.ylabel("Number of Statements")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

duplicates = train_df[
    train_df["statement"].duplicated(keep=False)
]

print(duplicates[["label", "statement"]].sort_values("statement"))
print(
    duplicates.groupby("statement")["label"]
    .nunique()
    .sort_values(ascending=False)
)
print("Original rows:", len(train_df))

print(
    "Duplicate text-label pairs:",
    train_df.duplicated(subset=["statement", "label"]).sum()
)

print(
    "Unique statements with conflicting labels:",
    train_df.groupby("statement")["label"].nunique().gt(1).sum()
)
train_clean = train_df.drop_duplicates(
    subset=["statement", "label"]
).copy()

print("Rows before cleaning:", len(train_df))
print("Rows after cleaning:", len(train_clean))
train_texts = set(train_clean["statement"])
valid_texts = set(valid_df["statement"])
test_texts = set(test_df["statement"])

print("Train-Validation overlap:",
      len(train_texts & valid_texts))

print("Train-Test overlap:",
      len(train_texts & test_texts))

print("Validation-Test overlap:",
      len(valid_texts & test_texts))
conflicts = (
    train_clean.groupby("statement")["label"]
    .nunique()
)

print("Conflicting statements:", (conflicts > 1).sum())