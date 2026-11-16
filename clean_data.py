
import pandas as pd
import os
import re

# Dataset paths
DATASET_PATH = "dataset"
OUTPUT_PATH = "dataset/cleaned"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# LIAR dataset column names
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

files = ["train.tsv", "valid.tsv", "test.tsv"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


for file in files:

    print(f"\nProcessing {file}...")

    file_path = os.path.join(DATASET_PATH, file)

    # Read TSV file
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=columns
    )

    print("Original records:", len(df))

    # Remove missing statements and labels
    df = df.dropna(subset=["statement", "label"])

    # Clean text
    df["statement"] = df["statement"].apply(clean_text)

    # Remove empty statements
    df = df[df["statement"] != ""]

    # Remove exact duplicate rows only
    df = df.drop_duplicates()

    print("Cleaned records:", len(df))

    # Save cleaned CSV
    output_file = file.replace(".tsv", ".csv")

    df.to_csv(
        os.path.join(OUTPUT_PATH, output_file),
        index=False
    )

    print("Saved:", output_file)


print("\n================================")
print("ALL DATASETS CLEANED SUCCESSFULLY!")
print("================================")
