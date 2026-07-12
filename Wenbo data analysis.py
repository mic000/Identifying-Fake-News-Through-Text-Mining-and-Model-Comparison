import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WELFake_Dataset.csv"

print("Data path:", DATA_PATH)
print("File exists:", DATA_PATH.exists())

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)

# 2. Check dataset structure
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# 3. Standardize column names
df.columns = df.columns.str.lower().str.strip()

print("\nColumns after lowercasing:")
print(df.columns.tolist())

# 4. Check labels
print("\nLabel counts:")
print(df["label"].value_counts())

print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True) * 100)

# 5. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 6. Check duplicated rows
print("\nDuplicated rows:")
print(df.duplicated().sum())

# 7. Check text length statistics
df["title_length"] = df["title"].fillna("").astype(str).apply(len)
df["text_length"] = df["text"].fillna("").astype(str).apply(len)

print("\nTitle length statistics:")
print(df["title_length"].describe())

print("\nText length statistics:")
print(df["text_length"].describe())

print("\nAverage title length by label:")
print(df.groupby("label")["title_length"].mean())

print("\nAverage text length by label:")
print(df.groupby("label")["text_length"].mean())

# ============================================================
# Progress Report EDA Outputs
# ============================================================

import matplotlib.pyplot as plt

# Create an output folder automatically
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove the serial-number column from analysis
eda_df = df.drop(columns=["unnamed: 0"], errors="ignore").copy()

# Prepare text fields without modifying the original CSV
eda_df["title_clean"] = (
    eda_df["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

eda_df["text_clean"] = (
    eda_df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Word counts are easier to interpret in a report than character counts
eda_df["title_word_count"] = (
    eda_df["title_clean"]
    .str.split()
    .str.len()
)

eda_df["text_word_count"] = (
    eda_df["text_clean"]
    .str.split()
    .str.len()
)

# ------------------------------------------------------------
# 1. More complete data-quality checks
# ------------------------------------------------------------

missing_title = int(df["title"].isna().sum())
missing_text = int(df["text"].isna().sum())
missing_label = int(df["label"].isna().sum())

missing_either_text_field = int(
    (df["title"].isna() | df["text"].isna()).sum()
)

missing_both_text_fields = int(
    (df["title"].isna() & df["text"].isna()).sum()
)

blank_title = int(eda_df["title_clean"].eq("").sum())
blank_text = int(eda_df["text_clean"].eq("").sum())

# Check duplicate article content without using the serial-number column
duplicate_title_text_label = int(
    eda_df.duplicated(
        subset=["title_clean", "text_clean", "label"],
        keep="first"
    ).sum()
)

# Check duplicate articles even if the labels are different
duplicate_title_text = int(
    eda_df.duplicated(
        subset=["title_clean", "text_clean"],
        keep="first"
    ).sum()
)

data_quality_summary = pd.DataFrame(
    {
        "Measure": [
            "Total records",
            "Total usable columns",
            "Missing titles",
            "Missing body texts",
            "Missing labels",
            "Rows missing title or body",
            "Rows missing both title and body",
            "Blank titles after stripping whitespace",
            "Blank body texts after stripping whitespace",
            "Duplicate title-text-label records",
            "Duplicate title-text articles ignoring label",
        ],
        "Value": [
            len(eda_df),
            3,
            missing_title,
            missing_text,
            missing_label,
            missing_either_text_field,
            missing_both_text_fields,
            blank_title,
            blank_text,
            duplicate_title_text_label,
            duplicate_title_text,
        ],
    }
)

print("\nData Quality Summary:")
print(data_quality_summary.to_string(index=False))

data_quality_summary.to_csv(
    OUTPUT_DIR / "data_quality_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 2. Length summary by label
# ------------------------------------------------------------

length_summary = (
    eda_df.groupby("label")
    .agg(
        record_count=("label", "size"),
        mean_title_words=("title_word_count", "mean"),
        median_title_words=("title_word_count", "median"),
        mean_text_words=("text_word_count", "mean"),
        median_text_words=("text_word_count", "median"),
    )
    .round(2)
)

print("\nLength Summary by Label:")
print(length_summary)

length_summary.to_csv(
    OUTPUT_DIR / "length_summary_by_label.csv"
)

# ------------------------------------------------------------
# 3. Class-distribution figure
# ------------------------------------------------------------

label_counts = eda_df["label"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(6, 4))
label_counts.plot(kind="bar", ax=ax)

ax.set_title("Class Distribution in the WELFake Dataset")
ax.set_xlabel("Observed Label")
ax.set_ylabel("Number of Articles")
ax.set_xticklabels(
    [f"Label {label}" for label in label_counts.index],
    rotation=0
)

ax.set_ylim(0, label_counts.max() * 1.17)

for position, count in enumerate(label_counts):
    percentage = count / len(eda_df) * 100
    ax.text(
        position,
        count + label_counts.max() * 0.02,
        f"{count:,}\n({percentage:.2f}%)",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "01_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# 4. Missing-values figure
# ------------------------------------------------------------

missing_counts = df[["title", "text", "label"]].isna().sum()

fig, ax = plt.subplots(figsize=(6, 4))
missing_counts.plot(kind="bar", ax=ax)

ax.set_title("Missing Values by Field")
ax.set_xlabel("Field")
ax.set_ylabel("Number of Missing Records")
ax.set_xticklabels(
    ["Title", "Body Text", "Label"],
    rotation=0
)

ax.set_ylim(0, max(missing_counts.max() * 1.20, 1))

for position, count in enumerate(missing_counts):
    ax.text(
        position,
        count + max(missing_counts.max() * 0.03, 2),
        f"{count:,}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "02_missing_values.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# 5. Title-length figure
# Use the 99th percentile only for visualization so that a few
# extreme values do not compress the entire boxplot.
# ------------------------------------------------------------

title_99th = eda_df["title_word_count"].quantile(0.99)

title_label_0 = (
    eda_df.loc[eda_df["label"] == 0, "title_word_count"]
    .clip(upper=title_99th)
)

title_label_1 = (
    eda_df.loc[eda_df["label"] == 1, "title_word_count"]
    .clip(upper=title_99th)
)

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot(
    [title_label_0, title_label_1],
    showfliers=False
)

ax.set_title("Title Word Count by Label")
ax.set_xlabel("Observed Label")
ax.set_ylabel("Number of Words in Title")
ax.set_xticklabels(["Label 0", "Label 1"])

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "03_title_word_count_by_label.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# 6. Body-text-length figure
# ------------------------------------------------------------

text_99th = eda_df["text_word_count"].quantile(0.99)

text_label_0 = (
    eda_df.loc[eda_df["label"] == 0, "text_word_count"]
    .clip(upper=text_99th)
)

text_label_1 = (
    eda_df.loc[eda_df["label"] == 1, "text_word_count"]
    .clip(upper=text_99th)
)

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot(
    [text_label_0, text_label_1],
    showfliers=False
)

ax.set_title("Body Text Word Count by Label")
ax.set_xlabel("Observed Label")
ax.set_ylabel("Number of Words in Body Text")
ax.set_xticklabels(["Label 0", "Label 1"])

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "04_body_word_count_by_label.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# 7. Confirm generated files
# ------------------------------------------------------------

print("\nVisualization limits:")
print(f"Title word-count 99th percentile: {title_99th:.2f}")
print(f"Body word-count 99th percentile: {text_99th:.2f}")

print("\nFiles saved in:")
print(OUTPUT_DIR)

for saved_file in sorted(OUTPUT_DIR.iterdir()):
    print("-", saved_file.name)

    # ============================================================
# Label Verification, Duplicate Details, Common Words,
# and Potential Label Leakage
# ============================================================

import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------------------------------------
# 8. Inspect example articles to verify label meaning
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE ARTICLES FOR LABEL VERIFICATION")
print("=" * 70)

for current_label in sorted(eda_df["label"].unique()):
    print(f"\nExamples from Label {current_label}:")

    sample_rows = (
        eda_df.loc[
            eda_df["label"] == current_label,
            ["title_clean", "text_clean"]
        ]
        .sample(n=5, random_state=503)
    )

    for sample_number, (_, row) in enumerate(
            sample_rows.iterrows(),
            start=1
    ):
        body_preview = row["text_clean"][:200].replace("\n", " ")

        print(f"\n{sample_number}. Title: {row['title_clean']}")
        print(f"   Body preview: {body_preview}...")

# ------------------------------------------------------------
# 9. Extended blank-value analysis
# ------------------------------------------------------------

blank_title_mask = eda_df["title_clean"].eq("")
blank_text_mask = eda_df["text_clean"].eq("")

blank_title_or_text = int(
    (blank_title_mask | blank_text_mask).sum()
)

blank_title_and_text = int(
    (blank_title_mask & blank_text_mask).sum()
)

non_null_but_blank_text = int(
    blank_text_mask.sum() - df["text"].isna().sum()
)

print("\n" + "=" * 70)
print("EXTENDED BLANK-VALUE CHECK")
print("=" * 70)

print("Rows with a blank title or blank body:", blank_title_or_text)
print("Rows with both title and body blank:", blank_title_and_text)
print(
    "Body texts that are blank but not stored as NaN:",
    non_null_but_blank_text
)

# ------------------------------------------------------------
# 10. More detailed duplicate analysis
# ------------------------------------------------------------

duplicate_group_summary = (
    eda_df.groupby(
        ["title_clean", "text_clean"],
        sort=False
    )["label"]
    .agg(
        group_size="size",
        number_of_labels="nunique"
    )
    .reset_index()
)

duplicate_groups = duplicate_group_summary[
    duplicate_group_summary["group_size"] > 1
    ]

number_of_duplicate_groups = len(duplicate_groups)
largest_duplicate_group = int(
    duplicate_groups["group_size"].max()
)

conflicting_label_groups = int(
    (
            duplicate_groups["number_of_labels"] > 1
    ).sum()
)

unique_article_count = int(
    eda_df.drop_duplicates(
        subset=["title_clean", "text_clean"]
    ).shape[0]
)

duplicate_percentage = (
        duplicate_title_text / len(eda_df) * 100
)

print("\n" + "=" * 70)
print("DETAILED DUPLICATE CHECK")
print("=" * 70)

print("Duplicate records beyond the first copy:", duplicate_title_text)
print(f"Duplicate percentage: {duplicate_percentage:.2f}%")
print("Number of duplicate article groups:", number_of_duplicate_groups)
print("Largest duplicate group size:", largest_duplicate_group)
print("Duplicate groups with conflicting labels:", conflicting_label_groups)
print("Unique articles after exact deduplication:", unique_article_count)

duplicate_details = pd.DataFrame(
    {
        "Measure": [
            "Duplicate records beyond first copy",
            "Duplicate percentage",
            "Number of duplicate article groups",
            "Largest duplicate group size",
            "Duplicate groups with conflicting labels",
            "Unique articles after exact deduplication",
        ],
        "Value": [
            duplicate_title_text,
            round(duplicate_percentage, 2),
            number_of_duplicate_groups,
            largest_duplicate_group,
            conflicting_label_groups,
            unique_article_count,
        ],
    }
)

duplicate_details.to_csv(
    OUTPUT_DIR / "duplicate_analysis.csv",
    index=False
)

# ------------------------------------------------------------
# 11. Find common words
# ------------------------------------------------------------

def calculate_top_words(
        texts,
        top_n=20,
        max_features=5000
):
    """
    Calculate the most frequent non-stopword terms.
    This is used only for exploratory analysis.
    """

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=10,
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        dtype=np.int32,
    )

    word_matrix = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    counts = np.asarray(
        word_matrix.sum(axis=0)
    ).ravel()

    result = pd.DataFrame(
        {
            "word": words,
            "count": counts,
        }
    )

    return (
        result.sort_values(
            "count",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def save_top_words_chart(
        top_words_df,
        title,
        filename
):
    plot_data = top_words_df.sort_values(
        "count",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.barh(
        plot_data["word"],
        plot_data["count"]
    )

    ax.set_title(title)
    ax.set_xlabel("Total Word Frequency")
    ax.set_ylabel("Word")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


top_word_results = {}

for field_name, field_column in [
    ("title", "title_clean"),
    ("body", "text_clean"),
]:
    for current_label in sorted(
            eda_df["label"].unique()
    ):
        current_texts = eda_df.loc[
            (eda_df["label"] == current_label)
            & (eda_df[field_column] != ""),
            field_column
        ]

        result_name = (
            f"{field_name}_label_{current_label}"
        )

        top_words_result = calculate_top_words(
            current_texts,
            top_n=20
        )

        top_word_results[result_name] = (
            top_words_result
        )

        top_words_result.to_csv(
            OUTPUT_DIR
            / f"top_words_{result_name}.csv",
            index=False
        )

        save_top_words_chart(
            top_words_result,
            (
                f"Most Frequent Words in "
                f"{field_name.title()} — "
                f"Label {current_label}"
            ),
            f"top_words_{result_name}.png"
        )

        print("\n" + "=" * 70)
        print(
            f"TOP WORDS: {field_name.upper()}, "
            f"LABEL {current_label}"
        )
        print("=" * 70)

        print(
            top_words_result.head(15).to_string(
                index=False
            )
        )

# ------------------------------------------------------------
# 12. Check potential source markers and bylines
# ------------------------------------------------------------

combined_lower = (
        eda_df["title_clean"]
        + " "
        + eda_df["text_clean"]
).str.lower()

source_markers = [
    "reuters",
    "associated press",
    "cnn",
    "fox news",
    "breitbart",
    "new york times",
    "nytimes",
    "washington post",
    "huffington post",
    "bbc",
    "the guardian",
    "daily mail",
]

marker_rows = []

for marker in source_markers:
    marker_pattern = (
        rf"\b{re.escape(marker)}\b"
    )

    for current_label in sorted(
            eda_df["label"].unique()
    ):
        label_mask = (
                eda_df["label"] == current_label
        )

        label_documents = combined_lower[
            label_mask
        ]

        marker_count = int(
            label_documents.str.contains(
                marker_pattern,
                regex=True,
                na=False
            ).sum()
        )

        marker_percentage = (
                marker_count
                / len(label_documents)
                * 100
        )

        marker_rows.append(
            {
                "marker": marker,
                "label": current_label,
                "document_count": marker_count,
                "document_percentage": round(
                    marker_percentage,
                    3
                ),
            }
        )

marker_summary = pd.DataFrame(marker_rows)

marker_pivot = marker_summary.pivot(
    index="marker",
    columns="label",
    values="document_percentage"
).reset_index()

marker_pivot.columns = [
    "marker",
    "label_0_percentage",
    "label_1_percentage",
]

marker_pivot["absolute_gap"] = (
        marker_pivot["label_1_percentage"]
        - marker_pivot["label_0_percentage"]
).abs()

marker_pivot = marker_pivot.sort_values(
    "absolute_gap",
    ascending=False
)

print("\n" + "=" * 70)
print("POTENTIAL SOURCE-MARKER LEAKAGE")
print("=" * 70)

print(marker_pivot.to_string(index=False))

marker_summary.to_csv(
    OUTPUT_DIR
    / "source_marker_counts_by_label.csv",
    index=False
)

marker_pivot.to_csv(
    OUTPUT_DIR
    / "source_marker_percentage_comparison.csv",
    index=False
)

print("\nNew files saved in:")
print(OUTPUT_DIR)