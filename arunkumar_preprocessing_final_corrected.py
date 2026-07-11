"""
CSC503 Group Project
Part 2: Text Preprocessing Pipeline
Student: Arunkumar Krishnamoorthy

Input file:
    WELFake after cleaning.csv.gz

Expected label mapping:
    0 = real news
    1 = fake news

Outputs:
    processed/WELFake_part2_preprocessed.csv
    processed/WELFake_part2_preprocessed.csv.gz
    processed/source_marker_audit.csv
    processed/before_after_samples.csv
    processed/preprocessing_quality_report.json
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Regular expressions used during cleaning
# -------------------------------------------------------------------

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
    flags=re.IGNORECASE,
)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
EXTRA_SPACE_PATTERN = re.compile(r"\s+")

# Keep letters and apostrophes, remove numbers and punctuation.
NON_LETTER_PATTERN = re.compile(r"[^a-zA-Z\s']")

# Conservative Reuters-style dateline removal.
REUTERS_DATELINE_PATTERN = re.compile(
    r"^\s*[A-Z][A-Z .'-]{1,40}\s*(?:\([A-Za-z]+\))?\s*[-–—]\s*"
)

# Source markers identified during Stage-1 EDA.
SOURCE_MARKERS = [
    "reuters",
    "new york times",
    "breitbart",
    "cnn",
    "fox news",
]


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def normalize_missing_values(series: pd.Series) -> pd.Series:
    """
    Replace missing values with empty strings and make all values strings.
    """
    return series.fillna("").astype(str)


def remove_source_markers(text: str) -> str:
    """
    Remove source markers identified by Stage-1 EDA.

    Only markers confirmed by the group are removed.
    """
    cleaned = str(text)

    # Remove "(Reuters)" style text.
    cleaned = re.sub(
        r"\(\s*reuters\s*\)",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Remove Reuters-style location/dateline at the beginning.
    cleaned = REUTERS_DATELINE_PATTERN.sub(" ", cleaned)

    # Remove publication/source names.
    for marker in sorted(SOURCE_MARKERS, key=len, reverse=True):
        cleaned = re.sub(
            rf"\b{re.escape(marker)}\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )

    return cleaned


def clean_text(text: object, remove_markers: bool = True) -> str:
    """
    Clean one title or article body.

    Steps:
    1. Handle missing values.
    2. Decode HTML entities.
    3. Remove HTML tags.
    4. Remove URLs and email addresses.
    5. Optionally remove source markers.
    6. Convert to lowercase.
    7. Remove punctuation, symbols, and numbers.
    8. Normalize whitespace.

    Stopword removal, stemming, and lemmatization are not applied because
    they may remove useful negation and writing-style information.
    """
    if text is None or pd.isna(text):
        return ""

    cleaned = html.unescape(str(text))
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = EMAIL_PATTERN.sub(" ", cleaned)

    if remove_markers:
        cleaned = remove_source_markers(cleaned)

    cleaned = cleaned.lower()
    cleaned = NON_LETTER_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"'{2,}", "'", cleaned)
    cleaned = EXTRA_SPACE_PATTERN.sub(" ", cleaned).strip()

    return cleaned


def build_marker_audit(
    dataframe: pd.DataFrame,
    text_column: str,
    stage_name: str,
) -> pd.DataFrame:
    """
    Count how often each source marker occurs in real and fake articles.
    """
    rows = []

    for marker in SOURCE_MARKERS:
        marker_pattern = rf"\b{re.escape(marker)}\b"

        contains_marker = (
            dataframe[text_column]
            .fillna("")
            .astype(str)
            .str.contains(marker_pattern, case=False, regex=True)
        )

        for label in [0, 1]:
            class_mask = dataframe["label"].eq(label)
            class_total = int(class_mask.sum())
            marker_count = int((contains_marker & class_mask).sum())

            percentage = (
                round((marker_count / class_total) * 100, 4)
                if class_total > 0
                else 0.0
            )

            rows.append(
                {
                    "stage": stage_name,
                    "marker": marker,
                    "label": label,
                    "class_name": "real" if label == 0 else "fake",
                    "articles_with_marker": marker_count,
                    "class_articles": class_total,
                    "percentage": percentage,
                }
            )

    return pd.DataFrame(rows)


def preprocess_dataset(
    dataframe: pd.DataFrame,
    remove_markers: bool = True,
) -> pd.DataFrame:
    """
    Create the final title-only, body-only, and combined text fields.

    This version first removes any old preprocessing columns that may already
    exist in the input file. That prevents duplicate-looking or repeated
    output columns when the script is run more than once.
    """
    required_columns = {"title", "text", "label"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    processed = dataframe.copy()

    # Remove accidental serial-number columns.
    unwanted_columns = [
        column
        for column in processed.columns
        if str(column).lower().startswith("unnamed:")
    ]
    if unwanted_columns:
        processed = processed.drop(columns=unwanted_columns)

    # Remove any preprocessing columns left from an earlier run.
    generated_columns = [
        "raw_title_body",
        "clean_title_with_markers",
        "clean_body_with_markers",
        "clean_title_body_with_markers",
        "clean_title",
        "clean_body",
        "clean_title_body",
        "clean_title_word_count",
        "clean_body_word_count",
        "clean_combined_word_count",
    ]

    existing_generated_columns = [
        column for column in generated_columns
        if column in processed.columns
    ]

    if existing_generated_columns:
        processed = processed.drop(columns=existing_generated_columns)

    # Make sure the input does not contain duplicate column names.
    if processed.columns.duplicated().any():
        duplicate_names = (
            processed.columns[processed.columns.duplicated()]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "The input file contains duplicate column names: "
            f"{duplicate_names}"
        )

    processed["title"] = normalize_missing_values(processed["title"])
    processed["text"] = normalize_missing_values(processed["text"])

    # Keep original combined text for audit/documentation.
    processed["raw_title_body"] = (
        processed["title"].str.strip()
        + " "
        + processed["text"].str.strip()
    ).str.strip()

    # Clean versions that retain source markers.
    processed["clean_title_with_markers"] = processed["title"].apply(
        lambda value: clean_text(value, remove_markers=False)
    )

    processed["clean_body_with_markers"] = processed["text"].apply(
        lambda value: clean_text(value, remove_markers=False)
    )

    processed["clean_title_body_with_markers"] = (
        processed["clean_title_with_markers"]
        + " "
        + processed["clean_body_with_markers"]
    ).str.strip()

    # Final leakage-controlled clean versions.
    processed["clean_title"] = processed["title"].apply(
        lambda value: clean_text(
            value,
            remove_markers=remove_markers,
        )
    )

    processed["clean_body"] = processed["text"].apply(
        lambda value: clean_text(
            value,
            remove_markers=remove_markers,
        )
    )

    processed["clean_title_body"] = (
        processed["clean_title"]
        + " "
        + processed["clean_body"]
    ).str.strip()

    # Quality-check columns.
    processed["clean_title_word_count"] = (
        processed["clean_title"]
        .str.split()
        .str.len()
        .fillna(0)
        .astype(int)
    )

    processed["clean_body_word_count"] = (
        processed["clean_body"]
        .str.split()
        .str.len()
        .fillna(0)
        .astype(int)
    )

    processed["clean_combined_word_count"] = (
        processed["clean_title_body"]
        .str.split()
        .str.len()
        .fillna(0)
        .astype(int)
    )

    # Put important columns in a clear, fixed order.
    preferred_order = [
        "article_id",
        "source_row_id",
        "row_id",
        "title",
        "text",
        "label",
        "class_name",
        "title_missing",
        "text_missing",
        "complete_case",
        "raw_title_body",
        "clean_title_with_markers",
        "clean_body_with_markers",
        "clean_title_body_with_markers",
        "clean_title",
        "clean_body",
        "clean_title_body",
        "clean_title_word_count",
        "clean_body_word_count",
        "clean_combined_word_count",
    ]

    ordered_columns = [
        column for column in preferred_order
        if column in processed.columns
    ]

    remaining_columns = [
        column for column in processed.columns
        if column not in ordered_columns
    ]

    processed = processed[ordered_columns + remaining_columns]

    # Final safety check before saving.
    if processed.columns.duplicated().any():
        duplicate_names = (
            processed.columns[processed.columns.duplicated()]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "Duplicate output columns were detected: "
            f"{duplicate_names}"
        )

    return processed


def create_quality_report(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    source_markers_removed: bool,
) -> dict:
    """
    Create a JSON-friendly summary of the preprocessing results.
    """
    report = {
        "student": "Arunkumar Krishnamoorthy",
        "project_part": "Part 2 - Text Preprocessing Pipeline",
        "input_rows": int(len(original)),
        "output_rows": int(len(processed)),
        "label_mapping": {
            "0": "real",
            "1": "fake",
        },
        "label_counts": {
            str(label): int(count)
            for label, count in (
                processed["label"]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            )
        },
        "blank_clean_titles": int(
            processed["clean_title"].eq("").sum()
        ),
        "blank_clean_bodies": int(
            processed["clean_body"].eq("").sum()
        ),
        "blank_clean_combined": int(
            processed["clean_title_body"].eq("").sum()
        ),
        "mean_clean_title_words": round(
            float(processed["clean_title_word_count"].mean()),
            3,
        ),
        "mean_clean_body_words": round(
            float(processed["clean_body_word_count"].mean()),
            3,
        ),
        "source_markers_removed": source_markers_removed,
        "configured_markers": SOURCE_MARKERS,
        "stopwords_removed": False,
        "stemming_applied": False,
        "lemmatization_applied": False,
    }

    if "article_id" in processed.columns:
        report["duplicate_article_ids"] = int(
            processed["article_id"].duplicated().sum()
        )

    return report


# -------------------------------------------------------------------
# Main program
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Part 2 preprocessing on the WELFake dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the Stage-1 CSV or CSV.GZ file.",
    )

    parser.add_argument(
        "--output-dir",
        default="processed",
        help="Folder where output files will be saved.",
    )

    parser.add_argument(
        "--keep-source-markers",
        action="store_true",
        help=(
            "Keep Reuters and other source markers in the final clean columns. "
            "By default, confirmed markers are removed."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {input_path.resolve()}"
        )

    print("Loading dataset...")
    original = pd.read_csv(input_path)

    print(f"Input rows: {len(original):,}")
    print(f"Input columns: {original.columns.tolist()}")

    # Prepare original combined text for marker auditing.
    audit_input = original.copy()
    audit_input["title"] = normalize_missing_values(audit_input["title"])
    audit_input["text"] = normalize_missing_values(audit_input["text"])
    audit_input["raw_title_body"] = (
        audit_input["title"].str.strip()
        + " "
        + audit_input["text"].str.strip()
    ).str.strip()

    marker_before = build_marker_audit(
        audit_input,
        text_column="raw_title_body",
        stage_name="before_preprocessing",
    )

    remove_markers = not args.keep_source_markers

    print("Preprocessing text...")
    processed = preprocess_dataset(
        original,
        remove_markers=remove_markers,
    )

    print("\nFinal output columns:")
    for position, column_name in enumerate(processed.columns, start=1):
        print(f"{position:02d}. {column_name}")

    if processed.columns.duplicated().any():
        raise ValueError("Duplicate column names detected before saving.")

    marker_after = build_marker_audit(
        processed,
        text_column="clean_title_body",
        stage_name="after_preprocessing",
    )

    marker_audit = pd.concat(
        [marker_before, marker_after],
        ignore_index=True,
    )

    # ---------------------------------------------------------------
    # Save normal CSV for Excel
    # ---------------------------------------------------------------
    normal_csv_path = (
        output_dir / "WELFake_part2_preprocessed.csv"
    )

    print("Saving Excel-friendly CSV...")
    processed.to_csv(
        normal_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------
    # Save compressed CSV for GitHub/team sharing
    # ---------------------------------------------------------------
    compressed_csv_path = (
        output_dir / "WELFake_part2_preprocessed.csv.gz"
    )

    print("Saving compressed CSV...")
    processed.to_csv(
        compressed_csv_path,
        index=False,
        compression="gzip",
        encoding="utf-8",
    )

    # Save source-marker audit.
    marker_audit_path = output_dir / "source_marker_audit.csv"
    marker_audit.to_csv(
        marker_audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Save before-and-after examples.
    sample_columns = [
        column
        for column in [
            "article_id",
            "row_id",
            "label",
            "class_name",
            "title",
            "clean_title",
            "text",
            "clean_body",
            "clean_title_body",
        ]
        if column in processed.columns
    ]

    samples_path = output_dir / "before_after_samples.csv"
    processed[sample_columns].head(20).to_csv(
        samples_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Save quality report.
    quality_report = create_quality_report(
        original=original,
        processed=processed,
        source_markers_removed=remove_markers,
    )

    report_path = (
        output_dir / "preprocessing_quality_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            quality_report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Final terminal summary.
    print("\nPreprocessing completed successfully.")
    print(json.dumps(quality_report, indent=2))
    print("\nFiles saved:")
    print(f"1. {normal_csv_path.resolve()}")
    print(f"2. {compressed_csv_path.resolve()}")
    print(f"3. {marker_audit_path.resolve()}")
    print(f"4. {samples_path.resolve()}")
    print(f"5. {report_path.resolve()}")


if __name__ == "__main__":
    main()
