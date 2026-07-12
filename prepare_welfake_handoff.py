from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "WELFake_Dataset.csv"
OUTPUT_DIR = BASE_DIR / "handoff"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = (
        OUTPUT_DIR / "WELFake_stage1_master_deduplicated.csv.gz"
)

COMPLETE_CASES_PATH = (
        OUTPUT_DIR / "WELFake_stage1_complete_cases.csv.gz"
)

REMOVED_DUPLICATES_PATH = (
        OUTPUT_DIR / "WELFake_removed_exact_duplicates.csv.gz"
)

INCOMPLETE_CASES_PATH = (
        OUTPUT_DIR / "WELFake_incomplete_cases.csv.gz"
)


def clean_text_field(series: pd.Series) -> pd.Series:
    """
    Stage-1 cleaning only:

    1. Convert missing values to empty strings.
    2. Convert values to strings.
    3. Remove leading and trailing whitespace.

    This function intentionally does NOT:
    - lowercase text
    - remove punctuation
    - remove stop words
    - remove source markers such as Reuters
    - perform stemming or lemmatization
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def main() -> None:
    # --------------------------------------------------------
    # 1. Check and load the original dataset
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found:\n{INPUT_PATH}"
        )

    print("Loading dataset from:")
    print(INPUT_PATH)

    raw = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    # Standardize column names
    raw.columns = (
        raw.columns
        .str.strip()
        .str.lower()
    )

    required_columns = {
        "title",
        "text",
        "label"
    }

    missing_columns = (
            required_columns - set(raw.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Rename the original serial-number column
    if "unnamed: 0" in raw.columns:
        raw = raw.rename(
            columns={
                "unnamed: 0": "source_row_id"
            }
        )
    else:
        raw.insert(
            0,
            "source_row_id",
            range(len(raw))
        )

    # Keep only fields required for this stage
    raw = raw[
        [
            "source_row_id",
            "title",
            "text",
            "label"
        ]
    ].copy()

    raw["label"] = pd.to_numeric(
        raw["label"],
        errors="raise"
    ).astype(int)

    observed_labels = set(
        raw["label"].unique()
    )

    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            f"Unexpected labels: "
            f"{sorted(observed_labels)}"
        )

    # --------------------------------------------------------
    # 2. Minimal Stage-1 text handling
    # --------------------------------------------------------

    raw["title"] = clean_text_field(
        raw["title"]
    )

    raw["text"] = clean_text_field(
        raw["text"]
    )

    # Mapping confirmed from the actual downloaded CSV
    raw["class_name"] = raw["label"].map(
        {
            0: "real",
            1: "fake"
        }
    )

    raw["title_missing"] = (
        raw["title"].eq("")
    )

    raw["text_missing"] = (
        raw["text"].eq("")
    )

    raw["complete_case"] = ~(
            raw["title_missing"]
            | raw["text_missing"]
    )

    # --------------------------------------------------------
    # 3. Check whether identical articles have conflicting labels
    # --------------------------------------------------------

    labels_per_article = (
        raw
        .groupby(
            ["title", "text"],
            sort=False
        )["label"]
        .nunique()
    )

    conflicting_groups = int(
        (labels_per_article > 1).sum()
    )

    if conflicting_groups > 0:
        raise ValueError(
            f"Found {conflicting_groups} duplicate "
            f"article groups with conflicting labels."
        )

    # --------------------------------------------------------
    # 4. Detect exact duplicate articles
    # --------------------------------------------------------

    duplicate_mask = raw.duplicated(
        subset=[
            "title",
            "text"
        ],
        keep="first"
    )

    # File 3: copies removed as exact duplicates
    removed_duplicates = (
        raw.loc[duplicate_mask]
        .copy()
        .reset_index(drop=True)
    )

    removed_duplicates["removal_reason"] = (
        "exact_duplicate_title_text"
    )

    # File 1: deduplicated master data
    master = (
        raw.loc[~duplicate_mask]
        .copy()
        .reset_index(drop=True)
    )

    master.insert(
        0,
        "article_id",
        range(len(master))
    )

    # --------------------------------------------------------
    # 5. Separate complete and incomplete unique records
    # --------------------------------------------------------

    # File 2: title and body are both available
    complete_cases = (
        master.loc[master["complete_case"]]
        .copy()
        .reset_index(drop=True)
    )

    # File 4: title or body is missing
    incomplete_cases = (
        master.loc[~master["complete_case"]]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 6. Save the four files
    # --------------------------------------------------------

    master.to_csv(
        MASTER_PATH,
        index=False,
        compression="gzip",
        encoding="utf-8"
    )

    complete_cases.to_csv(
        COMPLETE_CASES_PATH,
        index=False,
        compression="gzip",
        encoding="utf-8"
    )

    removed_duplicates.to_csv(
        REMOVED_DUPLICATES_PATH,
        index=False,
        compression="gzip",
        encoding="utf-8"
    )

    incomplete_cases.to_csv(
        INCOMPLETE_CASES_PATH,
        index=False,
        compression="gzip",
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 7. Print result summary
    # --------------------------------------------------------

    print("\nStage-1 data preparation completed.")

    print("\nOriginal records:")
    print(f"{len(raw):,}")

    print("\nExact duplicate copies removed:")
    print(f"{len(removed_duplicates):,}")

    print("\nDeduplicated master records:")
    print(f"{len(master):,}")

    print("\nComplete-case records:")
    print(f"{len(complete_cases):,}")

    print("\nIncomplete unique records:")
    print(f"{len(incomplete_cases):,}")

    print("\nDeduplicated master label counts:")
    print(master["label"].value_counts().sort_index())

    print("\nComplete-case label counts:")
    print(
        complete_cases[
            "label"
        ].value_counts().sort_index()
    )

    print("\nGenerated files:")

    for file_path in [
        MASTER_PATH,
        COMPLETE_CASES_PATH,
        REMOVED_DUPLICATES_PATH,
        INCOMPLETE_CASES_PATH,
    ]:
        size_mb = (
                file_path.stat().st_size
                / 1024
                / 1024
        )

        print(
            f"- {file_path.name}: "
            f"{size_mb:.2f} MB"
        )

    # Expected results for your current WELFake CSV
    expected_counts = {
        "original": 72134,
        "duplicates": 8458,
        "master": 63676,
        "complete": 62590,
        "incomplete": 1086,
    }

    actual_counts = {
        "original": len(raw),
        "duplicates": len(
            removed_duplicates
        ),
        "master": len(master),
        "complete": len(
            complete_cases
        ),
        "incomplete": len(
            incomplete_cases
        ),
    }

    print("\nCount verification:")

    for name, expected in expected_counts.items():
        actual = actual_counts[name]

        status = (
            "OK"
            if actual == expected
            else "CHECK"
        )

        print(
            f"{name}: "
            f"actual={actual:,}, "
            f"expected={expected:,} "
            f"[{status}]"
        )


if __name__ == "__main__":
    main()