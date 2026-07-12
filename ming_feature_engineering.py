"""
CSC503 Group Project
Part 3: Feature Engineering and Dimensionality Reduction
Student:

Input file:
    WELFake_part2_preprocessed.csv.gz   (output of Arunkumar's Part 2 script)

Required input columns:
    article_id, label, clean_title, clean_body, clean_title_body

Outputs (all under --output-dir):
    split_assignment.csv
        article_id, label, split   (split in {train, val, test})
        -> hand this file to Etta and Ming so every group member trains /
           validates / tests on the EXACT same rows.

    For each input setting in {title, body, combined}
    and each vectorizer in {bow_unigram, tfidf_unigram, tfidf_uni_bigram}:
        features/<setting>_<vectorizer>_<split>.npz   (sparse matrix)
        features/<setting>_<vectorizer>_vocab.json    (feature names)

    For each input setting in {title, body, combined}
    (SVD/PCA is only computed on the tfidf_uni_bigram representation,
     see report for justification):
        features_reduced/<setting>_svd_<split>.npz    (dense, sparse-saved)
        features_reduced/<setting>_svd_<split>.csv    (dense CSV, for KNN)
        features_reduced/<setting>_svd_explained_variance.json

    feature_engineering_report.json
        Summary of vocabulary sizes, matrix shapes, explained variance, etc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

RANDOM_SEED = 123

REQUIRED_COLUMNS = {
    "article_id",
    "label",
    "clean_title",
    "clean_body",
    "clean_title_body",
}

INPUT_SETTINGS = {
    "title": "clean_title",
    "body": "clean_body",
    "combined": "clean_title_body",
}

# Token pattern: cleaned text only contains lowercase letters, spaces and
# apostrophes (see Part 2 NON_LETTER_PATTERN), so this pattern matches
# Part 2's output exactly instead of relying on sklearn's default \w+.
TOKEN_PATTERN = r"(?u)[a-z']+"

# n_components for TruncatedSVD. Kept well below min(n_samples, n_features)
# and capped so KNN distance computations stay tractable. See report for
# how this number was chosen.
SVD_N_COMPONENTS = 300


# -------------------------------------------------------------------
# Splitting
# -------------------------------------------------------------------

def make_split(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified 80/10/10 train/val/test split on a fixed random seed.

    This assignment is saved separately so every group member (feature
    engineering, baseline models, advanced models) trains, tunes and
    tests on identical rows, as required by the project proposal.
    """
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=0.20,
        stratify=dataframe["label"],
        random_state=RANDOM_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )

    split_assignment = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ]
    )[["article_id", "label", "split"]].sort_values("article_id")

    return split_assignment


# -------------------------------------------------------------------
# Vectorization
# -------------------------------------------------------------------

def build_vectorizer(kind: str):
    """
    kind in {"bow_unigram", "tfidf_unigram", "tfidf_uni_bigram"}
    """
    if kind == "bow_unigram":
        return CountVectorizer(
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 1),
            lowercase=False,  # already lowercased in Part 2
        )
    if kind == "tfidf_unigram":
        return TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 1),
            lowercase=False,
        )
    if kind == "tfidf_uni_bigram":
        return TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 2),
            lowercase=False,
        )
    raise ValueError(f"Unknown vectorizer kind: {kind}")


def vectorize_setting(
    texts: dict[str, pd.Series],
    vectorizer_kind: str,
) -> dict:
    """
    Fit a vectorizer on texts['train'] ONLY, then transform train/val/test.

    Fitting on train-only prevents vocabulary/IDF leakage from val/test
    into the training-time feature space.
    """
    vectorizer = build_vectorizer(vectorizer_kind)

    matrices = {}
    matrices["train"] = vectorizer.fit_transform(texts["train"])
    matrices["val"] = vectorizer.transform(texts["val"])
    matrices["test"] = vectorizer.transform(texts["test"])

    vocabulary = vectorizer.get_feature_names_out().tolist()

    return {
        "matrices": matrices,
        "vocabulary": vocabulary,
        "vectorizer": vectorizer,
    }


# -------------------------------------------------------------------
# Dimensionality reduction
# -------------------------------------------------------------------

def reduce_with_svd(
    matrices: dict,
    n_components: int,
) -> dict:
    """
    TruncatedSVD fit on train ONLY, applied to val/test.

    TruncatedSVD (not classic PCA) is used because the input is a large
    sparse matrix. Classic PCA requires mean-centering, which would
    densify the matrix and likely exceed memory. TruncatedSVD operates
    directly on sparse input and is the standard scikit-learn approach
    for "PCA on text features" (this is literally what scikit-learn's
    documentation recommends for LSA-style dimensionality reduction).
    """
    n_components = min(
        n_components,
        matrices["train"].shape[0] - 1,
        matrices["train"].shape[1] - 1,
    )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=RANDOM_SEED,
    )

    reduced = {}
    reduced["train"] = svd.fit_transform(matrices["train"])
    reduced["val"] = svd.transform(matrices["val"])
    reduced["test"] = svd.transform(matrices["test"])

    explained_variance_ratio = svd.explained_variance_ratio_
    cumulative = np.cumsum(explained_variance_ratio)

    return {
        "reduced": reduced,
        "n_components": n_components,
        "explained_variance_ratio": explained_variance_ratio.tolist(),
        "cumulative_explained_variance": cumulative.tolist(),
        "total_explained_variance": float(cumulative[-1]) if len(cumulative) else 0.0,
    }


# -------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------

def save_sparse_split(
    matrices: dict,
    out_dir: Path,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, matrix in matrices.items():
        sparse.save_npz(out_dir / f"{prefix}_{split_name}.npz", matrix)


def save_dense_split(
    reduced: dict,
    out_dir: Path,
    prefix: str,
    article_ids: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, matrix in reduced.items():
        # .npz (dense, but saved via savez_compressed for consistency)
        np.savez_compressed(
            out_dir / f"{prefix}_{split_name}.npz",
            data=matrix,
            article_id=article_ids[split_name].to_numpy(),
        )
        # .csv, with article_id as the first column so it can be joined
        # back to labels/other feature sets downstream.
        column_names = [f"svd_{i:03d}" for i in range(matrix.shape[1])]
        dense_df = pd.DataFrame(matrix, columns=column_names)
        dense_df.insert(0, "article_id", article_ids[split_name].to_numpy())
        dense_df.to_csv(out_dir / f"{prefix}_{split_name}.csv", index=False)


def save_vocabulary(vocabulary: list[str], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{prefix}_vocab.json", "w", encoding="utf-8") as file:
        json.dump(vocabulary, file, ensure_ascii=False, indent=2)


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def run_pipeline(input_path: Path, output_dir: Path, svd_n_components: int) -> dict:
    print("Loading Part 2 preprocessed dataset...")
    df = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    print(f"Input rows: {len(df):,}")
    if len(df) != 62590:
        print(
            f"WARNING: expected 62,590 complete-case rows per project handoff "
            f"docs, got {len(df):,}. Continuing anyway."
        )

    for col in ["clean_title", "clean_body", "clean_title_body"]:
        df[col] = df[col].fillna("")

    print("Creating stratified 80/10/10 train/val/test split "
          f"(random_state={RANDOM_SEED})...")
    split_assignment = make_split(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "split_assignment.csv"
    split_assignment.to_csv(split_path, index=False)

    df_indexed = df.set_index("article_id")
    split_by_name = {
        name: split_assignment.loc[split_assignment["split"] == name, "article_id"]
        for name in ["train", "val", "test"]
    }

    report = {
        "student": "Ming",
        "project_part": "Part 3 - Feature Engineering and Dimensionality Reduction",
        "random_seed": RANDOM_SEED,
        "input_rows": int(len(df)),
        "split_sizes": {
            name: int(len(ids)) for name, ids in split_by_name.items()
        },
        "svd_n_components_requested": svd_n_components,
        "input_settings": {},
    }

    features_dir = output_dir / "features"
    reduced_dir = output_dir / "features_reduced"

    for setting_name, column in INPUT_SETTINGS.items():
        print(f"\n=== Input setting: {setting_name} ({column}) ===")

        texts = {
            name: df_indexed.loc[ids, column]
            for name, ids in split_by_name.items()
        }
        article_ids = split_by_name  # same ids, reused for output ordering

        setting_report = {"vectorizers": {}}

        for vec_kind in ["bow_unigram", "tfidf_unigram", "tfidf_uni_bigram"]:
            print(f"  Fitting vectorizer: {vec_kind} (fit on train only)...")
            result = vectorize_setting(texts, vec_kind)

            prefix = f"{setting_name}_{vec_kind}"
            save_sparse_split(result["matrices"], features_dir, prefix)
            save_vocabulary(result["vocabulary"], features_dir, prefix)

            setting_report["vectorizers"][vec_kind] = {
                "vocabulary_size": len(result["vocabulary"]),
                "train_shape": list(result["matrices"]["train"].shape),
                "val_shape": list(result["matrices"]["val"].shape),
                "test_shape": list(result["matrices"]["test"].shape),
            }

            print(
                f"    vocab size = {len(result['vocabulary']):,} | "
                f"train shape = {result['matrices']['train'].shape}"
            )

            # Only run SVD/PCA on the tfidf_uni_bigram representation.
            # Rationale is explained in the accompanying report.
            if vec_kind == "tfidf_uni_bigram":
                print(f"  Running TruncatedSVD (target={svd_n_components} "
                      "components, fit on train only)...")
                svd_result = reduce_with_svd(
                    result["matrices"], svd_n_components
                )
                svd_prefix = f"{setting_name}_svd"
                save_dense_split(
                    svd_result["reduced"], reduced_dir, svd_prefix, article_ids
                )

                with open(
                    reduced_dir / f"{svd_prefix}_explained_variance.json",
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        {
                            "n_components": svd_result["n_components"],
                            "total_explained_variance": svd_result[
                                "total_explained_variance"
                            ],
                            "explained_variance_ratio": svd_result[
                                "explained_variance_ratio"
                            ],
                            "cumulative_explained_variance": svd_result[
                                "cumulative_explained_variance"
                            ],
                        },
                        file,
                        indent=2,
                    )

                setting_report["svd"] = {
                    "n_components": svd_result["n_components"],
                    "total_explained_variance": round(
                        svd_result["total_explained_variance"], 4
                    ),
                }

                print(
                    f"    SVD n_components = {svd_result['n_components']} | "
                    "total explained variance = "
                    f"{svd_result['total_explained_variance']:.4f}"
                )

        report["input_settings"][setting_name] = setting_report

    report_path = output_dir / "feature_engineering_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nDone. Report saved to:", report_path.resolve())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Part 3 feature engineering + dimensionality "
        "reduction on the WELFake Part 2 preprocessed dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to WELFake_part2_preprocessed.csv or .csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="part3_output",
        help="Folder where output files will be saved.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=SVD_N_COMPONENTS,
        help=f"Target number of TruncatedSVD components (default: "
        f"{SVD_N_COMPONENTS}).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file was not found: {input_path.resolve()}")

    run_pipeline(
        input_path=input_path,
        output_dir=Path(args.output_dir),
        svd_n_components=args.svd_components,
    )


if __name__ == "__main__":
    main()