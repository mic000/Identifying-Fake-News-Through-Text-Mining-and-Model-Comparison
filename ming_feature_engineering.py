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
        features_reduced/<setting>_svd_<split>.csv    (dense CSV, for KNN;
            includes an `article_id` column for joining -- this is an ID,
            not a model feature, and must be dropped before training)
        features_reduced/<setting>_svd_explained_variance.json

    feature_engineering_report.json
        Summary of vocabulary sizes, matrix shapes, explained variance,
        sampling method used, and sample-vs-population length checks.

Optional class-balanced downsampling (--target-per-class):
    Selects an EXACT number of articles per label (e.g. 3000 real +
    3000 fake) using length-quantile-stratified sampling, so the
    downsampled subset's word-count distribution (and therefore its
    mean) closely tracks the full population's, rather than cherry-
    picking near-mean-length articles only. See report Section 8.

Optional fractional dev-sample mode (--sample-frac):
    Stratified-by-label random fraction of the data, for fast local
    iteration only -- output directory is auto-suffixed so it can't be
    mistaken for an official run. Mutually exclusive with
    --target-per-class.
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
# Vocabulary size controls
# -------------------------------------------------------------------
MIN_DF = 5           # a token must appear in >= 5 TRAINING documents
MAX_DF = 0.95        # drop tokens appearing in > 95% of TRAINING documents
MAX_FEATURES = {
    "bow_unigram": 15_000,
    "tfidf_unigram": 15_000,
    "tfidf_uni_bigram": 30_000,  # higher cap: bigram vocabulary is much larger
}

DEFAULT_TARGET_PER_CLASS = 3000


# -------------------------------------------------------------------
# Optional fractional dev-sample (fast iteration only)
# -------------------------------------------------------------------

def apply_dev_sample(dataframe: pd.DataFrame, sample_frac: float) -> pd.DataFrame:
    """
    Stratified (by label) random downsample of the FULL dataset, used
    only for fast iteration / debugging -- see --target-per-class for
    the length-representative, exact-count downsampling used for actual
    experiments.

    Uses an explicit per-group loop (rather than groupby().apply()) so
    the "label" column is guaranteed to stay in the output regardless
    of pandas version.
    """
    sampled_parts = [
        group.sample(frac=sample_frac, random_state=RANDOM_SEED)
        for _, group in dataframe.groupby("label")
    ]
    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled


# -------------------------------------------------------------------
# Optional exact-count, length-representative downsampling
# -------------------------------------------------------------------

def apply_length_balanced_sample(
    dataframe: pd.DataFrame,
    target_per_class: int,
    length_col: str = "clean_title_body",
    n_bins: int = 5,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    For each label class, take an EXACT `target_per_class` articles,
    selected via length-quantile-stratified sampling rather than plain
    random sampling.

    Why quantile-stratified instead of either (a) plain random sampling
    or (b) filtering to only near-mean-length articles:
      (a) Plain random sampling usually lands close to the population
          mean for n=3000, but "usually close" isn't the same as
          "verified close" -- and it does nothing to preserve the
          *shape* of the length distribution (spread, skew), only the
          mean tends to converge.
      (b) Filtering down to only articles whose length is close to the
          mean would directly bias the sample against naturally short
          or long articles, which are a real part of the population the
          model will eventually need to classify. That approach was
          considered and rejected -- it optimizes for a summary
          statistic while damaging real-world representativeness.
    Quantile-stratified sampling buckets each class's articles into
    `n_bins` equal-sized bins by word count, then samples proportionally
    from every bin. This keeps the sampled subset's word-count mean AND
    spread close to the full class's, by construction, rather than by
    luck.

    Returns a dataframe with exactly `target_per_class` rows per label
    (fewer only if a class has fewer than `target_per_class` rows to
    begin with, which is not the case for WELFake's complete-case file).
    """
    working = dataframe.copy()
    working["_word_count"] = (
        working[length_col].fillna("").str.split().apply(len)
    )

    sampled_parts = []
    for label_value, group in working.groupby("label"):
        group = group.copy()
        try:
            group["_length_bin"] = pd.qcut(
                group["_word_count"], q=n_bins, duplicates="drop"
            )
        except ValueError:
            # Degenerate case (e.g. all articles the same length, or too
            # few rows for n_bins) -- fall back to a single bin, which
            # reduces to plain random sampling within the class.
            group["_length_bin"] = 0

        bins = sorted(group["_length_bin"].unique(), key=str)
        n_bins_actual = len(bins)
        per_bin_target = target_per_class // n_bins_actual
        remainder = target_per_class - per_bin_target * n_bins_actual

        bin_samples = []
        for i, bin_value in enumerate(bins):
            bin_group = group[group["_length_bin"] == bin_value]
            n_take = per_bin_target + (1 if i < remainder else 0)
            n_take = min(n_take, len(bin_group))
            bin_samples.append(
                bin_group.sample(n=n_take, random_state=random_seed)
            )

        class_sample = pd.concat(bin_samples)

        # If some bins were too small to fill their quota, top up from
        # whatever's left in the class so we still hit target_per_class
        # exactly (as long as the class itself has enough rows).
        shortfall = target_per_class - len(class_sample)
        if shortfall > 0:
            remaining = group.drop(class_sample.index)
            top_up = remaining.sample(
                n=min(shortfall, len(remaining)), random_state=random_seed
            )
            class_sample = pd.concat([class_sample, top_up])

        sampled_parts.append(class_sample.drop(columns=["_length_bin"]))

    result = (
        pd.concat(sampled_parts)
        .drop(columns=["_word_count"])
        .reset_index(drop=True)
    )
    return result


def length_distribution_check(
    full_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    length_col: str = "clean_title_body",
) -> dict:
    """
    Compares word-count mean/std between the full population and the
    downsampled subset, per label, so the report can show -- not just
    claim -- that the sample's length distribution tracks the
    population's.
    """
    def stats_by_label(frame: pd.DataFrame) -> dict:
        word_counts = frame[length_col].fillna("").str.split().apply(len)
        out = {}
        for label_value, group_counts in word_counts.groupby(frame["label"]):
            out[str(label_value)] = {
                "mean": round(float(group_counts.mean()), 2),
                "std": round(float(group_counts.std()), 2),
                "n": int(len(group_counts)),
            }
        return out

    return {
        "population": stats_by_label(full_df),
        "sample": stats_by_label(sampled_df),
    }


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
    max_features = MAX_FEATURES[kind]
    if kind == "bow_unigram":
        return CountVectorizer(
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 1),
            lowercase=False,  # already lowercased in Part 2
            min_df=MIN_DF,
            max_df=MAX_DF,
            max_features=max_features,
        )
    if kind == "tfidf_unigram":
        return TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 1),
            lowercase=False,
            min_df=MIN_DF,
            max_df=MAX_DF,
            max_features=max_features,
        )
    if kind == "tfidf_uni_bigram":
        return TfidfVectorizer(
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 2),
            lowercase=False,
            min_df=MIN_DF,
            max_df=MAX_DF,
            max_features=max_features,
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

def run_pipeline(
    input_path: Path,
    output_dir: Path,
    svd_n_components: int,
    sample_frac: float | None = None,
    target_per_class: int | None = None,
) -> dict:
    if sample_frac is not None and target_per_class is not None:
        raise ValueError(
            "--sample-frac and --target-per-class are mutually exclusive "
            "-- pick one downsampling strategy."
        )

    print("Loading Part 2 preprocessed dataset...")
    df = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    print(f"Input rows: {len(df):,}")
    if sample_frac is None and target_per_class is None and len(df) != 62590:
        print(
            f"WARNING: expected 62,590 complete-case rows per project handoff "
            f"docs, got {len(df):,}. Continuing anyway."
        )

    for col in ["clean_title", "clean_body", "clean_title_body"]:
        df[col] = df[col].fillna("")

    sampling_method = "none"
    length_check = None
    is_dev_sample = False

    if target_per_class is not None:
        sampling_method = "length_stratified_downsample"
        class_counts = df["label"].value_counts()
        too_small = class_counts[class_counts < target_per_class]
        if len(too_small) > 0:
            raise ValueError(
                f"--target-per-class={target_per_class} exceeds available "
                f"rows for label(s): {too_small.to_dict()}. Lower the "
                "target or check the input file."
            )
        full_df_for_check = df
        df = apply_length_balanced_sample(df, target_per_class)
        length_check = length_distribution_check(full_df_for_check, df)
        print(
            f"*** LENGTH-STRATIFIED DOWNSAMPLE: {target_per_class:,} per "
            f"class -> {len(df):,} rows total. Word-count mean/std by "
            "class (population vs sample) recorded in the report -- "
            "check feature_engineering_report.json['length_distribution_check']. ***"
        )
        for label_value, pop_stats in length_check["population"].items():
            sample_stats = length_check["sample"][label_value]
            print(
                f"    label={label_value}: population mean={pop_stats['mean']} "
                f"(n={pop_stats['n']}) vs sample mean={sample_stats['mean']} "
                f"(n={sample_stats['n']})"
            )
    elif sample_frac is not None:
        sampling_method = "random_stratified_fraction"
        is_dev_sample = True
        original_rows = len(df)
        df = apply_dev_sample(df, sample_frac)
        print(
            f"*** DEV SAMPLE MODE: stratified sample_frac={sample_frac} -> "
            f"{len(df):,} rows (from {original_rows:,}). "
            "DO NOT use this run's numbers as final results. ***"
        )

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
        "sampling_method": sampling_method,
        "is_dev_sample": is_dev_sample,
        "target_per_class": target_per_class,
        "dev_sample_frac": sample_frac,
        "length_distribution_check": length_check,
        "input_rows": int(len(df)),
        "split_sizes": {
            name: int(len(ids)) for name, ids in split_by_name.items()
        },
        "svd_n_components_requested": svd_n_components,
        "vectorizer_config": {
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "max_features": MAX_FEATURES,
        },
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
    global MIN_DF, MAX_DF, MAX_FEATURES

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
    parser.add_argument(
        "--min-df",
        type=int,
        default=MIN_DF,
        help=f"Minimum document frequency for a token to be kept "
        f"(default: {MIN_DF}).",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=MAX_DF,
        help=f"Maximum document frequency (as a fraction of training docs) "
        f"for a token to be kept (default: {MAX_DF}).",
    )
    parser.add_argument(
        "--max-features-unigram",
        type=int,
        default=MAX_FEATURES["bow_unigram"],
        help="Vocabulary cap for bow_unigram and tfidf_unigram "
        f"(default: {MAX_FEATURES['bow_unigram']:,}).",
    )
    parser.add_argument(
        "--max-features-bigram",
        type=int,
        default=MAX_FEATURES["tfidf_uni_bigram"],
        help="Vocabulary cap for tfidf_uni_bigram "
        f"(default: {MAX_FEATURES['tfidf_uni_bigram']:,}).",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help=(
            "If set (e.g. 3000), downsample to EXACTLY this many articles "
            "per label class, using length-quantile-stratified sampling "
            "so the sample's word-count distribution matches the full "
            "population's. This is the recommended downsampling mode for "
            "actual experiments. Mutually exclusive with --sample-frac."
        ),
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help=(
            "If set (e.g. 0.2), run on a stratified-by-label random "
            "sample of this fraction of the input rows instead of the "
            "full dataset. FOR FAST ITERATION / DEBUGGING ONLY -- never "
            "use this for the group's final reported numbers. The "
            "output directory is automatically suffixed with "
            "'_devsample_<pct>pct'. Mutually exclusive with "
            "--target-per-class."
        ),
    )
    args = parser.parse_args()

    if args.sample_frac is not None and args.target_per_class is not None:
        raise ValueError(
            "--sample-frac and --target-per-class are mutually exclusive "
            "-- pick one downsampling strategy."
        )

    MIN_DF = args.min_df
    MAX_DF = args.max_df
    MAX_FEATURES = {
        "bow_unigram": args.max_features_unigram,
        "tfidf_unigram": args.max_features_unigram,
        "tfidf_uni_bigram": args.max_features_bigram,
    }

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file was not found: {input_path.resolve()}")

    output_dir = Path(args.output_dir)
    if args.sample_frac is not None:
        if not (0 < args.sample_frac < 1):
            raise ValueError("--sample-frac must be between 0 and 1 (exclusive).")
        pct = round(args.sample_frac * 100)
        output_dir = output_dir.parent / f"{output_dir.name}_devsample_{pct}pct"
        print(
            f"NOTE: --sample-frac={args.sample_frac} given. Output will be "
            f"written to '{output_dir}', separate from any full-data run."
        )
    elif args.target_per_class is not None:
        output_dir = (
            output_dir.parent
            / f"{output_dir.name}_balanced_{args.target_per_class}perclass"
        )
        print(
            f"NOTE: --target-per-class={args.target_per_class} given. "
            f"Output will be written to '{output_dir}'."
        )

    run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        svd_n_components=args.svd_components,
        sample_frac=args.sample_frac,
        target_per_class=args.target_per_class,
    )


if __name__ == "__main__":
    main()