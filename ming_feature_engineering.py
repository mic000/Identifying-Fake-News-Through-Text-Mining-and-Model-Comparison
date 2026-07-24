"""
Part 3: Downsampling + Feature Engineering (Bag-of-Words / TF-IDF)

Input file (already cleaned by Wenbo + Arunkumar):
    WELFake_part2_preprocessed.csv.gz
    Required columns: article_id, label, clean_title, clean_body, clean_title_body

  1. Downsamples the data (e.g. 1500 real + 1500 fake), stratified by label, fixed random
     seed.
  2. Splits the downsampled data into train/val/test (80/10/10),
     stratified by label, same fixed seed -- saved to
     split_assignment.csv so everyone on the team trains/tunes/tests
     on the exact same rows.
  3. Builds Bag-of-Words and TF-IDF features (unigram, and
     unigram+bigram) for each input setting: title, body, title+body.
     Every vectorizer is fit on the TRAIN split only, then applied to
     val/test, so no vocabulary information leaks from val/test into
     training.

Output (under --output-dir):
    split_assignment.csv
    features/<setting>_<representation>_<split>.npz
    features/<setting>_<representation>_vocab.json
    feature_summary.csv

Example:
    python part3_downsample_and_features.py \
        --input processed/WELFake_part2_preprocessed.csv.gz \
        --output-dir part3_output \
        --target-per-class 1500 \
        --max-features-unigram 3000 \
        --max-features-bigram 6000 \
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

RANDOM_SEED = 123
INPUT_SETTINGS = {
    "title": "clean_title",
    "body": "clean_body",
    "combined": "clean_title_body",
}

# custom pattern
TOKEN_PATTERN = r"(?u)[a-z']+"
MIN_DF = 5          # words keep once appears over this many documents
MAX_DF = 0.95       # words drop once over this fraction


def downsample(df, target_per_class):
    class_counts = df["label"].value_counts()
    for label_value, count in class_counts.items():
        if count < target_per_class:
            return df

    parts = []
    for label_value, group in df.groupby("label"):
        parts.append(group.sample(n=target_per_class, random_state=RANDOM_SEED))

    return pd.concat(parts).reset_index(drop=True)


def make_split(df):
    train_df, rest_df = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.50, stratify=rest_df["label"], random_state=RANDOM_SEED
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    split_assignment = pd.concat([train_df, val_df, test_df])
    split_assignment = split_assignment[["article_id", "label", "split"]]
    return split_assignment.sort_values("article_id")


def build_vectorizer(kind, max_features):
    if kind == "bow_unigram":
        return CountVectorizer(
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 1),
            lowercase=False,
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


def run_pipeline(input_path, output_dir, target_per_class, max_features_unigram, max_features_bigram):
    df = pd.read_csv(input_path)
    for col in ["clean_title", "clean_body", "clean_title_body"]:
        df[col] = df[col].fillna("")

    print(f"Input rows: {len(df):,}")
    df = downsample(df, target_per_class)
    print(f"Rows after down-sampling: {len(df):,}")
    print(f"\ndataset has been split train/val/test with random_state = {RANDOM_SEED}")
    split_assignment = make_split(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_assignment.to_csv(output_dir/"split_assignment.csv", index=False)

    df_by_id = df.set_index("article_id")
    ids_by_split = {
        name: split_assignment.loc[split_assignment["split"] == name, "article_id"]
        for name in ["train", "val", "test"]
    }
    print("Split sizes:", {name: len(ids) for name, ids in ids_by_split.items()})

    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for setting_name, column in INPUT_SETTINGS.items():
        print(f"\n=== Input setting: {setting_name} ({column}) ===")

        train_text = df_by_id.loc[ids_by_split["train"], column]
        val_text = df_by_id.loc[ids_by_split["val"], column]
        test_text = df_by_id.loc[ids_by_split["test"], column]

        for kind in ["tfidf_unigram", "tfidf_uni_bigram"]:
            max_features  = max_features_bigram if "bigram" in kind else max_features_unigram
            vectorizer = build_vectorizer(kind, max_features)

            X_train = vectorizer.fit_transform(train_text)
            X_val = vectorizer.transform(val_text)
            X_test = vectorizer.transform(test_text)

            prefix = f"{setting_name}_{kind}"
            sparse.save_npz(features_dir / f"{prefix}_train.npz", X_train)
            sparse.save_npz(features_dir / f"{prefix}_val.npz", X_val)
            sparse.save_npz(features_dir / f"{prefix}_test.npz", X_test)

            vocabulary = vectorizer.get_feature_names_out().tolist()
            with open(features_dir / f"{prefix}_vocab.json", "w", encoding="utf-8") as f:
                json.dump(vocabulary, f, ensure_ascii=False, indent=2)

            print(f"  {kind}: vocab size = {len(vocabulary):,} | train shape = {X_train.shape}")

            summary_rows.append(
                {
                    "input_setting": setting_name,
                    "representation": kind,
                    "vocab_size": len(vocabulary),
                    "train_rows": X_train.shape[0],
                    "val_rows": X_val.shape[0],
                    "test_rows": X_test.shape[0],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "feature_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nDone. Feature summary saved to: {summary_path.resolve()}")
    print(f"Features saved under: {features_dir.resolve()}")
    print(f"Split assignment saved to: {(output_dir / 'split_assignment.csv').resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Part 3: downsampling + Bag-of-Words/TF-IDF feature engineering for WELFake."
    )
    parser.add_argument("--input",
                        required=True, help="Path to WELFake_part2_preprocessed.csv or .csv.gz")
    parser.add_argument("--output-dir",
                        default="part3_output", help="Folder where output files will be saved")
    parser.add_argument("--target-per-class",
                        type=int, required=True,
                        help="Exact number of articles to keep per label, e.g. 1500 for 1500 real + 1500 fake")
    parser.add_argument("--max-features-unigram",
                        type=int, default=3000,
                        help="Vocabulary size cap for bow_unigram and tfidf_unigram (default: 3000)")
    parser.add_argument("--max-features-bigram",
                        type=int, default=6000,
                        help="Vocabulary size cap for tfidf_uni_bigram (default: 6000)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file was not found: {input_path.resolve()}")

    output_dir = Path(args.output_dir) / f"{args.target_per_class}perclass"

    run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        target_per_class=args.target_per_class,
        max_features_unigram=args.max_features_unigram,
        max_features_bigram=args.max_features_bigram,
    )


if __name__ == "__main__":
    main()