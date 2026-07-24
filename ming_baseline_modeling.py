import numpy as np
import pandas as pd
from scipy import sparse

RANDOM_SEED = 123
INPUT_SETTINGS = ["title", "body", "combined"]
REPRESENTATIONS = ["tfidf_unigram", "tfidf_uni_bigram"]
LOGREG_C_VALUES = [0.001, 0.01, 0.1, 1, 10]
NB_ALPHA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
KNN_K_VALUES = [3, 5, 7, 9, 15, 25]

BASELINE_SETTING = "combined"
BASELINE_REPRESENTATION = "tfidf_uni_bigram"
BASELINE_MODEL = "logistic_regression"


def load_split_table(split_file):
    split_df = pd.read_csv(split_file)
    tables = {}
    for split_name in ["train", "val", "test"]:
        rows = split_df[split_df["split"] == split_name].sort_values("article_id")
        tables[split_name] = rows.reset_index(drop=True)

    return tables


def load_sparse_features(features_dir, setting_name, representation):
    prefix = f"{setting_name}_{representation}"
    matrices = {}
    for split_name in ["train", "val", "test"]:
        path = features_dir / f"{prefix}_{split_name}.npz"
        matrices[split_name] = sparse.load_npz(path)
    return matrices


