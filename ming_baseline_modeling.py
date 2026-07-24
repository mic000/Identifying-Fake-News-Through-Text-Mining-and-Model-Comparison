import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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


def compute_svd_features(tfidf_matrices, n_components):
    X_train = tfidf_matrices["train"]
    safe_n_components = min(n_components, X_train.shape[0] - 1, X_train.shape[1] - 1)
    svd = TruncatedSVD(n_components=safe_n_components, random_state=RANDOM_SEED)
    reduced = {
        "train": svd.fit_transform(X_train),
        "val": svd.transform(tfidf_matrices["val"]),
        "test": svd.transform(tfidf_matrices["test"]),
    }

    return reduced, safe_n_components


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def compute_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "true_negative": int(matrix[0, 0]),  # real predicted real
        "false_positive": int(matrix[0, 1]),  # real predicted fake
        "false_negative": int(matrix[1, 0]),  # fake predicted real
        "true_positive": int(matrix[1, 1]),  # fake predicted fake
    }


def tune_logistic_regression(X_train, y_train, X_val, y_val):
    best_val_f1 = -1
    best_model = None
    best_params = None

    for c_value in LOGREG_C_VALUES:
        model = LogisticRegression(C=c_value, max_iter=1000, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_params = {"C": c_value}

    return best_model, best_params, round(float(best_val_f1), 4)


def tune_naive_bayes(X_train, y_train, X_val, y_val):
    best_val_f1 = -1
    best_model = None
    best_params = None

    for alpha_value in NB_ALPHA_VALUES:
        model = MultinomialNB(alpha=alpha_value)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_params = {"alpha": alpha_value}

    return best_model, best_params, round(float(best_val_f1), 4)


def tune_knn(X_train, y_train, X_val, y_val):
    best_val_f1 = -1
    best_model = None
    best_params = None

    for k_value in KNN_K_VALUES:
        model = KNeighborsClassifier(n_neighbors=k_value)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_params = {"n_neighbors": k_value}

    return best_model, best_params, round(float(best_val_f1), 4)


def run_one_config(model_name, tune_function, X_train, y_train, X_val, y_val, X_test, y_test, test_article_ids,
                   config_tag, predictions_dir,):
    best_model, best_params, val_f1 = tune_function(X_train, y_train, X_val, y_val)
    test_pred = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_pred)
    test_confusion = compute_confusion_matrix(y_test, test_pred)

    print(
        f"      best params = {best_params} | val F1 = {val_f1} "
        f"| test F1 = {test_metrics['f1']}"
    )

    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(
        {"article_id": test_article_ids,
         "true_label": y_test,
         "predicted_label": test_pred,}
    )
    predictions_df.to_csv(
        predictions_dir / f"{config_tag}_{model_name}_test_predictions.csv",
        index=False,
    )

    return {
        "model": model_name,
        "best_params": best_params,
        "val_f1": val_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "confusion_matrix": test_confusion,
    }