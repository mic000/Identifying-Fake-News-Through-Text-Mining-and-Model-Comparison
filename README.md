# Identifying Fake News Through Text Mining and Model Comparison

A machine learning project for CSC503 that detects fake news articles using only the text content of the article (title and body), without relying on author name or source. We build a full text-mining pipeline and compare several classical ML models on the [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) dataset.

> **Status:** 🚧 Project setup — implementation has not started yet. This README will be expanded as the project progresses.

---

## Team

| Name 
| Wenbo Wu 
| Arunkumar Krishnamoorthy
| Yashwanthkrishna Nagharaj 
| Etta Zhang 
| Ming Chen 

---

## Motivation

Online blogs and self-media are widely read, especially by younger audiences, and many posts contain misinformation that can shape opinions on politics, history, and current events. Fake political news has been shown to affect elections and reduce public trust in journalism. Because the volume of daily content is far too large for manual review, automated detection tools are needed.

This project builds one such tool: given a news article (title + body), the model predicts whether it is **real (1)** or **fake (0)** using only the words in the article itself.

---

## Dataset

We use the **WELFake** dataset (Verma et al., 2021), hosted on Kaggle and mirrored on Zenodo and Hugging Face.

- **Size:** 72,134 articles (35,028 real + 37,106 fake — nearly balanced)
- **Features:** `title`, `text` (body), `label` (1 = real, 0 = fake)
- **Construction:** Merged from four older datasets (Kaggle, McIntire, Reuters, BuzzFeed Political) so no single source dominates, reducing overfitting risk.

**Backup plan:** If WELFake has unfixable problems, we fall back to the [ISOT Fake News dataset](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php) from the University of Victoria.

### Known Risk: Label Leakage

Earlier fake-news datasets sometimes contain source markers (e.g., real articles starting with "Reuters") that let a model predict the label without reading the content. To guard against this, we will:

1. Run EDA to count the most common words per class.
2. Train a quick Logistic Regression as a sanity model and inspect its top-weighted words.
3. If top words look like source markers rather than content, remove them during preprocessing.

---

## Research Questions

1. Which words appear most often in fake versus real articles, and do they look like meaningful content signals or dataset artifacts?
2. Which of our methods works best on simple Bag-of-Words or TF-IDF features?
3. How much does PCA help KNN, given that Ahmed et al. (2017) found KNN can drop to 47% accuracy on high-dimensional features?
4. Do different models make the same mistakes, or different ones?

---

## Approach

### Input settings
We evaluate every model under three input configurations to see how much the body adds beyond the headline:
- Title only
- Body only
- Title + body

### Feature extraction
- **Bag-of-Words** (count vs. binary)
- **TF-IDF**
- **N-gram comparison:** unigrams vs. unigrams + bigrams

### Models
| Category | Model |
|---|---|
| Baseline | Logistic Regression |
| Classical | Naive Bayes, K-Nearest Neighbors (KNN), KNN + PCA |
| Advanced | Linear SVM, Decision Tree, Random Forest |

A more complex model is considered worth keeping only if it gives a meaningful improvement in F1 over the Logistic Regression baseline.

### Evaluation
- **Metrics:** Accuracy, Precision, Recall, **F1 (primary)**
- **Split:** 80% train / 10% validation / 10% test, with a fixed random seed
- **Protocol:** Hyperparameter tuning on validation only; the test set is touched exactly once at the end
- **Error analysis:** Confusion matrices, plus separate discussion of false positives (real flagged as fake) and false negatives (fake missed)

---

## Project Timeline

| Date | Stage |
|---|---|
| **Jun 15** | Kickoff meeting — discuss EDA findings, agree on cleaning rules |
| **Jun 15 – 21** | Text cleaning and preprocessing |
| **Jun 22** | Feature extraction (BoW vs. TF-IDF, unigrams vs. bigrams, 3 input settings) |
| **Jun 29** | Initial model comparison on validation set; drop weak models |
| **Jul 6** | Final training, test-set evaluation, error analysis |
| **Jul 7+** | Report writing, slides, presentation practice |

We aim to finish all technical work at least one week before the final presentation.

---

## Planned Repository Structure

```
.
├── data/                  # Raw and processed data (not committed; see download instructions)
│   ├── raw/
│   └── processed/
├── notebooks/             # Jupyter notebooks for EDA and experiments
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_features.ipynb
│   └── 04_models.ipynb
├── src/
│   ├── data/              # Loading and splitting
│   ├── preprocessing/     # Text cleaning pipeline
│   ├── features/          # BoW, TF-IDF, PCA
│   ├── models/            # Logistic Regression, NB, KNN, SVM, Decision Tree, RF
│   └── evaluation/        # Metrics, confusion matrices, error analysis
├── results/               # Saved metrics, plots, confusion matrices
├── report/                # Final report and figures
├── requirements.txt
└── README.md
```

---

## Task Breakdown

### Part 1 — Dataset, Data Quality & EDA (Wenbo)
- Load and inspect the WELFake dataset
- Analyze class distribution, missing values, duplicates
- Identify potential label leakage
- Produce visualizations

### Part 2 — Text Preprocessing Pipeline (Arunkumar)
- Clean and standardize raw text
- Handle title and body consistently
- Support all three input configurations
- Ensure reproducibility across experiments

### Part 3 — Feature Engineering & Dimensionality Reduction (Yashwanthkrishna)
- Implement BoW and TF-IDF
- Build feature matrices for all input settings
- Apply PCA and evaluate its impact on high-dimensional text features

### Part 4 — Baseline & Classical Models (Etta)
- Logistic Regression (baseline), Naive Bayes, KNN
- Hyperparameter tuning on validation set
- Evaluate KNN sensitivity to dimensionality and PCA

### Part 5 — Advanced Models, Evaluation & Final Analysis (Ming)
- Linear SVM, Decision Tree, Random Forest
- Full evaluation (Accuracy, Precision, Recall, F1)
- Confusion matrices and error analysis
- Compare every model against the Logistic Regression baseline
- Summarize and interpret final results

---

## Getting Started

> Setup instructions will be added once the code is in place.

Planned tooling: Python 3.10+, scikit-learn, pandas, numpy, matplotlib, seaborn, nltk.

```bash
# Placeholder
git clone <repo-url>
cd fake-news-detection
pip install -r requirements.txt
```

---

## References

1. Verma, P. K., Agrawal, P., Amorim, I., & Prodan, R. (2021). *WELFake: Word Embedding Over Linguistic Features for Fake News Detection.* IEEE Transactions on Computational Social Systems, 8(4), 881–893.
2. Ahmed, H., Traore, I., & Saad, S. (2017). *Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques.* ISDDC 2017, LNCS vol. 10618, pp. 127–138. Springer, Cham.
3. Alshuwaier, F. A., & Alsulaiman, F. A. (2025). *Fake News Detection Using Machine Learning and Deep Learning Algorithms: A Comprehensive Review and Future Perspectives.* Computers, 14(9), 394.
4. scikit-learn documentation — text feature extraction and classification models.
5. CSC503 course materials.

---

## Course

CSC503 — University of Victoria. For questions on feature design, model selection, or data leakage, we plan to consult Professor Nishant Mehta during office hours.
