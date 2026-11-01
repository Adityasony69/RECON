"""Train the Recon model and write an honest evaluation.

Run:  python train.py
Needs Dataset/Fake.csv and Dataset/True.csv (see README for the download link).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from preprocess import clean_text

ROOT = Path(__file__).parent
DATA = ROOT / "Dataset"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def load_data():
    fake = pd.read_csv(DATA / "Fake.csv")
    true = pd.read_csv(DATA / "True.csv")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    print("articles loaded:", len(df))

    df["clean_text"] = df["text"].apply(clean_text)

    # 630 fake articles have no text at all, only a title
    df = df[df["clean_text"].str.split().str.len() >= 20]
    print("after dropping empty / very short articles:", len(df))

    # duplicates would end up in both train and test and inflate the score
    df = df.drop_duplicates(subset="clean_text")
    print("after dropping duplicates:", len(df))

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    return vectorizer, model, predictions


def external_test(vectorizer, model, seen_texts):
    """Test on a different dataset the model never saw (McIntire fake_or_real_news).

    ISOT's real news is all Reuters, so a high ISOT score can just mean
    "detects Reuters style". This checks if the model works on other sources.
    """
    path = DATA / "fake_or_real_news.csv"
    if not path.exists():
        print("\n(external test skipped: Dataset/fake_or_real_news.csv not found)")
        return None

    ext = pd.read_csv(path)
    ext["clean_text"] = ext["text"].apply(clean_text)
    ext = ext[ext["clean_text"].str.split().str.len() >= 20]
    ext = ext[~ext["clean_text"].isin(seen_texts)].drop_duplicates(subset="clean_text")
    y = (ext["label"] == "REAL").astype(int)

    predictions = model.predict(vectorizer.transform(ext["clean_text"]))
    accuracy = accuracy_score(y, predictions)
    report = classification_report(y, predictions, target_names=["fake", "real"], output_dict=True)

    print("\n--- external test (fake_or_real_news, never seen in training) ---")
    print("articles:", len(ext))
    print("accuracy:", round(accuracy, 4))
    print(classification_report(y, predictions, target_names=["fake", "real"], digits=4))
    print(confusion_matrix(y, predictions))

    return {
        "articles": len(ext),
        "accuracy": round(accuracy, 4),
        "f1_fake": round(report["fake"]["f1-score"], 4),
        "f1_real": round(report["real"]["f1-score"], 4),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }


def top_words(vectorizer, model, n=20):
    words = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(model.coef_[0])
    return {"real": words[order[-n:][::-1]].tolist(), "fake": words[order[:n]].tolist()}


def main():
    df = load_data()
    print(df["label"].value_counts().rename({0: "fake", 1: "real"}).to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer, model, predictions = train_and_evaluate(X_train, X_test, y_train, y_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["fake", "real"], output_dict=True)
    matrix = confusion_matrix(y_test, predictions)
    words = top_words(vectorizer, model)

    print("\naccuracy:", round(accuracy, 4))
    print(classification_report(y_test, predictions, target_names=["fake", "real"], digits=4))
    print("confusion matrix (rows = actual fake/real, cols = predicted fake/real):")
    print(matrix)
    print("\nwords pushing towards REAL:", words["real"])
    print("words pushing towards FAKE:", words["fake"])

    external = external_test(vectorizer, model, set(df["clean_text"]))

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "fake_news_model.pkl")
    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.pkl")

    metrics = {
        "articles_used": len(df),
        "test_size": len(y_test),
        "accuracy": round(accuracy, 4),
        "f1_fake": round(report["fake"]["f1-score"], 4),
        "f1_real": round(report["real"]["f1-score"], 4),
        "confusion_matrix": matrix.tolist(),
        "top_words": words,
        "external_test": external,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nsaved models/ and reports/metrics.json")


if __name__ == "__main__":
    main()
