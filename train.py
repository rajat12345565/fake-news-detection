from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/articles.csv")
MODEL_PATH = Path("model.joblib")


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.Series, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Create a CSV with a label column and text and/or title columns."
        )

    data = pd.read_csv(path)
    if "label" not in data.columns:
        raise ValueError("CSV must include a 'label' column.")

    text_columns = [column for column in ("title", "text") if column in data.columns]
    if not text_columns:
        raise ValueError("CSV must include at least one of 'title' or 'text'.")

    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["label"].isin(["fake", "real"])].copy()
    if data["label"].nunique() != 2:
        raise ValueError("Dataset must contain both 'fake' and 'real' labels.")

    pieces = [data[column].fillna("").astype(str).str.strip() for column in text_columns]
    text = pieces[0]
    for piece in pieces[1:]:
        text = text + " " + piece
    keep = text.str.strip().ne("")
    text, labels = text[keep], data.loc[keep, "label"]
    if labels.value_counts().min() < 2:
        raise ValueError("Each label needs at least two non-empty examples for a stratified split.")
    return text, labels


def create_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def main() -> None:
    texts, labels = load_dataset()
    train_text, test_text, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = create_pipeline()
    model.fit(train_text, train_labels)
    predictions = model.predict(test_text)
    print(classification_report(test_labels, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
