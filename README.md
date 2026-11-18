# AI/ML Fake News Detection

A complete, educational fake-news classification app built with Python, scikit-learn, TF-IDF text features, and Logistic Regression. It trains from labeled examples and serves predictions through a Flask web interface.

> **Important:** The model identifies writing patterns learned from its training data. It cannot verify facts or establish whether a story is true. Do not use its output as a substitute for fact-checking.

## What it includes

- Training and evaluation on labeled news articles
- Text preprocessing and TF-IDF unigram/bigram features
- Logistic Regression classifier
- Saved model (`model.joblib`)
- Flask web app and JSON prediction endpoint
- Basic tests

## Setup

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare a dataset

Create `data/articles.csv` with a `label` column and at least one of `text` or `title`:

```csv
label,title,text
real,Example headline,Article body goes here
fake,Another headline,Article body goes here
```

Labels must be `fake` or `real` (case-insensitive). The trainer combines `title` and `text` when both are present. Use a properly licensed, representative dataset with reliable labels. Keep the dataset out of Git if its license or size does not permit redistribution.

## Train and evaluate

```bash
python train.py
```

The script uses a stratified holdout split, prints precision/recall/F1 metrics, then saves `model.joblib`.

## Run the web app

```bash
python app.py
```

Open <http://127.0.0.1:5000>. The JSON API is `POST /predict` with `{"text":"article text"}`.

## Run tests

```bash
pytest
```

## Limitations

Model performance depends on dataset quality, class balance, and how closely future articles resemble training examples. A confidence score is not a probability that the article is true. Check claims against primary sources and trusted independent reporting.
