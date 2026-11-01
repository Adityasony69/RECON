# Project Recon

A Machine Learning based Fake News Detection system built using Python and Scikit-learn, with an LLM that explains each prediction.

The model classifies news articles as **Fake** or **Real** using NLP (TF-IDF) and Logistic Regression, trained on the ISOT dataset (~38,000 articles after cleaning).

Current Status: **Phase 2 in progress**

---

## The interesting part: my 98.67% was not real

In Phase 1 I got **98.67% accuracy**. When I looked at which words the model trusted most, the top "real" word was `reuters`.

In the ISOT dataset, 99.8% of real articles come from Reuters and start with `WASHINGTON (Reuters) -`. The fake articles have their own giveaways: `Featured image via ...` credits, and contractions written as `don t` instead of `don’t`. The model was mostly learning which file an article came from.

What I changed:

- removed the Reuters dateline, image credits, links, @handles and source names (`preprocess.py`)
- made contractions the same in both files
- removed 630 empty articles and 5,725 duplicates (duplicates were landing in both train and test)
- used a stratified train/test split
- tested on a **completely different dataset** ([fake_or_real_news](https://github.com/lutzhamel/fake-news), real news from many outlets, not just Reuters)

## Results

| | ISOT test set | Different dataset (fake_or_real_news) |
|---|---|---|
| Phase 1 (original) | 98.67% | 64.15% |
| After removing leakage | 97.35% (F1: fake 0.970, real 0.976) | 61.05% (F1: fake 0.696, real 0.458) |

On the different dataset the model calls most real news fake (it only recognises 33% of real articles). Even after cleaning, the top "real" words are `said`, `wednesday`, `spokesman`, and the top "fake" words are `just`, `read`, `watch`, `didnt`. So the model has learned **wire-service writing style vs. blog writing style**, not truth.

Full numbers are in `reports/metrics.json` (created by `train.py`).

---

## Features

- Predicts whether an article is Fake or Real with a confidence score
- Same text cleaning for training and prediction (`preprocess.py`)
- LLM explanation of the prediction (RECON Insight, via Groq)
- Command line version
- Honest evaluation including a cross-dataset test

---

## Tech Stack

- **Language:** Python
- **ML:** Scikit-learn (TF-IDF Vectorizer, Logistic Regression), Pandas, NumPy, Joblib
- **LLM:** Groq API (OpenAI-compatible client)
- **Testing:** pytest

---

## Dataset

Not included in the repo (too large). Download and put in the `Dataset/` folder:

- `Fake.csv` and `True.csv`: [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- `fake_or_real_news.csv` (optional, for the cross-dataset test): [lutzhamel/fake-news](https://github.com/lutzhamel/fake-news/tree/master/data)

The trained model is already in `models/`, so `recon_ai.py` works without the dataset.

---

## How to Run

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

Optional, for LLM explanations, create a `.env` file:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Then:

```bash
python recon_ai.py   # command line version
python train.py      # retrain + evaluate (needs the dataset)
pytest               # run tests
```

---

## Project Structure

```
Recon/
│── recon_ai.py             command line version
│── recon_core.py           prediction + LLM explanation
│── preprocess.py           text cleaning (removes dataset leakage)
│── train.py                training + evaluation
│── Recon.ipynb             Phase 1 notebook (my learning process)
│── engineering_journal.md
│── models/                 trained model + vectorizer
│── reports/metrics.json    evaluation results
│── tests/
│── requirements.txt
```

---

## Limitations

- Trained only on 2016–2017 US political news.
- All real training news is from Reuters, so it judges writing style, not facts. Around 61% on other sources.
- The LLM explains the model's prediction. It does not fact-check.
- **Do not use this to decide whether news is true.**

---

## Future Plans

### Phase 2

- [x] User enters news, model predicts Fake or Real
- [ ] Simple web interface
- [ ] Train on more sources so the model learns more than Reuters style

### Phase 3

Upgrade Project Recon into an AI powered Rumor Verification Platform.

Planned features:

- Sentence Embeddings
- Vector Database
- Similar Historical News
- Evidence Based Responses

Instead of only predicting Fake or Real, the system will also explain **why** it reached that conclusion, using real evidence.

---

## What I Learned

During this project I learned:

- Data preprocessing
- Exploratory Data Analysis
- Text Cleaning
- TF-IDF
- Feature Engineering
- Train/Test Split
- Logistic Regression
- Machine Learning workflow
- Model Evaluation
- Debugging and understanding Python errors
- **Data leakage:** a high accuracy can be a bug, and you have to check what the model actually learned
- Testing on a different dataset to see if a model really generalises

---

## Author

**Aditya Soni**

MCA Hons.(AI & ML)

This project is part of my journey to learn AI by building real-world applications instead of only following tutorials.

License: MIT
