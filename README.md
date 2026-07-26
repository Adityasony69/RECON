# Project Recon

A Machine Learning based Fake News Detection system built using Python and Scikit-learn.

This project classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) and a Logistic Regression model trained on more than 44,000 news articles.

Current Status: **Phase 1 Completed ✅**

---

## Features

- Detects whether a news article is Fake or Real
- Text preprocessing pipeline
- TF-IDF feature extraction
- Logistic Regression classifier
- Trained on 44k+ news articles
- Accuracy of **98.67%**

---

## Tech Stack

### Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

### Machine Learning

- Logistic Regression
- TF-IDF Vectorizer

---

## Dataset

Dataset contains two CSV files.

- Fake.csv
- True.csv

The data is merged, cleaned and converted into numerical features before training the model.

---

## Project Workflow

```text
Dataset

↓

Data Cleaning

↓

Text Preprocessing

↓

TF-IDF Feature Engineering

↓

Train/Test Split

↓

Model Training

↓

Prediction

↓

Evaluation
```

---

## Model Performance

| Metric | Value |
|---------|--------|
| Algorithm | Logistic Regression |
| Accuracy | **98.67%** |
| Dataset Size | 44,000+ Articles |

---

## Project Structure

```
Project-Recon/

│── notebook.ipynb

│── Fake.csv

│── True.csv

│── README.md

│── ENGINEERING_JOURNAL.md

│── requirements.txt

│── fake_news_model.pkl      (Coming Soon)

│── tfidf_vectorizer.pkl     (Coming Soon)
```

---

## Future Plans

### Phase 2

Convert the model into a web application.

Planned features:

- User enters news
- Model predicts Fake or Real
- FastAPI backend
- Simple responsive frontend

---

### Phase 3

Upgrade Project Recon into an AI powered Rumor Verification Platform.

Planned features:

- Sentence Embeddings
- Vector Database
- LLM Integration
- Confidence Score
- AI Explanation
- Similar Historical News
- Evidence Based Responses

Instead of only predicting Fake or Real, the system will also explain **why** it reached that conclusion.

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

---

## Next Goal

The current model works completely offline using classical Machine Learning.

The next objective is to transform Project Recon into a real AI application where users can verify news and rumors through a web interface powered by Machine Learning and Large Language Models.

---

## Author

**Aditya Soni**

MCA Hons.(AI & ML)

This project is part of my journey to learn AI by building real-world applications instead of only following tutorials.