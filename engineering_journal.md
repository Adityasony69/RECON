# Project Recon - Fake News Detection using Machine Learning

## Status

Phase 1 Completed ✅

Current Accuracy: **98.67%**

This project started as a basic machine learning project to understand how fake news detection works. The goal was not just to get good accuracy but to actually understand every step instead of copying code from tutorials.

---

# Why I Built This

I wanted to learn machine learning by building something real instead of watching videos all day. I also wanted to understand what actually happens behind every line of code.

This project detects whether a news article is Fake or Real using Machine Learning.

---

# Dataset

Dataset Used:

- Fake.csv
- True.csv

Total Articles: ~45,000

Labels:

- Fake = 0
- Real = 1

---

# What I Learned

## Python

- Functions
- Variables
- Lists
- String formatting
- Importing modules
- Reading error messages
- Debugging

---

## Pandas

- DataFrame
- Series
- head()
- tail()
- shape
- info()
- columns
- dtypes
- isnull()
- value_counts()
- concat()
- sample()
- reset_index()

---

## Data Cleaning

Learned how raw text is cleaned before training.

Steps:

- Convert text to lowercase
- Remove punctuation
- Remove numbers
- Remove special characters
- Remove extra spaces

---

## Feature Engineering

This was one of the biggest topics I learned.

Raw sentence

```
Donald Trump Won Election!!!
```

↓

Cleaned sentence

```
donald trump won election
```

↓

TF-IDF

```
0.12 0.00 0.61 0.42 ...
```

I learned that computers never understand English directly.

Everything becomes numbers before training.

---

## Machine Learning

Algorithm Used:

Logistic Regression

Workflow:

Dataset

↓

Cleaning

↓

TF-IDF

↓

Train/Test Split

↓

Model Training

↓

Prediction

↓

Accuracy

---

# Biggest Concepts I Understood

- Why we split train and test data
- Why labels are numbers
- Why TF-IDF is required
- Why model.fit() is needed
- Why model.predict() returns numbers
- Difference between training and testing
- Why feature engineering is important

---

# Mistakes I Made

These mistakes actually helped me understand more.

- Forgot to run previous notebook cells
- Got "pd is not defined"
- Got "model is not defined"
- Forgot to import LogisticRegression
- Forgot to import accuracy_score
- Mixed X_test with y_train
- Thought predict() would print output automatically
- Didn't understand f-strings
- Confused % inside strings with modulo operator
- Spent time trying to fix the date column before realizing it wasn't useful for this project

---

# Biggest Realizations

Machine Learning is not magic.

Most of the work is preparing the data.

Training the model was actually one line.

Understanding the data took much longer.

---

# Results

Model:

Logistic Regression

Accuracy:

98.67%

---

# Current Project Progress

## Phase 1 ✅

- Dataset collected
- Data explored
- Data cleaned
- Feature engineering completed
- TF-IDF completed
- Model trained
- Predictions generated
- Accuracy evaluated

Completed: 100%

---

# Future Roadmap

## Phase 2

Convert this notebook into a real web application.

Planned stack:

- FastAPI
- HTML/CSS
- JavaScript
- Load trained model (.pkl)
- User inputs news
- Model predicts Fake or Real

---

## Phase 3

Convert the project into an AI powered Rumor Verification System.

Future features:

- Embeddings
- Vector Database
- LLM Integration
- AI explanation
- Confidence score
- Similar historical news
- Better UI
- API deployment

Goal:

Instead of only saying

"Fake"

The system should explain

- Why it thinks the news is fake
- Similar old news
- Confidence percentage
- Supporting evidence

---

# Personal Notes

This project changed how I think.

At the beginning I was mostly asking what every line of code does.

By the end I started asking why we use a certain method and what happens behind the scenes.

I learned that understanding one line properly is better than copying fifty lines without knowing what they do.

This is my first complete Machine Learning project and it will be the base for a much bigger AI project in the future.

Project Recon is not finished.

Phase 1 is finished.

The next goal is to turn it into a real AI application that people can actually use.