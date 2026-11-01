"""Prediction + LLM explanation, used by the command line version (recon_ai.py)."""

import os
from pathlib import Path

import joblib
from dotenv import load_dotenv

from preprocess import clean_text

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

model = joblib.load(ROOT / "models" / "fake_news_model.pkl")
vectorizer = joblib.load(ROOT / "models" / "tfidf_vectorizer.pkl")

MIN_WORDS = 20


def predict(news):
    """Return ("FAKE" or "REAL", confidence in %)."""
    news_vector = vectorizer.transform([clean_text(news)])
    prediction = model.predict(news_vector)[0]
    confidence = model.predict_proba(news_vector)[0].max() * 100
    result = "REAL" if prediction == 1 else "FAKE"
    return result, confidence


def explain(news, result, confidence):
    """Ask the LLM to explain the prediction. Returns None if no API key is set."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    prompt = f"""
You are RECON Insight.

A machine learning model has analysed a news article.

Prediction: {result}
Confidence: {confidence:.2f}%

Article:
{news[:4000]}

Explain why the model may have predicted this result.

Do not claim the article is factually true or false.

Mention that the explanation is based on statistical language patterns rather than live fact checking.

Keep the answer below 120 words.
"""

    # Groq retires models over time (llama-3.1-8b-instant is gone), so this can be changed in .env
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model_name else {}

    response = client.chat.completions.create(
        model=model_name,
        **extra,
        messages=[
            {"role": "system", "content": "You explain machine learning predictions. Never fact-check news."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content
