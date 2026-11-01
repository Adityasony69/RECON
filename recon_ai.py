import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("=" * 50)
print("        PROJECT RECON")
print("=" * 50)

news = input("\nPaste a news article:\n\n> ").strip()

if not news:
    print("No article entered.")
    exit()

news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)
confidence = model.predict_proba(news_vector)

result = "REAL" if prediction[0] == 1 else "FAKE"
confidence_score = max(confidence[0]) * 100

print("\n" + "=" * 50)
print(f"Prediction : {result}")
print(f"Confidence : {confidence_score:.2f}%")
print("=" * 50)

prompt = f"""
You are RECON Insight.

A machine learning model has analysed a news article.

Prediction: {result}
Confidence: {confidence_score:.2f}%

Article:
{news}

Explain why the model may have predicted this result.

Do not claim the article is factually true or false.

Mention that the explanation is based on statistical language patterns rather than live fact checking.

Keep the answer below 120 words.
"""

try:

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You explain machine learning predictions. Never fact-check news."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=180
    )

    print("\n" + "=" * 50)
    print("RECON Insight")
    print("=" * 50)
    print(response.choices[0].message.content)

except Exception as e:

    print("\n" + "=" * 50)
    print("RECON Insight")
    print("=" * 50)
    print("LLM explanation unavailable.")
    print(e)