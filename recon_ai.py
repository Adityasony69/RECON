from recon_core import MIN_WORDS, explain, predict

print("=" * 50)
print("        PROJECT RECON")
print("=" * 50)

news = input("\nPaste a news article:\n\n> ").strip()

if len(news.split()) < MIN_WORDS:
    print(f"Please paste a full article (at least {MIN_WORDS} words).")
    raise SystemExit

result, confidence = predict(news)

print("\n" + "=" * 50)
print(f"Prediction : {result}")
print(f"Confidence : {confidence:.2f}%")
print("=" * 50)

print("\n" + "=" * 50)
print("RECON Insight")
print("=" * 50)

try:
    explanation = explain(news, result, confidence)
    print(explanation or "LLM explanation unavailable (GROQ_API_KEY not set).")
except Exception as e:
    print("LLM explanation unavailable.")
    print(e)
