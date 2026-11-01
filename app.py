"""Web demo for Project Recon.  Run:  python app.py"""

import gradio as gr

from recon_core import MIN_WORDS, explain, predict

NOTE = (
    "This model was trained on the ISOT dataset, where all real news is from Reuters. "
    "It recognises writing style (wire-service vs. blog), not facts, and gets about 61% "
    "on news from other sources. Don't use it to decide whether something is true."
)


def analyse(news):
    news = (news or "").strip()
    if len(news.split()) < MIN_WORDS:
        return f"Please paste a full article (at least {MIN_WORDS} words).", ""

    result, confidence = predict(news)
    verdict = f"## {result}  ({confidence:.1f}% confidence)"

    try:
        explanation = explain(news, result, confidence) or "LLM explanation unavailable (no API key set)."
    except Exception as e:
        explanation = f"LLM explanation unavailable: {e}"

    return verdict, explanation


with gr.Blocks(title="Project Recon") as demo:
    gr.Markdown("# Project Recon\nPaste a news article to see what the model thinks and why.")
    article = gr.Textbox(label="News article", lines=12, placeholder="Paste the full article text here...")
    button = gr.Button("Analyse", variant="primary")
    verdict = gr.Markdown()
    explanation = gr.Textbox(label="RECON Insight (LLM explanation)", lines=6)
    gr.Markdown(f"**Limitation:** {NOTE}")
    button.click(analyse, inputs=article, outputs=[verdict, explanation])

if __name__ == "__main__":
    demo.launch()
