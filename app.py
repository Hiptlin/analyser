from flask import Flask, render_template, request, jsonify
from datetime import datetime
from pymongo import MongoClient
import os
import pandas as pd
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
from keybert import KeyBERT
from transformers import MarianMTModel, MarianTokenizer

# ───── Flask Setup ─────
app = Flask(__name__)

# ───── NLTK Setup ─────
nltk.download("vader_lexicon")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")

# ───── MongoDB ─────
client = MongoClient("mongodb://localhost:27017/")
db = client["SentimentDB"]
collection = db["ReviewAnalysis"]

# ───── Model Setup ─────
sia = SentimentIntensityAnalyzer()
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_mod = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
kw_model = KeyBERT()
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")
multi_sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def polarity_scores_roberta(text: str) -> dict:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    probs = softmax(roberta_mod(**encoded).logits[0].detach().numpy())
    return dict(zip(["roberta_neg", "roberta_neu", "roberta_pos"], map(float, probs)))

def analyse_single_review(text: str) -> dict:
    v_raw = sia.polarity_scores(text)
    v_label = "Positive" if v_raw["compound"] > 0.05 else "Negative" if v_raw["compound"] < -0.05 else "Neutral"
    r_raw = polarity_scores_roberta(text)
    r_label = max(r_raw, key=r_raw.get).split("_")[-1].capitalize()
    return {
        **{f"vader_{k}": v for k, v in v_raw.items()},
        **r_raw,
        "vader_label": v_label,
        "roberta_label": r_label,
    }

def extract_keywords(text: str, top_n: int = 5):
    try:
        return [kw[0] for kw in kw_model.extract_keywords(text, top_n=top_n)]
    except:
        return ["[Extraction error]"]

def summarize_review(text: str):
    if len(text.split()) < 30:
        return text
    try:
        return summarizer(text, max_length=50, min_length=20, do_sample=False)[0]['summary_text'].strip()
    except:
        return "[Summarization error]"

def detect_toxicity(text: str):
    try:
        result = toxicity_model(text)[0]
        return {"label": result['label'], "score": round(result['score'], 4)}
    except:
        return {"label": "Error", "score": 0.0}

def multilingual_sentiment(text: str):
    try:
        return multi_sentiment(text)[0]
    except:
        return {"label": "Error", "score": 0.0}

def translate_to_english(text: str, src_lang: str = "ta"):
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        tokens = tokenizer(text, return_tensors="pt", truncation=True)
        translation = model.generate(**tokens)
        return tokenizer.decode(translation[0], skip_special_tokens=True)
    except:
        return "[Translation failed]"

def detect_lang(text):
    if any(ord(c) > 128 for c in text):
        return "ta"  # assume Tamil for example
    return "en"

def extended_analysis(text: str) -> dict:
    lang = detect_lang(text)
    translated = translate_to_english(text, lang) if lang != "en" else text
    base = analyse_single_review(translated)
    keywords = extract_keywords(translated)
    summary = summarize_review(translated)
    toxicity = detect_toxicity(translated)
    multi_sent = multilingual_sentiment(translated)
    return {
        **base,
        "translated_text": translated,
        "detected_language": lang,
        "keywords": keywords,
        "summary": summary,
        "toxicity_label": toxicity['label'],
        "toxicity_score": toxicity['score'],
        "multi_sentiment_label": multi_sent['label'],
        "multi_sentiment_score": round(multi_sent['score'], 4),
    }

def store_to_mongodb(original_text: str, analysis: dict):
    doc = {
        "review_text": original_text,
        "translated_text": analysis.get("translated_text", ""),
        "language": analysis.get("detected_language", "en"),
        "vader_label": analysis.get("vader_label", "Unknown"),
        "analysis": analysis,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(doc)

@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    if request.method == "POST":
        review_text = request.form["review_text"]
        result = extended_analysis(review_text)
        store_to_mongodb(review_text, result)
        label = result["vader_label"]
    return render_template("index.html", mode="index", vader_label=label)

@app.route("/admin")
def admin():
    data = list(collection.find().sort("timestamp", -1))
    return render_template("index.html", mode="admin", reviews=data)

if __name__ == "__main__":
    app.run(debug=True)
