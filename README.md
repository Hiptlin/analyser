# 🧠 Sentiment Analyzer with Flask and MongoDB

This is a comprehensive sentiment analysis web application built using **Flask**, **Transformers**, and **MongoDB (Local)**. The app analyzes user-submitted reviews and classifies them using multiple sentiment models while storing the results in a local MongoDB database.

---

## 🚀 Features

- ✅ Sentiment analysis using:
  - **VADER** (Lexicon-based)
  - **Roberta (cardiffnlp/twitter-roberta-base-sentiment)**
  - **Multilingual BERT (nlptown)**
- ✅ Keyword Extraction using **KeyBERT**
- ✅ Summarization using **DistilBART**
- ✅ Toxicity Detection using **Toxic-BERT**
- ✅ Language detection & translation (e.g., Tamil to English)
- ✅ Data stored in **MongoDB (local instance)**
- ✅ Simple and interactive frontend

---

## 🗂️ Folder Structure

main/
│
├── app.py # Main Flask backend
├── templates/
│ └── index.html # Frontend HTML (Jinja2 Template)



| Stack            | Tool / Library                                 |
|------------------|------------------------------------------------|
| Backend          | Flask                                          |
| Database         | MongoDB (Local)                                |
| Sentiment Models | VADER, Roberta, BERT (via HuggingFace)         |
| Keyword Extract  | KeyBERT                                        |
| Toxicity Detect  | Unitary/toxic-bert                             |
| Summarization    | sshleifer/distilbart-cnn-12-6                  |
| Translation      | Helsinki-NLP MarianMT                          |
| Language Detect  | Custom (based on Unicode range)                |



🌐 How It Works
User enters a product review on the homepage.

Backend (Flask) receives the text and performs:

Sentiment analysis (VADER + Roberta)

Keyword extraction

Summary generation

Toxicity classification

Multilingual sentiment (if applicable)

The original + processed data is saved to the local MongoDB database.

Admin Page shows the stored reviews sorted by timestamp.



💬 Example Models Used
cardiffnlp/twitter-roberta-base-sentiment

nlptown/bert-base-multilingual-uncased-sentiment

unitary/toxic-bert

sshleifer/distilbart-cnn-12-6

Helsinki-NLP/opus-mt-ta-en


 Notes
MongoDB must be installed and running locally (localhost:27017)

The entire backend logic is in main/app.py

Frontend template is in main/templates/index.html
