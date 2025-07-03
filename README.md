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



## 🌐 How It Works

1. **User enters a product review** on the homepage.
2. **Backend (Flask)** receives the text and performs:
   - **Sentiment analysis** (**VADER + Roberta**)
   - **Keyword extraction**
   - **Summary generation**
   - **Toxicity classification**
   - **Multilingual sentiment** (if applicable)
3. **The original + processed data is saved** to the local **MongoDB** database.
4. **Admin Page** shows the **stored reviews sorted by timestamp**.


## 💬 **__Example Models Used__**

- **`cardiffnlp/twitter-roberta-base-sentiment`**  
  Used to perform **fine-grained sentiment classification** (Positive, Neutral, Negative) on text using a pre-trained RoBERTa model optimized for Twitter and social content.

- **`nlptown/bert-base-multilingual-uncased-sentiment`**  
  Enables **multilingual sentiment analysis**, capable of understanding and analyzing text in multiple languages (including English, French, German, etc.).

- **`unitary/toxic-bert`**  
  Detects **toxicity in user input**, such as offensive or harmful language. Returns both the **label (toxic/neutral)** and **confidence score**.

- **`sshleifer/distilbart-cnn-12-6`**  
  Used for **text summarization**, helping condense long reviews or feedback into a brief, readable summary without losing key meaning.

- **`Helsinki-NLP/opus-mt-ta-en`**  
  This **machine translation model** is used to **translate non-English (e.g., Tamil) reviews into English** for consistent analysis across languages.



 ## 📌 **__Notes__**

- **MongoDB must be installed and running locally** (`localhost:27017`)
- **The entire backend logic is in** `main/app.py`
- **Frontend template is in** `main/templates/index.html`

