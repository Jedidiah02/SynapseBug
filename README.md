# SynapseBug

A Flask web app for extractive text summarization. Users register and log in, paste text on a dashboard, and get a shorter summary built from its most keyword-rich sentences.

## Features
- Extractive summarization: spaCy sentence splitting, NLTK stopword filtering, RAKE keyword scoring
- User registration and login with bcrypt password hashing and a password policy
- Role-based access: admins can view and delete users
- SQLite database through SQLAlchemy

## Tech stack
Python, Flask, SQLAlchemy, SQLite, spaCy, NLTK, RAKE, Bootstrap

## Setup
```
git clone https://github.com/Jedidiah02/SynapseBug.git
cd SynapseBug
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
Copy `.env.example` to `.env` and set your own values.

## Run
```
python app.py
```
Open http://127.0.0.1:5000

## Limitations and next steps
- Summaries pick existing sentences only (next: LLM-based abstractive summaries and a RAG question-answering feature)
- Password reset does not verify the user's identity yet (next: token-based reset by email)
