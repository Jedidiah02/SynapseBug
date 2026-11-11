import os
from dotenv import load_dotenv
import re
import spacy
import numpy as np
import nltk
from flask import Flask, request, render_template, redirect, session, flash, url_for
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords
import secrets
from datetime import datetime, timedelta
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('stopwords')

# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(16)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    reset_token = db.Column(db.String(100), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)

    def __init__(self, name, email, username, password, is_admin=False):
        self.name = name
        self.email = email
        self.username = username
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        self.is_admin = is_admin

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

with app.app_context():
    db.create_all()
    if ADMIN_PASSWORD and not User.query.filter_by(email=ADMIN_EMAIL).first():
        admin = User(name='admin', email=ADMIN_EMAIL, username='admin',
                     password=ADMIN_PASSWORD, is_admin=True)
        db.session.add(admin)
        db.session.commit()

def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    filtered_tokens = remove_stopwords(tokens)
    return filtered_tokens

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

def extract_keywords_rake(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    return keywords

def generate_summary(text):
    preprocessed_text = preprocess_text(text)
    keywords = extract_keywords_rake(text)
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    ranked_sentences = []
    for sentence in sentences:
        sentence_tokens = preprocess_text(sentence)
        sentence_score = sum(1 for word in sentence_tokens if word in keywords)
        ranked_sentences.append((sentence, sentence_score))
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[1], reverse=True)
    num_sentences = min(len(ranked_sentences), max(1, len(sentences) // 3))
    summary_sentences = [sentence[0] for sentence in ranked_sentences[:num_sentences]]
    summary_sentences = sorted(summary_sentences, key=lambda x: sentences.index(x))
    summary = " ".join(summary_sentences)
    return summary

def validate_password(password):
    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    if not re.search(r'[0-9]', password):
        errors.append("Password must contain at least one digit")
    if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
        errors.append("Password must contain at least one special character")
    if errors:
        return False, errors
    return True, []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    error_messages = None
    if request.method == 'POST':
        logger.debug(request.form)  # Use logging instead of print
        name = request.form['name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            error_messages = ['Email already exists!']
            return render_template('register.html', error=error_messages)
        is_valid, validation_messages = validate_password(password)
        if not is_valid:
            error_messages = validation_messages
            return render_template('register.html', error=error_messages)
        new_user = User(name=name, email=email, username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html', error=error_messages)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['email'] = user.email
            session['is_admin'] = user.is_admin
            if user.is_admin:
                return redirect('/admin')
            else:
                return redirect('/dashboard')
        else:
            error = 'Invalid username or password'
    return render_template('login.html', error=error)

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'email' not in session:
        return redirect('/login')
    user = User.query.filter_by(email=session['email']).first()
    if request.method == 'POST':
        input_text = request.form['input_text']
        summary = generate_summary(input_text)
        return render_template('output.html', summary=summary)
    return render_template('input.html', user=user)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email, is_admin=True).first()
        if user and user.check_password(password):
            session['email'] = user.email
            session['is_admin'] = user.is_admin
            return redirect('/admin')
        else:
            error = 'Invalid admin credentials'
    return render_template('admin_login.html', error=error)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'email' not in session or not session.get('is_admin'):
        return redirect('/login')
    
    users = User.query.all()

    if request.method == 'POST':
        delete_user_id = request.form.get('delete_user_id')
        if delete_user_id:
            user_to_delete = User.query.get(delete_user_id)
            if user_to_delete:
                db.session.delete(user_to_delete)
                db.session.commit()
                flash('User deleted successfully', 'success')
            else:
                flash('User not found', 'error')

    return render_template('admin.html', users=users)

@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    if request.method == 'POST':
        email = request.form.get('email')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        if new_password == confirm_password:
            user = User.query.filter_by(email=email).first()
            if user:
                user.password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                db.session.commit()
                flash('Password reset successfully!', 'success')
                return redirect('/login')
            else:
                flash('User with that email not found.', 'error')
        else:
            flash('Passwords do not match!', 'error')
    return render_template('reset_password_request.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        if new_password == confirm_password:
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            # Here you would update the user's password in the database
            flash('Password reset successfully!', 'success')
            return redirect('/login')
        else:
            flash('Passwords do not match!', 'error')
    return render_template('reset_password.html')

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect('/')

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    error_message = None
    if 'email' not in session:
        return redirect('/login')
    
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        user = User.query.filter_by(email=session['email']).first()
        
        if not user or not user.check_password(current_password):
            error_message = 'Invalid current password'
        elif new_password != confirm_password:
            error_message = 'New password and confirm password do not match'
        else:
            is_valid, validation_messages = validate_password(new_password)
            if not is_valid:
                error_message = validation_messages[0]  # Display the first validation error
            else:
                user.password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                db.session.commit()
                flash('Password changed successfully!', 'success')
                return redirect('/dashboard')

    return render_template('change_password.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
