from flask import Flask, render_template, request, jsonify, send_file
import os
import PyPDF2
import io
from werkzeug.utils import secure_filename
import re
import json
from datetime import datetime
from collections import Counter
import math
import numpy as np

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 8MB max file size for free tier

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def preprocess_text(text):
    """Clean and preprocess the text"""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\'\"\(\)]', '', text)
    # Remove multiple periods
    text = re.sub(r'\.+', '.', text)
    return text.strip()

def get_stop_words():
    """Get comprehensive English stop words"""
    return {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose',
        'if', 'then', 'else', 'than', 'as', 'so', 'up', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'too', 'very', 'just', 'now', 'also', 'well', 'very', 'even', 'still',
        'back', 'get', 'go', 'come', 'make', 'take', 'give', 'say', 'see', 'know', 'think',
        'look', 'want', 'like', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try',
        'leave', 'call', 'good', 'new', 'first', 'last', 'long', 'great', 'little', 'own',
        'other', 'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next', 'early',
        'young', 'important', 'few', 'public', 'bad', 'same', 'able', 'above', 'across', 'after',
        'against', 'almost', 'alone', 'along', 'already', 'always', 'among', 'around', 'away',
        'because', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'both',
        'bottom', 'build', 'either', 'enough', 'every', 'everybody', 'everyone', 'everything',
        'everywhere', 'except', 'face', 'fact', 'far', 'feel', 'find', 'found', 'front', 'give',
        'group', 'hand', 'head', 'help', 'high', 'home', 'however', 'important', 'interest',
        'keep', 'kind', 'large', 'last', 'late', 'learn', 'left', 'let', 'life', 'light',
        'like', 'line', 'long', 'look', 'made', 'make', 'many', 'may', 'mean', 'might',
        'much', 'must', 'name', 'near', 'never', 'new', 'next', 'night', 'often', 'once',
        'open', 'order', 'part', 'people', 'place', 'point', 'power', 'put', 'question',
        'quite', 'rather', 'real', 'right', 'room', 'run', 'say', 'seem', 'set', 'show',
        'side', 'small', 'something', 'sometimes', 'sound', 'still', 'such', 'take', 'tell',
        'thing', 'think', 'though', 'through', 'together', 'turn', 'under', 'until', 'upon',
        'used', 'very', 'want', 'way', 'well', 'while', 'without', 'word', 'work', 'world',
        'year', 'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'
    }

def tokenize_sentence(sentence):
    """Tokenize sentence into words, removing stop words and short words"""
    stop_words = get_stop_words()
    words = re.findall(r'\b\w+\b', sentence.lower())
    return [word for word in words if word not in stop_words and len(word) > 2]

def calculate_sentence_similarity(sentence1, sentence2):
    """Calculate similarity between two sentences using cosine similarity"""
    words1 = set(tokenize_sentence(sentence1))
    words2 = set(tokenize_sentence(sentence2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)
