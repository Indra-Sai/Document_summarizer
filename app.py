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
def build_similarity_matrix(sentences):
    """Build similarity matrix for sentences"""
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i][j] = calculate_sentence_similarity(sentences[i], sentences[j])
    
    return similarity_matrix

def normalize_matrix(matrix):
    """Normalize matrix by row sums"""
    row_sums = matrix.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    return matrix / row_sums[:, np.newaxis]

def textrank_summarize(sentences, summary_ratio=0.3, damping=0.85, max_iter=100, tolerance=1e-6):
    """Implement TextRank algorithm for extractive summarization"""
    if len(sentences) <= 3:
        return sentences
    
    # Build similarity matrix
    similarity_matrix = build_similarity_matrix(sentences)
    
    # Normalize the similarity matrix
    normalized_matrix = normalize_matrix(similarity_matrix)
    
    # Initialize scores
    scores = np.ones(len(sentences)) / len(sentences)
    
    # Iterative TextRank algorithm
    for _ in range(max_iter):
        new_scores = (1 - damping) + damping * normalized_matrix.T.dot(scores)
        
        # Check convergence
        if np.sum(np.abs(new_scores - scores)) < tolerance:
            break
        
        scores = new_scores
    
    # Select top sentences
    num_sentences = max(3, int(len(sentences) * summary_ratio))
    top_indices = np.argsort(scores)[-num_sentences:]
    
    # Sort by original order to maintain coherence
    top_indices = sorted(top_indices)
    
    # Create summary
    summary_sentences = [sentences[i] for i in top_indices]
    summary = ' '.join(summary_sentences)
    
    return summary

def enhance_summary_with_keywords(text, summary):
    """Enhance summary by ensuring important keywords are included"""
    # Extract important keywords from original text
    stop_words = get_stop_words()
    words = re.findall(r'\b\w+\b', text.lower())
    words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Get word frequencies
    word_freq = Counter(words)
    important_words = [word for word, freq in word_freq.most_common(10)]
    
    # Check if important words are in summary
    summary_words = set(re.findall(r'\b\w+\b', summary.lower()))
    missing_words = [word for word in important_words if word not in summary_words]
    
    # If important words are missing, try to add relevant sentences
    if missing_words and len(missing_words) > 0:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for word in missing_words[:3]:  # Add at most 3 missing important words
            for sentence in sentences:
                if word in sentence.lower() and sentence not in summary:
                    # Add sentence if it's not too long and doesn't make summary too long
                    if len(sentence) < 200 and len(summary) + len(sentence) < len(text) * 0.5:
                        summary += ' ' + sentence
                        break
    
    return summary

def summarize_text_advanced(text, summary_ratio=0.25):
    """Generate a summary using TextRank algorithm with enhancements"""
    if not text.strip():
        return "No text found in the document."
    
    # Preprocess text
    text = preprocess_text(text)
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    if len(sentences) <= 3:
        return text
    
    # Generate summary using TextRank
    summary = textrank_summarize(sentences, summary_ratio)
    
    # Enhance summary with important keywords
    summary = enhance_summary_with_keywords(text, summary)
    
    # Post-process summary for better readability
    summary = re.sub(r'\s+', ' ', summary)  # Clean up whitespace
    summary = summary.strip()
    
    # Ensure summary is not too long
    if len(summary) > len(text) * 0.5:
        # Truncate to reasonable length
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]
        target_length = len(text) * 0.4
        current_length = 0
        final_sentences = []
        
        for sentence in sentences:
            if current_length + len(sentence) <= target_length:
                final_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        summary = ' '.join(final_sentences)
    
    return summary

def count_words_and_sentences(text):
    """Count words and sentences in text"""
    # Count words
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    
    # Count sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Count paragraphs
    paragraphs = text.split('\n\n')
    paragraph_count = len([p for p in paragraphs if p.strip()])
    
    return word_count, sentence_count, paragraph_count
