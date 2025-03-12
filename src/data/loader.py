"""
Data loading utilities for the recommendation system.
"""

import pandas as pd
import os
import json

def load_articles(filepath):
    """
    Load articles from a JSON or CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the articles data
    """
    _, ext = os.path.splitext(filepath)
    
    if ext.lower() == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    
    elif ext.lower() == '.csv':
        return pd.read_csv(filepath)
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def save_articles(df, filepath):
    """
    Save articles to a file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the articles data
    filepath : str
        Path to save the data file
    """
    _, ext = os.path.splitext(filepath)
    
    if ext.lower() == '.json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(df.to_dict('records'), f, indent=2)
    
    elif ext.lower() == '.csv':
        df.to_csv(filepath, index=False)
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def create_sample_data():
    """
    Create a sample dataset of articles.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sample articles data
    """
    articles = [
        {
            "id": 1,
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
            "category": "AI",
            "tags": ["machine learning", "AI", "algorithms"]
        },
        {
            "id": 2,
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning is a subset of machine learning that uses neural networks with many layers.",
            "category": "AI",
            "tags": ["deep learning", "neural networks", "AI"]
        },
        {
            "id": 3,
            "title": "Python for Data Science",
            "content": "Python is a popular programming language for data science due to its simplicity and powerful libraries.",
            "category": "Programming",
            "tags": ["python", "programming", "data science"]
        },
        {
            "id": 4,
            "title": "Natural Language Processing",
            "content": "NLP is a field of AI that focuses on the interaction between computers and human language.",
            "category": "AI",
            "tags": ["NLP", "AI", "text processing"]
        },
        {
            "id": 5,
            "title": "Data Visualization Techniques",
            "content": "Data visualization is the graphical representation of information and data using visual elements.",
            "category": "Data Science",
            "tags": ["visualization", "data science", "charts"]
        }
    ]
    
    return pd.DataFrame(articles)