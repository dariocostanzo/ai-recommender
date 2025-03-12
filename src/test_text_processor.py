"""
Test script for the TextProcessor class.
"""

from features.text_processor import TextProcessor
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    # Create sample documents
    documents = [
        "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers.",
        "Python is a popular programming language for data science due to its simplicity and powerful libraries.",
        "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
        "Data visualization is the graphical representation of information and data using visual elements."
    ]
    
    # Initialize text processor
    processor = TextProcessor(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        stemming=False,
        lemmatization=True
    )
    
    # Test preprocessing
    print("=== Text Preprocessing ===")
    preprocessed = processor.preprocess_documents(documents)
    for i, (original, processed) in enumerate(zip(documents, preprocessed)):
        print(f"\nDocument {i+1}:")
        print(f"Original: {original}")
        print(f"Processed: {processed}")
    
    # Test TF-IDF vectorization
    print("\n\n=== TF-IDF Vectorization ===")
    tfidf_matrix, feature_names = processor.create_tfidf_vectors(documents)
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"Number of Features: {len(feature_names)}")
    print(f"Sample Features: {feature_names[:10]}")
    
    # Test keyword extraction
    print("\n\n=== Keyword Extraction ===")
    for i, doc in enumerate(documents):
        keywords = processor.extract_keywords(doc, top_n=3)
        print(f"\nTop keywords for Document {i+1}:")
        for keyword, score in keywords:
            print(f"  - {keyword}: {score:.4f}")
    
    # Test document similarity
    print("\n\n=== Document Similarity ===")
    # Compare ML and DL documents
    sim1 = processor.compute_document_similarity(documents[0], documents[1])
    print(f"Similarity between ML and DL documents: {sim1:.4f}")
    
    # Compare ML and Python documents
    sim2 = processor.compute_document_similarity(documents[0], documents[2])
    print(f"Similarity between ML and Python documents: {sim2:.4f}")
    
    # Compare ML and NLP documents
    sim3 = processor.compute_document_similarity(documents[0], documents[3])
    print(f"Similarity between ML and NLP documents: {sim3:.4f}")
    
    # Compare ML and Data Viz documents
    sim4 = processor.compute_document_similarity(documents[0], documents[4])
    print(f"Similarity between ML and Data Viz documents: {sim4:.4f}")

if __name__ == "__main__":
    main()