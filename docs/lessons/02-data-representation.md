# Lesson 2: Understanding Data Representation

## Introduction

In this lesson, we'll explore how to represent text data for machine learning models. Understanding data representation is crucial for building effective recommendation systems.

## Text Representation Techniques

### Bag of Words (BoW)

The simplest way to represent text is through the Bag of Words model:

- Count the frequency of each word in a document
- Ignore grammar and word order
- Create a vocabulary of all unique words in the corpus
- Represent each document as a vector of word counts

### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF improves on BoW by weighting terms:

- Term Frequency (TF): How often a word appears in a document
- Inverse Document Frequency (IDF): How unique or rare a word is across all documents
- TF-IDF = TF × IDF

This helps reduce the importance of common words like "the" or "and" that appear in many documents.

## Implementing Basic Text Representation

We'll use scikit-learn to implement these techniques:

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample documents
documents = [
    "I love machine learning and AI",
    "Recommendation systems are powerful",
    "AI and machine learning are transforming industries"
]

# Bag of Words
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(documents)
print("BoW vocabulary:", bow_vectorizer.get_feature_names_out())
print("BoW matrix shape:", bow_matrix.shape)
print("BoW representation:\n", bow_matrix.toarray())

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print("\nTF-IDF vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("TF-IDF representation:\n", tfidf_matrix.toarray())
```

## Word Embeddings

Word embeddings are more advanced representations that capture semantic meaning:

- Words are represented as dense vectors in a continuous vector space
- Similar words are positioned close to each other
- Relationships between words can be captured mathematically

### Word2Vec

Word2Vec learns word associations from a large corpus of text:

- Continuous Bag of Words (CBOW): Predicts a word given its context
- Skip-gram: Predicts context words given a target word

### GloVe (Global Vectors for Word Representation)

GloVe combines global matrix factorization and local context window methods.

### Sentence Transformers

For our recommendation system, we'll use Sentence Transformers to get embeddings for entire documents:

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(documents)

print("Embedding shape:", embeddings.shape)
print("First document embedding:", embeddings[0][:10])  # Show first 10 dimensions
```

## Comparing Documents Using Embeddings

Once we have embeddings, we can compare documents using similarity measures:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between all document pairs
similarity_matrix = cosine_similarity(embeddings)

print("Similarity matrix:")
print(similarity_matrix)

# Find most similar document to the first document
doc_index = 0
similarities = similarity_matrix[doc_index]
most_similar_index = similarities.argsort()[-2]  # Second highest (highest is itself)
print(f"Most similar to '{documents[doc_index]}' is: '{documents[most_similar_index]}'")
```

## Exercise

1. Create embeddings for a set of articles or blog posts
2. Calculate similarity between articles
3. Build a simple function that returns the top 3 most similar articles to a given article

## Next Steps

In the next lesson, we'll build a basic recommendation system using the embedding techniques we've learned.

## Resources

- [Scikit-learn Documentation: Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Word2Vec Explained](https://towardsdatascience.com/word2vec-explained-49c52b4ccb71)
