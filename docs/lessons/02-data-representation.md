# Lesson 2: Understanding Data Representation

## Introduction

In this lesson, we'll explore different ways to represent text data for our recommendation system. How we represent our data is crucial - it determines what patterns our system can learn and how effectively it can make recommendations.

## Text Representation Techniques

### 1. Bag of Words (BoW)

The simplest way to represent text is the Bag of Words model:

- Each document is represented as a vector of word counts
- The position and order of words are ignored
- Each dimension in the vector corresponds to a word in the vocabulary

**Advantages:**

- Simple to understand and implement
- Works well for basic text classification

**Disadvantages:**

- Ignores word order and context
- Results in sparse, high-dimensional vectors
- Doesn't capture semantic meaning

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF improves on BoW by weighting terms based on their importance:

- Term Frequency (TF): How often a word appears in a document
- Inverse Document Frequency (IDF): How rare a word is across all documents
- TF-IDF = TF × IDF

**Advantages:**

- Reduces the impact of common words
- Gives more weight to distinctive terms
- Better performance than simple BoW

**Disadvantages:**

- Still uses sparse vectors
- Doesn't capture semantic relationships between words
- No understanding of word context

### 3. Word Embeddings

Word embeddings represent words as dense vectors in a continuous vector space:

- Words with similar meanings are positioned close to each other
- Vectors capture semantic relationships
- Pre-trained models like Word2Vec, GloVe, or BERT can be used

**Advantages:**

- Captures semantic meaning
- Lower-dimensional representation
- Can represent relationships between words (e.g., king - man + woman ≈ queen)

**Disadvantages:**

- More complex to implement
- Requires more computational resources
- May need large amounts of training data

## Document Similarity

Once we have vector representations of documents, we can calculate similarity between them:

### Cosine Similarity

The most common similarity measure for text data:

- Measures the cosine of the angle between two vectors
- Ranges from -1 (completely opposite) to 1 (exactly the same)
- Formula: cos(θ) = (A·B) / (||A|| × ||B||)

### Euclidean Distance

The straight-line distance between two points:

- Smaller values indicate more similar documents
- Formula: d(A, B) = √(Σ(Ai - Bi)²)

## Building a Simple Recommendation System

Using these representation techniques, we can build a simple recommendation system:

1. Represent all documents as vectors
2. For a given query or document, find the most similar documents
3. Recommend the top N most similar documents

## Practical Implementation

In the accompanying notebook, we'll implement:

1. BoW and TF-IDF representations using scikit-learn
2. Word embeddings using Sentence Transformers
3. Document similarity calculations
4. A simple recommendation function

## Exercise

Before moving on to the next lesson:

1. Experiment with different preprocessing options in the TextProcessor class
2. Try different similarity measures
3. Test the recommendation system with your own queries
4. Compare the results from different representation methods

## Resources

- [scikit-learn: Working with Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Understanding TF-IDF](https://monkeylearn.com/blog/what-is-tf-idf/)
- [Word Embeddings Guide](https://jalammar.github.io/illustrated-word2vec/)
