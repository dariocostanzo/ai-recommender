# Lesson 4: Advanced Similarity Techniques

## Introduction

In this lesson, we'll explore advanced similarity techniques to improve our recommendation system. We'll learn about different similarity measures and how to combine them for better recommendations.

## Similarity Measures

### Cosine Similarity

We've already used cosine similarity, which measures the cosine of the angle between two vectors:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example vectors
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

# Calculate cosine similarity
cos_sim = cosine_similarity([vec1], [vec2])[0][0]
print(f"Cosine similarity: {cos_sim:.4f}")
```

### Euclidean Distance

Euclidean distance measures the straight-line distance between two points:

```python
from sklearn.metrics.pairwise import euclidean_distances

# Calculate Euclidean distance
euc_dist = euclidean_distances([vec1], [vec2])[0][0]
print(f"Euclidean distance: {euc_dist:.4f}")

# Convert to similarity (closer to 1 means more similar)
euc_sim = 1 / (1 + euc_dist)
print(f"Euclidean similarity: {euc_sim:.4f}")
```

### Jaccard Similarity

Jaccard similarity measures the similarity between finite sets:

```python
def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# Example sets (words in documents)
doc1 = set("machine learning is fascinating".split())
doc2 = set("deep learning is powerful".split())

jac_sim = jaccard_similarity(doc1, doc2)
print(f"Jaccard similarity: {jac_sim:.4f}")
```

### Manhattan Distance

Manhattan distance (also known as L1 distance or taxicab distance) measures the sum of absolute differences between coordinates:

```python
from sklearn.metrics.pairwise import manhattan_distances

# Calculate Manhattan distance
man_dist = manhattan_distances([vec1], [vec2])[0][0]
print(f"Manhattan distance: {man_dist:.4f}")

# Convert to similarity
man_sim = 1 / (1 + man_dist)
print(f"Manhattan similarity: {man_sim:.4f}")
```

## Implementing Advanced Similarity in Our Recommender

Let's enhance our recommender with multiple similarity measures:

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load our articles dataset (from previous lesson)
articles = [
    {
        "id": "1",
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
        "category": "AI"
    },
    {
        "id": "2",
        "title": "Python for Data Science",
        "content": "Python is a popular programming language for data science and machine learning due to its simplicity and the availability of powerful libraries like NumPy, Pandas, and Scikit-learn.",
        "category": "Programming"
    },
    {
        "id": "3",
        "title": "Neural Networks Explained",
        "content": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains. They are designed to recognize patterns and interpret sensory data through machine perception.",
        "category": "AI"
    },
    {
        "id": "4",
        "title": "Introduction to JavaScript",
        "content": "JavaScript is a programming language that enables interactive web pages. It is an essential part of web applications, and all major web browsers have a dedicated JavaScript engine to execute it.",
        "category": "Programming"
    },
    {
        "id": "5",
        "title": "Recommendation Systems Overview",
        "content": "Recommendation systems are algorithms designed to suggest relevant items to users. They are used in many applications like suggesting products on e-commerce sites, movies on streaming platforms, or articles on news sites.",
        "category": "AI"
    }
]

df = pd.DataFrame(articles)
df['text'] = df['title'] + ": " + df['content']

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
df['embedding'] = df['text'].apply(lambda x: model.encode(x))

# Create bag of words representation for Jaccard similarity
vectorizer = CountVectorizer(stop_words='english')
bow_matrix = vectorizer.fit_transform(df['text'])
feature_names = vectorizer.get_feature_names_out()

# Store word sets for each document
df['words'] = df['text'].apply(lambda x: set(x.lower().split()))
```

Now, let's create an advanced recommender class that uses multiple similarity measures:

```python
class AdvancedRecommender:
    def __init__(self, articles_df):
        self.df = articles_df

    def get_article_by_id(self, article_id):
        """Return article details by ID"""
        return self.df[self.df['id'] == article_id].iloc[0]

    def calculate_similarity(self, article_id, method='cosine', top_n=3):
        """
        Calculate similarity between the target article and all other articles

        Parameters:
        article_id (str): ID of the target article
        method (str): Similarity method ('cosine', 'euclidean', 'manhattan', 'jaccard', or 'ensemble')
        top_n (int): Number of similar articles to return

        Returns:
        DataFrame: Top N similar articles
        """
        # Get the target article
        target_article = self.df[self.df['id'] == article_id].iloc[0]
        target_embedding = target_article['embedding']

        # Create a copy of the dataframe to avoid modifying the original
        results_df = self.df[self.df['id'] != article_id].copy()

        if method == 'cosine':
            # Cosine similarity
            results_df['similarity'] = results_df['embedding'].apply(
                lambda x: cosine_similarity([target_embedding], [x])[0][0]
            )
        elif method == 'euclidean':
            # Euclidean similarity
            results_df['similarity'] = results_df['embedding'].apply(
                lambda x: 1 / (1 + euclidean_distances([target_embedding], [x])[0][0])
            )
        elif method == 'manhattan':
            # Manhattan similarity
            results_df['similarity'] = results_df['embedding'].apply(
                lambda x: 1 / (1 + manhattan_distances([target_embedding], [x])[0][0])
            )
        elif method == 'jaccard':
            # Jaccard similarity
            target_words = target_article['words']
            results_df['similarity'] = results_df['words'].apply(
                lambda x: jaccard_similarity(target_words, x)
            )
        elif method == 'ensemble':
            # Ensemble of all methods (weighted average)
            results_df['cosine_sim'] = results_df['embedding'].apply(
                lambda x: cosine_similarity([target_embedding], [x])[0][0]
            )
            results_df['euclidean_sim'] = results_df['embedding'].apply(
                lambda x: 1 / (1 + euclidean_distances([target_embedding], [x])[0][0])
            )
            results_df['manhattan_sim'] = results_df['embedding'].apply(
                lambda x: 1 / (1 + manhattan_distances([target_embedding], [x])[0][0])
            )
            target_words = target_article['words']
            results_df['jaccard_sim'] = results_df['words'].apply(
                lambda x: jaccard_similarity(target_words, x)
            )

            # Weighted average (you can adjust weights based on performance)
            weights = {
                'cosine_sim': 0.4,
                'euclidean_sim': 0.3,
                'manhattan_sim': 0.2,
                'jaccard_sim': 0.1
            }

            results_df['similarity'] = (
                weights['cosine_sim'] * results_df['cosine_sim'] +
                weights['euclidean_sim'] * results_df['euclidean_sim'] +
                weights['manhattan_sim'] * results_df['manhattan_sim'] +
                weights['jaccard_sim'] * results_df['jaccard_sim']
            )
        else:
            raise ValueError(f"Unknown similarity method: {method}")

        # Sort by similarity (descending)
        results_df = results_df.sort_values('similarity', ascending=False).head(top_n)

        return results_df[['id', 'title', 'category', 'similarity']]

    def recommend_similar(self, article_id, method='ensemble', top_n=3):
        """Recommend articles similar to the given article"""
        return self.calculate_similarity(article_id, method, top_n)

# Create our advanced recommender
advanced_recommender = AdvancedRecommender(df)

# Test different similarity methods
target_article_id = "1"  # "Introduction to Machine Learning"

print("\nCosine similarity recommendations:")
print(advanced_recommender.recommend_similar(target_article_id, method='cosine'))

print("\nEuclidean similarity recommendations:")
print(advanced_recommender.recommend_similar(target_article_id, method='euclidean'))

print("\nManhattan similarity recommendations:")
print(advanced_recommender.recommend_similar(target_article_id, method='manhattan'))

print("\nJaccard similarity recommendations:")
print(advanced_recommender.recommend_similar(target_article_id, method='jaccard'))

print("\nEnsemble similarity recommendations:")
print(advanced_recommender.recommend_similar(target_article_id, method='ensemble'))
```

## Semantic Similarity

Semantic similarity goes beyond simple vector comparisons and tries to capture the meaning of text:

```python
from sentence_transformers import util

def semantic_similarity(text1, text2, model):
    """
    Calculate semantic similarity between two texts

    Parameters:
    text1 (str): First text
    text2 (str): Second text
    model: Sentence transformer model

    Returns:
    float: Similarity score
    """
    # Encode texts
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

    return similarity

# Example usage
text1 = "Machine learning algorithms build mathematical models based on sample data."
text2 = "AI systems learn from data to make predictions and decisions."
text3 = "JavaScript is a programming language used for web development."

print(f"Similarity between text1 and text2: {semantic_similarity(text1, text2, model):.4f}")
print(f"Similarity between text1 and text3: {semantic_similarity(text1, text3, model):.4f}")
```

## Contextual Similarity

We can also use contextual embeddings to capture the meaning of words in context:

```python
def get_contextual_similarity(query, documents, model):
    """
    Find documents similar to a query using contextual embeddings

    Parameters:
    query (str): Query text
    documents (list): List of documents
    model: Sentence transformer model

    Returns:
    list: Sorted list of (document, similarity) pairs
    """
    # Encode query and documents
    query_embedding = model.encode(query, convert_to_tensor=True)
    document_embeddings = model.encode(documents, convert_to_tensor=True)

    # Calculate similarities
    similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

    # Create pairs of (document, similarity)
    document_similarity_pairs = list(zip(documents, similarities.tolist()))

    # Sort by similarity (descending)
    document_similarity_pairs.sort(key=lambda x: x[1], reverse=True)

    return document_similarity_pairs

# Example usage
query = "How do neural networks work?"
documents = [
    "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
    "Python is a popular programming language for data science and machine learning.",
    "Recommendation systems suggest relevant items to users based on their preferences.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers."
]

results = get_contextual_similarity(query, documents, model)

print("\nQuery:", query)
print("\nResults:")
for doc, sim in results:
    print(f"Similarity: {sim:.4f} - {doc}")
```

## Improving Our Recommender with Advanced Similarity

Let's enhance our recommender to use semantic similarity for text queries:

````python
class EnhancedRecommender(AdvancedRecommender):
    def recommend_by_query(self, query, top_n=3):
        """
        Recommend articles based on a text query

        Parameters:
        query (str): Query text
        top_n (int): Number of recommendations to return

        Returns:
        DataFrame: Top N recommended articles
        """
        # Encode the query
        query_embedding = model.encode(query)

        # Create a copy of the dataframe
        results_df = self.df.copy()

        # Calculate similarity with query
        results_df['similarity'] = results_df['embedding'].apply(
            lambda x: cosine_similarity([query_embedding], [x])[0][0]
        )

        # Sort by similarity (descending)
        results_df = results_df.sort_values('similarity', ascending=False).head(top_n)

        return results_df[['id', 'title', 'category', 'similarity']]

    def hybrid_recommend(self, article_id=None, query=None, user_profile=None, weights=None, top_n=3):
        """
        Hybrid recommendation combining content-based, query-based, and user-based approaches

        Parameters:
        article_id (str): ID of the article to find similar articles for
        query (str): Query text
        user_profile (numpy.ndarray): User profile embedding
        weights (dict): Weights for different recommendation approaches
        top_n (int): Number of recommendations to return

        Returns:
        DataFrame: Top N recommended articles
        """
        if weights is None:
            weights = {
                'content': 0.4,
                'query': 0.3,
                'user': 0.3
            }

        # Create a copy of the dataframe
        results_df = self.df.copy()
        results_df['final_score'] = 0.0

        # Content-based similarity
        if article_id is not None and weights['content'] > 0:
            target_article = self.df[self.df['id'] == article_id].iloc[0]
            target_embedding = target_article['embedding']

            results_df['content_sim'] = results_df['embedding'].apply(
                lambda x: cosine_similarity([target_embedding], [x])[0][0]
            )

            # Add weighted content similarity to final score
            results_df['final_score'] += weights['content'] * results_df['content_sim']

            # Exclude the target article
            results_df = results_df[results_df['id'] != article_id]

        # Query-based similarity
        if query is not None and weights['query'] > 0:
            query_embedding = model.encode
            I'll continue with the rest of the Lesson 4 content. Here's the completion of the hybrid recommendation method and the rest of the file:

```markdown:c:\Users\dario\projects\ai-recommender\docs\lessons\04-advanced-similarity.md
            query_embedding = model.encode(query)

            results_df['query_sim'] = results_df['embedding'].apply(
                lambda x: cosine_similarity([query_embedding], [x])[0][0]
            )

            # Add weighted query similarity to final score
            results_df['final_score'] += weights['query'] * results_df['query_sim']

        # User-based similarity
        if user_profile is not None and weights['user'] > 0:
            results_df['user_sim'] = results_df['embedding'].apply(
                lambda x: cosine_similarity([user_profile], [x])[0][0]
            )

            # Add weighted user similarity to final score
            results_df['final_score'] += weights['user'] * results_df['user_sim']

        # Sort by final score (descending)
        results_df = results_df.sort_values('final_score', ascending=False).head(top_n)

        return results_df[['id', 'title', 'category', 'final_score']]

# Create our enhanced recommender
enhanced_recommender = EnhancedRecommender(df)

# Test query-based recommendations
query = "How do neural networks work?"
print("\nQuery-based recommendations:")
print(enhanced_recommender.recommend_by_query(query))

# Test hybrid recommendations
# Let's say the user has liked articles 1 and 3, and is now looking at article 5
user_liked_articles = ["1", "3"]
user_profile = np.mean([df[df['id'] == id]['embedding'].iloc[0] for id in user_liked_articles], axis=0)
current_article_id = "5"
search_query = "machine learning techniques"

print("\nHybrid recommendations:")
print(enhanced_recommender.hybrid_recommend(
    article_id=current_article_id,
    query=search_query,
    user_profile=user_profile
))
````

## Visualizing Similarity

Visualizing similarity can help us understand the relationships between articles:

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings(df, method='tsne'):
    """
    Visualize article embeddings in 2D space

    Parameters:
    df (DataFrame): DataFrame containing articles and their embeddings
    method (str): Dimensionality reduction method ('tsne' or 'pca')
    """
    # Extract embeddings
    embeddings = np.array(df['embedding'].tolist())

    # Reduce dimensionality to 2D
    if method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))

    # Define colors for categories
    categories = df['category'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
    category_color = {cat: col for cat, col in zip(categories, colors)}

    # Plot points
    for i, row in df.iterrows():
        plt.scatter(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            color=category_color[row['category']],
            s=100
        )
        plt.text(
            embeddings_2d[i, 0] + 0.03,
            embeddings_2d[i, 1] + 0.03,
            row['title'],
            fontsize=9
        )

    # Add legend
    for cat, col in category_color.items():
        plt.scatter([], [], color=col, label=cat)
    plt.legend()

    plt.title(f'Article Embeddings Visualization using {method.upper()}')
    plt.tight_layout()
    plt.show()

# Visualize our articles
visualize_embeddings(df, method='tsne')
```

## Evaluating Different Similarity Measures

Let's compare the performance of different similarity measures:

```python
def evaluate_similarity_methods(df, true_similar_pairs, methods=None):
    """
    Evaluate different similarity methods

    Parameters:
    df (DataFrame): DataFrame containing articles and their embeddings
    true_similar_pairs (list): List of tuples (article_id, list_of_truly_similar_ids)
    methods (list): List of similarity methods to evaluate

    Returns:
    DataFrame: Evaluation results
    """
    if methods is None:
        methods = ['cosine', 'euclidean', 'manhattan', 'jaccard', 'ensemble']

    # Create a recommender
    recommender = AdvancedRecommender(df)

    # Initialize results
    results = []

    # Evaluate each method
    for method in methods:
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0

        for article_id, true_similar_ids in true_similar_pairs:
            # Get recommendations using this method
            recommendations = recommender.recommend_similar(
                article_id,
                method=method,
                top_n=len(true_similar_ids)
            )

            # Get recommended IDs
            recommended_ids = recommendations['id'].tolist()

            # Calculate metrics
            true_relevant = set(true_similar_ids)
            recommended = set(recommended_ids)

            relevant_and_recommended = true_relevant.intersection(recommended)

            precision = len(relevant_and_recommended) / len(recommended) if recommended else 0
            recall = len(relevant_and_recommended) / len(true_relevant) if true_relevant else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precision_sum += precision
            recall_sum += recall
            f1_sum += f1

        # Calculate averages
        n_pairs = len(true_similar_pairs)
        avg_precision = precision_sum / n_pairs
        avg_recall = recall_sum / n_pairs
        avg_f1 = f1_sum / n_pairs

        # Add to results
        results.append({
            'method': method,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

# Define some ground truth similar pairs
# For example, article 1 is truly similar to articles 3 and 5
# article 2 is truly similar to article 4
true_similar_pairs = [
    ("1", ["3", "5"]),
    ("2", ["4"]),
    ("3", ["1", "5"]),
    ("4", ["2"]),
    ("5", ["1", "3"])
]

# Evaluate methods
evaluation_results = evaluate_similarity_methods(df, true_similar_pairs)
print("\nEvaluation of similarity methods:")
print(evaluation_results)

# Plot results
plt.figure(figsize=(12, 6))
evaluation_results.set_index('method')[['precision', 'recall', 'f1_score']].plot(kind='bar')
plt.title('Comparison of Similarity Methods')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
```

## Optimizing Similarity Weights

We can optimize the weights for our ensemble method:

```python
def optimize_ensemble_weights(df, true_similar_pairs, weight_combinations=None):
    """
    Find optimal weights for the ensemble method

    Parameters:
    df (DataFrame): DataFrame containing articles and their embeddings
    true_similar_pairs (list): List of tuples (article_id, list_of_truly_similar_ids)
    weight_combinations (list): List of weight dictionaries to try

    Returns:
    tuple: (best_weights, best_f1_score)
    """
    if weight_combinations is None:
        # Generate some weight combinations
        weight_combinations = []
        for cosine_w in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for euclidean_w in [0.1, 0.3, 0.5]:
                for manhattan_w in [0.1, 0.3]:
                    jaccard_w = 1.0 - cosine_w - euclidean_w - manhattan_w
                    if jaccard_w >= 0:
                        weight_combinations.append({
                            'cosine_sim': cosine_w,
                            'euclidean_sim': euclidean_w,
                            'manhattan_sim': manhattan_w,
                            'jaccard_sim': jaccard_w
                        })

    best_f1 = 0
    best_weights = None

    for weights in weight_combinations:
        # Create a custom recommender with these weights
        class CustomEnsembleRecommender(AdvancedRecommender):
            def calculate_similarity(self, article_id, method='ensemble', top_n=3):
                if method != 'ensemble':
                    return super().calculate_similarity(article_id, method, top_n)

                # Get the target article
                target_article = self.df[self.df['id'] == article_id].iloc[0]
                target_embedding = target_article['embedding']

                # Create a copy of the dataframe
                results_df = self.df[self.df['id'] != article_id].copy()

                # Calculate similarities using different methods
                results_df['cosine_sim'] = results_df['embedding'].apply(
                    lambda x: cosine_similarity([target_embedding], [x])[0][0]
                )
                results_df['euclidean_sim'] = results_df['embedding'].apply(
                    lambda x: 1 / (1 + euclidean_distances([target_embedding], [x])[0][0])
                )
                results_df['manhattan_sim'] = results_df['embedding'].apply(
                    lambda x: 1 / (1 + manhattan_distances([target_embedding], [x])[0][0])
                )
                target_words = target_article['words']
                results_df['jaccard_sim'] = results_df['words'].apply(
                    lambda x: jaccard_similarity(target_words, x)
                )

                # Apply custom weights
                results_df['similarity'] = (
                    weights['cosine_sim'] * results_df['cosine_sim'] +
                    weights['euclidean_sim'] * results_df['euclidean_sim'] +
                    weights['manhattan_sim'] * results_df['manhattan_sim'] +
                    weights['jaccard_sim'] * results_df['jaccard_sim']
                )

                # Sort by similarity (descending)
                results_df = results_df.sort_values('similarity', ascending=False).head(top_n)

                return results_df[['id', 'title', 'category', 'similarity']]

        # Create the recommender
        recommender = CustomEnsembleRecommender(df)

        # Evaluate
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0

        for article_id, true_similar_ids in true_similar_pairs:
            # Get recommendations
            recommendations = recommender.recommend_similar(
                article_id,
                method='ensemble',
                top_n=len(true_similar_ids)
            )

            # Get recommended IDs
            recommended_ids = recommendations['id'].tolist()

            # Calculate metrics
            true_relevant = set(true_similar_ids)
            recommended = set(recommended_ids)

            relevant_and_recommended = true_relevant.intersection(recommended)

            precision = len(relevant_and_recommended) / len(recommended) if recommended else 0
            recall = len(relevant_and_recommended) / len(true_relevant) if true_relevant else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precision_sum += precision
            recall_sum += recall
            f1_sum += f1

        # Calculate average F1 score
        avg_f1 = f1_sum / len(true_similar_pairs)

        # Check if this is the best so far
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_weights = weights

    return best_weights, best_f1

# Find optimal weights
best_weights, best_f1 = optimize_ensemble_weights(df, true_similar_pairs)
print("\nOptimal ensemble weights:")
print(best_weights)
print(f"Best F1 score: {best_f1:.4f}")
```

## Exercise

1. Implement a new similarity measure not covered in this lesson (e.g., Pearson correlation)
2. Create a visualization that shows the similarity between all pairs of articles
3. Extend the hybrid recommender to include category-based filtering
4. Experiment with different weighting schemes for the ensemble method
5. Implement a function to explain why a particular article was recommended

## Next Steps

In the next lesson, we'll explore collaborative filtering, which recommends items based on user behavior rather than content similarity.

## Resources

- [Similarity Measures in Machine Learning](https://towardsdatascience.com/17-types-of-similarity-and-dissimilarity-measures-used-in-data-science-3eb914d2681)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
- [Visualizing Embeddings with t-SNE](https://distill.pub/2016/misread-tsne/)
