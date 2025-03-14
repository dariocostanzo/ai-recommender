# Lesson 3: Building a Basic Recommendation System

## Introduction

In this lesson, we'll build our first recommendation system using the text representation techniques we learned in the previous lesson. We'll focus on content-based filtering, which recommends items similar to what a user has liked in the past.

## Content-Based Filtering

Content-based filtering recommends items by comparing the content of items rather than relying on user interactions. The basic steps are:

1. Create a profile for each item (using embeddings)
2. Create a profile for the user based on items they've liked
3. Compare the user profile with item profiles to find recommendations

## Setting Up Our Dataset

Let's create a simple dataset of articles to work with:

```python
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Sample articles dataset
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

# Convert to DataFrame
df = pd.DataFrame(articles)
print(df[["id", "title", "category"]])
```

## Creating Item Profiles

Now, let's create embeddings for each article using Sentence Transformers:

```python
# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for each article
# We'll combine title and content for better representation
df['text'] = df['title'] + ": " + df['content']
df['embedding'] = df['text'].apply(lambda x: model.encode(x))

print("Embedding shape:", df['embedding'].iloc[0].shape)
```

## Finding Similar Articles

Let's implement a function to find similar articles based on content:

```python
def get_similar_articles(article_id, df, top_n=3):
    """
    Find articles similar to the given article_id

    Parameters:
    article_id (str): ID of the article to find similar articles for
    df (DataFrame): DataFrame containing articles and their embeddings
    top_n (int): Number of similar articles to return

    Returns:
    DataFrame: Top N similar articles
    """
    # Get the embedding of the target article
    target_embedding = df[df['id'] == article_id]['embedding'].iloc[0]

    # Calculate similarity with all other articles
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity([target_embedding], [x])[0][0])

    # Sort by similarity (descending) and exclude the target article
    similar_articles = df[df['id'] != article_id].sort_values('similarity', ascending=False).head(top_n)

    return similar_articles[['id', 'title', 'category', 'similarity']]

# Test our function
target_article_id = "1"  # "Introduction to Machine Learning"
similar_articles = get_similar_articles(target_article_id, df)

print(f"Articles similar to '{df[df['id'] == target_article_id]['title'].iloc[0]}':")
print(similar_articles)
```

## Building a Simple Recommendation System

Now, let's build a simple recommendation system class:

```python
class ContentBasedRecommender:
    def __init__(self, articles_df):
        self.df = articles_df

    def get_article_by_id(self, article_id):
        """Return article details by ID"""
        return self.df[self.df['id'] == article_id].iloc[0]

    def recommend_similar(self, article_id, top_n=3):
        """Recommend articles similar to the given article"""
        return get_similar_articles(article_id, self.df, top_n)

    def recommend_by_category(self, category, top_n=3):
        """Recommend top articles from a specific category"""
        category_articles = self.df[self.df['category'] == category]

        # If we want to sort by some relevance metric, we could do that here
        # For now, we'll just return the first N articles in the category
        return category_articles.head(top_n)[['id', 'title', 'category']]

# Create our recommender
recommender = ContentBasedRecommender(df)

# Test category-based recommendations
print("\nTop articles in 'AI' category:")
print(recommender.recommend_by_category('AI'))
```

## User Profiles

To personalize recommendations, we can create user profiles based on articles they've liked:

```python
def create_user_profile(liked_article_ids, df):
    """
    Create a user profile based on articles they've liked

    Parameters:
    liked_article_ids (list): List of article IDs the user has liked
    df (DataFrame): DataFrame containing articles and their embeddings

    Returns:
    numpy.ndarray: User profile embedding
    """
    # Get embeddings of liked articles
    liked_embeddings = df[df['id'].isin(liked_article_ids)]['embedding'].tolist()

    if not liked_embeddings:
        return None

    # Create user profile by averaging embeddings of liked articles
    user_profile = np.mean(liked_embeddings, axis=0)
    return user_profile

def recommend_for_user(user_profile, df, top_n=3, exclude_ids=None):
    """
    Recommend articles for a user based on their profile

    Parameters:
    user_profile (numpy.ndarray): User profile embedding
    df (DataFrame): DataFrame containing articles and their embeddings
    top_n (int): Number of recommendations to return
    exclude_ids (list): List of article IDs to exclude from recommendations

    Returns:
    DataFrame: Top N recommended articles
    """
    if exclude_ids is None:
        exclude_ids = []

    # Filter out articles the user has already interacted with
    recommendations_df = df[~df['id'].isin(exclude_ids)].copy()

    # Calculate similarity with user profile
    recommendations_df['similarity'] = recommendations_df['embedding'].apply(
        lambda x: cosine_similarity([user_profile], [x])[0][0]
    )

    # Sort by similarity (descending)
    recommendations_df = recommendations_df.sort_values('similarity', ascending=False).head(top_n)

    return recommendations_df[['id', 'title', 'category', 'similarity']]

# Test user-based recommendations
user_liked_articles = ["1", "3"]  # User likes ML and Neural Networks
user_profile = create_user_profile(user_liked_articles, df)

print("\nRecommendations for user who liked articles about ML and Neural Networks:")
print(recommend_for_user(user_profile, df, exclude_ids=user_liked_articles))
```

## Saving and Loading the Recommender

Let's add functionality to save and load our recommendation model:

```python
import pickle
import os

def save_recommender(recommender, filepath):
    """Save the recommender to a file"""
    with open(filepath, 'wb') as f:
        pickle.dump(recommender, f)
    print(f"Recommender saved to {filepath}")

def load_recommender(filepath):
    """Load a recommender from a file"""
    with open(filepath, 'rb') as f:
        recommender = pickle.load(f)
    print(f"Recommender loaded from {filepath}")
    return recommender

# Example usage
models_dir = "c:\\Users\\dario\\projects\\ai-recommender\\models"
os.makedirs(models_dir, exist_ok=True)
save_recommender(recommender, os.path.join(models_dir, "content_recommender.pkl"))
```

## Evaluating Recommendations

To evaluate our recommender, we can use metrics like precision and recall:

```python
def evaluate_recommendations(true_relevant_ids, recommended_ids):
    """
    Evaluate recommendations using precision and recall

    Parameters:
    true_relevant_ids (list): List of truly relevant item IDs
    recommended_ids (list): List of recommended item IDs

    Returns:
    dict: Dictionary with precision and recall metrics
    """
    true_relevant = set(true_relevant_ids)
    recommended = set(recommended_ids)

    # Calculate metrics
    relevant_and_recommended = true_relevant.intersection(recommended)

    precision = len(relevant_and_recommended) / len(recommended) if recommended else 0
    recall = len(relevant_and_recommended) / len(true_relevant) if true_relevant else 0

    # F1 score (harmonic mean of precision and recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# Example evaluation
# Let's say articles 2 and 5 are truly relevant for a user who liked articles 1 and 3
true_relevant = ["2", "5"]
recommended = recommend_for_user(user_profile, df, exclude_ids=user_liked_articles)['id'].tolist()

metrics = evaluate_recommendations(true_relevant, recommended)
print("\nRecommendation evaluation:")
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"F1 Score: {metrics['f1_score']:.2f}")
```

## Exercise

1. Expand the dataset with more articles (at least 10)
2. Implement a function to recommend articles based on a text query
3. Create a hybrid recommender that combines category-based and content-based recommendations
4. Add a feature to save and load user profiles

## Next Steps

In the next lesson, we'll explore advanced similarity techniques to improve our recommendations.

## Resources

- [Building Recommendation Systems with Python](https://realpython.com/build-recommendation-engine-collaborative-filtering/)
- [Content-Based Filtering](https://developers.google.com/machine-learning/recommendation/content-based/basics)
- [Sentence Transformers Examples](https://www.sbert.net/examples/applications/computing-embeddings/README.html)
