import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class ContentBasedRecommender:
    def __init__(self, articles_df=None):
        self.df = articles_df
        
    def fit(self, articles_df):
        """
        Fit the recommender with a dataframe of articles
        
        Parameters:
        articles_df (DataFrame): DataFrame containing articles with 'title' and 'content' columns
        """
        self.df = articles_df
        
        # Create text field and embeddings if they don't exist
        if 'text' not in self.df.columns:
            self.df['text'] = self.df['title'] + ": " + self.df['content']
            
        if 'embedding' not in self.df.columns:
            # Load the model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            # Create embeddings
            self.df['embedding'] = self.df['text'].apply(lambda x: model.encode(x))
            
        return self
    
    def get_article_by_id(self, article_id):
        """Return article details by ID"""
        return self.df[self.df['id'] == article_id].iloc[0]
    
    def recommend_similar(self, article_id, top_n=3):
        """Recommend articles similar to the given article"""
        return self._get_similar_articles(article_id, top_n)
    
    def _get_similar_articles(self, article_id, top_n=3):
        """
        Find articles similar to the given article_id
        
        Parameters:
        article_id (str): ID of the article to find similar articles for
        top_n (int): Number of similar articles to return
        
        Returns:
        DataFrame: Top N similar articles
        """
        # Get the embedding of the target article
        target_embedding = self.df[self.df['id'] == article_id]['embedding'].iloc[0]
        
        # Calculate similarity with all other articles
        self.df['similarity'] = self.df['embedding'].apply(lambda x: cosine_similarity([target_embedding], [x])[0][0])
        
        # Sort by similarity (descending) and exclude the target article
        similar_articles = self.df[self.df['id'] != article_id].sort_values('similarity', ascending=False).head(top_n)
        
        return similar_articles[['id', 'title', 'category', 'similarity']]
    
    def recommend_by_category(self, category, top_n=3):
        """Recommend top articles from a specific category"""
        category_articles = self.df[self.df['category'] == category]
        
        # If we want to sort by some relevance metric, we could do that here
        # For now, we'll just return the first N articles in the category
        return category_articles.head(top_n)[['id', 'title', 'category']]
    
    def create_user_profile(self, liked_article_ids):
        """
        Create a user profile based on articles they've liked
        
        Parameters:
        liked_article_ids (list): List of article IDs the user has liked
        
        Returns:
        numpy.ndarray: User profile embedding
        """
        # Get embeddings of liked articles
        liked_embeddings = self.df[self.df['id'].isin(liked_article_ids)]['embedding'].tolist()
        
        if not liked_embeddings:
            return None
        
        # Create user profile by averaging embeddings of liked articles
        user_profile = np.mean(liked_embeddings, axis=0)
        return user_profile
    
    def recommend_for_user(self, user_profile, top_n=3, exclude_ids=None):
        """
        Recommend articles for a user based on their profile
        
        Parameters:
        user_profile (numpy.ndarray): User profile embedding
        top_n (int): Number of recommendations to return
        exclude_ids (list): List of article IDs to exclude from recommendations
        
        Returns:
        DataFrame: Top N recommended articles
        """
        if exclude_ids is None:
            exclude_ids = []
        
        # Filter out articles the user has already interacted with
        recommendations_df = self.df[~self.df['id'].isin(exclude_ids)].copy()
        
        # Calculate similarity with user profile
        recommendations_df['similarity'] = recommendations_df['embedding'].apply(
            lambda x: cosine_similarity([user_profile], [x])[0][0]
        )
        
        # Sort by similarity (descending)
        recommendations_df = recommendations_df.sort_values('similarity', ascending=False).head(top_n)
        
        return recommendations_df[['id', 'title', 'category', 'similarity']]
    
    def recommend_by_text_query(self, query_text, top_n=3):
        """
        Recommend articles based on a text query
        
        Parameters:
        query_text (str): Text query to find similar articles
        top_n (int): Number of recommendations to return
        
        Returns:
        DataFrame: Top N recommended articles
        """
        # Load the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embedding for the query
        query_embedding = model.encode(query_text)
        
        # Calculate similarity with all articles
        recommendations_df = self.df.copy()
        recommendations_df['similarity'] = recommendations_df['embedding'].apply(
            lambda x: cosine_similarity([query_embedding], [x])[0][0]
        )
        
        # Sort by similarity (descending)
        recommendations_df = recommendations_df.sort_values('similarity', ascending=False).head(top_n)
        
        return recommendations_df[['id', 'title', 'category', 'similarity']]

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