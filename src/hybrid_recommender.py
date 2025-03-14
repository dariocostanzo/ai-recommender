import pandas as pd
import numpy as np
from recommender import ContentBasedRecommender
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, content_recommender=None):
        self.content_recommender = content_recommender or ContentBasedRecommender()
        
    def fit(self, articles_df):
        """Fit the recommender with a dataframe of articles"""
        self.content_recommender.fit(articles_df)
        return self
    
    def recommend_hybrid(self, article_id=None, user_profile=None, category=None, weights=None, top_n=5):
        """
        Generate hybrid recommendations using multiple signals
        
        Parameters:
        article_id (str): ID of an article to find similar content
        user_profile (numpy.ndarray): User profile for personalized recommendations
        category (str): Category to filter recommendations
        weights (dict): Weights for different recommendation sources (default: equal weights)
        top_n (int): Number of recommendations to return
        
        Returns:
        DataFrame: Top N recommended articles
        """
        if weights is None:
            # Default weights
            weights = {
                'content': 0.4,
                'user': 0.4,
                'category': 0.2
            }
            
        # Initialize an empty DataFrame to store all candidates
        all_candidates = pd.DataFrame()
        
        # Get content-based recommendations if article_id is provided
        if article_id:
            content_recs = self.content_recommender.recommend_similar(article_id, top_n=top_n*2)
            content_recs['score'] = content_recs['similarity'] * weights['content']
            content_recs['source'] = 'content'
            all_candidates = pd.concat([all_candidates, content_recs])
        
        # Get user-based recommendations if user_profile is provided
        if user_profile is not None:
            user_recs = self.content_recommender.recommend_for_user(user_profile, top_n=top_n*2)
            user_recs['score'] = user_recs['similarity'] * weights['user']
            user_recs['source'] = 'user'
            all_candidates = pd.concat([all_candidates, user_recs])
        
        # Get category-based recommendations if category is provided
        if category:
            category_recs = self.content_recommender.recommend_by_category(category, top_n=top_n*2)
            # Since category recommendations don't have similarity scores, we'll assign a default value
            category_recs['similarity'] = 1.0
            category_recs['score'] = category_recs['similarity'] * weights['category']
            category_recs['source'] = 'category'
            all_candidates = pd.concat([all_candidates, category_recs])
        
        # If we don't have any candidates, return empty DataFrame
        if all_candidates.empty:
            return pd.DataFrame(columns=['id', 'title', 'category', 'score', 'source'])
        
        # Group by article ID and aggregate scores
        # For each article that appears in multiple recommendation sources, we'll take the max score
        grouped = all_candidates.groupby('id').agg({
            'title': 'first',
            'category': 'first',
            'score': 'sum',  # Sum scores from different sources
            'source': lambda x: '+'.join(sorted(set(x)))  # Combine sources
        }).reset_index()
        
        # Sort by score and return top N
        recommendations = grouped.sort_values('score', ascending=False).head(top_n)
        return recommendations[['id', 'title', 'category', 'score', 'source']]
    
    def recommend_for_user_with_preferences(self, user_profile, preferred_categories=None, 
                                           disliked_categories=None, top_n=5):
        """
        Generate personalized recommendations with category preferences
        
        Parameters:
        user_profile (numpy.ndarray): User profile embedding
        preferred_categories (list): Categories the user prefers
        disliked_categories (list): Categories the user dislikes
        top_n (int): Number of recommendations to return
        
        Returns:
        DataFrame: Top N recommended articles
        """
        # Get base recommendations using user profile
        base_recs = self.content_recommender.recommend_for_user(user_profile, top_n=top_n*3)
        
        # Apply category preferences
        if preferred_categories:
            # Boost scores for preferred categories
            base_recs['score'] = base_recs.apply(
                lambda row: row['similarity'] * 1.5 if row['category'] in preferred_categories else row['similarity'],
                axis=1
            )
        else:
            base_recs['score'] = base_recs['similarity']
        
        # Filter out disliked categories
        if disliked_categories:
            base_recs = base_recs[~base_recs['category'].isin(disliked_categories)]
        
        # Sort by adjusted score and return top N
        recommendations = base_recs.sort_values('score', ascending=False).head(top_n)
        return recommendations[['id', 'title', 'category', 'score']]
    
    def recommend_diverse(self, user_profile, diversity_weight=0.3, top_n=5):
        """
        Generate diverse recommendations by penalizing similar items
        
        Parameters:
        user_profile (numpy.ndarray): User profile embedding
        diversity_weight (float): Weight for diversity (0-1)
        top_n (int): Number of recommendations to return
        
        Returns:
        DataFrame: Top N diverse recommended articles
        """
        # Get base recommendations
        base_recs = self.content_recommender.recommend_for_user(user_profile, top_n=top_n*3)
        
        # Initialize list to store selected items
        selected_items = []
        selected_indices = []
        
        # Get all embeddings
        embeddings = base_recs['embedding'].tolist()
        
        # Select items one by one
        while len(selected_items) < top_n and not base_recs.empty:
            if not selected_items:
                # Select the first item (highest similarity)
                idx = 0
            else:
                # Calculate diversity penalty
                penalties = []
                for i in range(len(base_recs)):
                    # Calculate max similarity with already selected items
                    embedding = embeddings[i]
                    max_sim = max([cosine_similarity([embedding], [selected_items[j]])[0][0] 
                                  for j in range(len(selected_items))])
                    
                    # Apply penalty: original_score * (1 - diversity_weight * max_similarity)
                    penalty = base_recs.iloc[i]['similarity'] * (1 - diversity_weight * max_sim)
                    penalties.append(penalty)
                
                # Select item with highest penalized score
                idx = np.argmax(penalties)
            
            # Add selected item
            selected_items.append(embeddings[idx])
            selected_indices.append(base_recs.index[idx])
            
            # Remove selected item from candidates
            base_recs = base_recs.drop(base_recs.index[idx])
            embeddings.pop(idx)
        
        # Get the selected recommendations
        diverse_recs = self.content_recommender.df.loc[selected_indices]
        
        return diverse_recs[['id', 'title', 'category']]