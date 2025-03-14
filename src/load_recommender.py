import pickle
import os

def load_recommender(filepath):
    """Load a recommender from a file"""
    with open(filepath, 'rb') as f:
        recommender = pickle.load(f)
    print(f"Recommender loaded from {filepath}")
    return recommender

# Path to the saved recommender
model_path = "c:\\Users\\dario\\projects\\ai-recommender\\models\\content_recommender.pkl"

# Load the recommender
recommender = load_recommender(model_path)

# Print some information about the recommender
print("\nRecommender Information:")
print(f"Type: {type(recommender).__name__}")
print(f"Number of articles: {len(recommender.df)}")
print(f"Categories: {recommender.df['category'].unique()}")

# Show a sample recommendation
if len(recommender.df) > 0:
    article_id = recommender.df['id'].iloc[0]
    print(f"\nSample recommendations for article '{article_id}':")
    similar_articles = recommender.recommend_similar(article_id)
    for _, row in similar_articles.iterrows():
        print(f"- {row['title']} (Similarity: {row['similarity']:.2f})")