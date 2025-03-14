import pandas as pd
import numpy as np
from recommender import ContentBasedRecommender
from hybrid_recommender import HybridRecommender

def main():
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
        },
        {
            "id": "6",
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has revolutionized computer vision, natural language processing, and many other fields.",
            "category": "AI"
        },
        {
            "id": "7",
            "title": "Web Development with React",
            "content": "React is a JavaScript library for building user interfaces. It allows developers to create reusable UI components and manage application state efficiently.",
            "category": "Programming"
        },
        {
            "id": "8",
            "title": "Natural Language Processing Techniques",
            "content": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It includes tasks like sentiment analysis, named entity recognition, and machine translation.",
            "category": "AI"
        },
        {
            "id": "9",
            "title": "Data Visualization with Python",
            "content": "Data visualization is the graphical representation of information and data. Python offers libraries like Matplotlib, Seaborn, and Plotly for creating informative and attractive visualizations.",
            "category": "Data Science"
        },
        {
            "id": "10",
            "title": "Introduction to SQL",
            "content": "SQL (Structured Query Language) is a domain-specific language used for managing and querying relational databases. It's essential for data analysis and backend development.",
            "category": "Programming"
        }
    ]

    # Convert to DataFrame
    df = pd.DataFrame(articles)
    print("Dataset:")
    print(df[["id", "title", "category"]])
    print("\n")

    # Create and fit the recommender
    content_recommender = ContentBasedRecommender()
    content_recommender.fit(df)
    
    # Create hybrid recommender
    hybrid_recommender = HybridRecommender(content_recommender)
    
    # Test hybrid recommendations
    print("Testing hybrid recommendations:")
    
    # User liked articles
    user_liked_articles = ["1", "3"]  # User likes ML and Neural Networks
    user_profile = content_recommender.create_user_profile(user_liked_articles)
    
    # Get hybrid recommendations using multiple signals
    hybrid_recs = hybrid_recommender.recommend_hybrid(
        article_id="1",  # Similar to Intro to ML
        user_profile=user_profile,  # Based on user's liked articles
        category="AI",  # From AI category
        weights={'content': 0.3, 'user': 0.5, 'category': 0.2}  # Custom weights
    )
    
    print("Hybrid recommendations:")
    print(hybrid_recs)
    print("\n")
    
    # Test recommendations with category preferences
    preferred_categories = ["AI"]
    disliked_categories = ["Programming"]
    
    print("Recommendations with category preferences:")
    print(f"Preferred: {preferred_categories}, Disliked: {disliked_categories}")
    pref_recs = hybrid_recommender.recommend_for_user_with_preferences(
        user_profile, 
        preferred_categories=preferred_categories,
        disliked_categories=disliked_categories
    )
    print(pref_recs)
    print("\n")
    
    # Test diverse recommendations
    print("Diverse recommendations:")
    diverse_recs = hybrid_recommender.recommend_diverse(user_profile, diversity_weight=0.5)
    print(diverse_recs)

if __name__ == "__main__":
    main()