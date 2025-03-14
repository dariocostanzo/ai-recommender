import pandas as pd
import numpy as np
import os
from recommender import ContentBasedRecommender, save_recommender, load_recommender
from hybrid_recommender import HybridRecommender

def main():
    print("=== AI Recommender System Demo ===\n")
    
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
    print("Dataset loaded with", len(df), "articles")
    print("Categories:", df['category'].unique())
    print("\n")

    print("Creating content-based recommender...")
    # Create and fit the recommender
    content_recommender = ContentBasedRecommender()
    content_recommender.fit(df)
    print("Content-based recommender created successfully!")
    
    print("\nCreating hybrid recommender...")
    # Create hybrid recommender
    hybrid_recommender = HybridRecommender(content_recommender)
    print("Hybrid recommender created successfully!")
    
    # Demo 1: Content-based recommendations
    print("\n=== Demo 1: Content-based Recommendations ===")
    target_article_id = "1"  # "Introduction to Machine Learning"
    target_article = content_recommender.get_article_by_id(target_article_id)
    print(f"Finding articles similar to: {target_article['title']}")
    
    similar_articles = content_recommender.recommend_similar(target_article_id)
    print("\nSimilar articles:")
    for _, row in similar_articles.iterrows():
        print(f"- {row['title']} (Category: {row['category']}, Similarity: {row['similarity']:.2f})")
    
    # Demo 2: Category-based recommendations
    print("\n=== Demo 2: Category-based Recommendations ===")
    category = "AI"
    print(f"Top articles in '{category}' category:")
    category_articles = content_recommender.recommend_by_category(category)
    for _, row in category_articles.iterrows():
        print(f"- {row['title']} (ID: {row['id']})")
    
    # Demo 3: User profile recommendations
    print("\n=== Demo 3: User Profile Recommendations ===")
    user_liked_articles = ["1", "3"]  # User likes ML and Neural Networks
    liked_titles = [content_recommender.get_article_by_id(id)['title'] for id in user_liked_articles]
    print(f"User has liked: {', '.join(liked_titles)}")
    
    user_profile = content_recommender.create_user_profile(user_liked_articles)
    print("Created user profile based on liked articles")
    
    print("\nRecommendations for this user:")
    user_recommendations = content_recommender.recommend_for_user(user_profile, exclude_ids=user_liked_articles)
    for _, row in user_recommendations.iterrows():
        print(f"- {row['title']} (Category: {row['category']}, Similarity: {row['similarity']:.2f})")
    
    # Demo 4: Hybrid recommendations
    print("\n=== Demo 4: Hybrid Recommendations ===")
    print("Combining content-based, user-based, and category-based signals")
    
    hybrid_recs = hybrid_recommender.recommend_hybrid(
        article_id="1",  # Similar to Intro to ML
        user_profile=user_profile,  # Based on user's liked articles
        category="AI",  # From AI category
        weights={'content': 0.3, 'user': 0.5, 'category': 0.2}  # Custom weights
    )
    
    print("\nHybrid recommendations:")
    for _, row in hybrid_recs.iterrows():
        print(f"- {row['title']} (Category: {row['category']}, Score: {row['score']:.2f}, Source: {row['source']})")
    
    # Demo 5: Recommendations with preferences
    print("\n=== Demo 5: Recommendations with Category Preferences ===")
    preferred_categories = ["AI"]
    disliked_categories = ["Programming"]
    
    print(f"User preferences: Likes {preferred_categories}, Dislikes {disliked_categories}")
    pref_recs = hybrid_recommender.recommend_for_user_with_preferences(
        user_profile, 
        preferred_categories=preferred_categories,
        disliked_categories=disliked_categories
    )
    
    print("\nPersonalized recommendations with preferences:")
    for _, row in pref_recs.iterrows():
        print(f"- {row['title']} (Category: {row['category']}, Score: {row['score']:.2f})")
    
    # Save the recommender
    print("\nSaving recommender models...")
    models_dir = "c:\\Users\\dario\\projects\\ai-recommender\\models"
    os.makedirs(models_dir, exist_ok=True)
    save_recommender(content_recommender, os.path.join(models_dir, "content_recommender.pkl"))
    print("Recommender models saved successfully!")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()