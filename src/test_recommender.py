import pandas as pd
import numpy as np
import os
from recommender import ContentBasedRecommender, save_recommender, load_recommender, evaluate_recommendations

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
    recommender = ContentBasedRecommender()
    recommender.fit(df)

    # Test article-based recommendations
    target_article_id = "1"  # "Introduction to Machine Learning"
    similar_articles = recommender.recommend_similar(target_article_id)
    print(f"Articles similar to '{df[df['id'] == target_article_id]['title'].iloc[0]}':")
    print(similar_articles)
    print("\n")

    # Test category-based recommendations
    print("Top articles in 'AI' category:")
    print(recommender.recommend_by_category('AI'))
    print("\n")

    # Test user-based recommendations
    user_liked_articles = ["1", "3"]  # User likes ML and Neural Networks
    user_profile = recommender.create_user_profile(user_liked_articles)
    print("Recommendations for user who liked articles about ML and Neural Networks:")
    user_recommendations = recommender.recommend_for_user(user_profile, exclude_ids=user_liked_articles)
    print(user_recommendations)
    print("\n")

    # Test text query recommendations
    query = "machine learning applications"
    print(f"Recommendations for query: '{query}'")
    query_recommendations = recommender.recommend_by_text_query(query)
    print(query_recommendations)
    print("\n")

    # Save the recommender
    models_dir = "c:\\Users\\dario\\projects\\ai-recommender\\models"
    os.makedirs(models_dir, exist_ok=True)
    save_recommender(recommender, os.path.join(models_dir, "content_recommender.pkl"))

    # Load the recommender
    loaded_recommender = load_recommender(os.path.join(models_dir, "content_recommender.pkl"))

    # Evaluate recommendations
    true_relevant = ["2", "5"]  # Let's say these are truly relevant for our user
    recommended = user_recommendations['id'].tolist()
    metrics = evaluate_recommendations(true_relevant, recommended)
    print("Recommendation evaluation:")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1 Score: {metrics['f1_score']:.2f}")

if __name__ == "__main__":
    main()