# Lesson 5: Collaborative Filtering

## Introduction

In this lesson, we'll explore collaborative filtering, a recommendation technique that relies on user behavior rather than item content. Unlike content-based filtering, collaborative filtering makes recommendations based on the preferences of similar users or items.

## Types of Collaborative Filtering

There are two main types of collaborative filtering:

1. **User-Based Collaborative Filtering**: Recommends items that similar users have liked
2. **Item-Based Collaborative Filtering**: Recommends items similar to those the user has liked in the past

## Creating a User-Item Interaction Matrix

Let's start by creating a user-item interaction matrix:

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item interactions (ratings)
ratings_data = [
    {"user_id": "user1", "item_id": "1", "rating": 5},
    {"user_id": "user1", "item_id": "2", "rating": 3},
    {"user_id": "user1", "item_id": "3", "rating": 4},
    {"user_id": "user2", "item_id": "1", "rating": 4},
    {"user_id": "user2", "item_id": "3", "rating": 5},
    {"user_id": "user2", "item_id": "5", "rating": 3},
    {"user_id": "user3", "item_id": "2", "rating": 4},
    {"user_id": "user3", "item_id": "3", "rating": 3},
    {"user_id": "user3", "item_id": "4", "rating": 5},
    {"user_id": "user4", "item_id": "1", "rating": 5},
    {"user_id": "user4", "item_id": "2", "rating": 2},
    {"user_id": "user4", "item_id": "5", "rating": 4},
    {"user_id": "user5", "item_id": "3", "rating": 4},
    {"user_id": "user5", "item_id": "4", "rating": 5},
    {"user_id": "user5", "item_id": "5", "rating": 3},
]

# Convert to DataFrame
ratings_df = pd.DataFrame(ratings_data)

# Create user-item matrix
user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating')

# Fill missing values with 0 (or you could use the mean rating)
user_item_matrix = user_item_matrix.fillna(0)

print("User-Item Matrix:")
print(user_item_matrix)
```

## User-Based Collaborative Filtering

Let's implement user-based collaborative filtering:

```python
class UserBasedCF:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.user_similarity_matrix = None
        self._compute_similarity()

    def _compute_similarity(self):
        """Compute similarity between users"""
        # Convert DataFrame to numpy array
        matrix = self.user_item_matrix.values

        # Compute cosine similarity
        self.user_similarity_matrix = cosine_similarity(matrix)

        # Convert to DataFrame for easier indexing
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

    def get_similar_users(self, user_id, n=2):
        """Get top N similar users for a given user"""
        # Get similarity scores for the user
        user_similarities = self.user_similarity_matrix[user_id]

        # Sort by similarity (descending) and exclude the user itself
        similar_users = user_similarities.drop(user_id).sort_values(ascending=False).head(n)

        return similar_users

    def predict_rating(self, user_id, item_id, k=2):
        """
        Predict rating for a user-item pair

        Parameters:
        user_id (str): ID of the user
        item_id (str): ID of the item
        k (int): Number of similar users to consider

        Returns:
        float: Predicted rating
        """
        # Check if the user has already rated the item
        if self.user_item_matrix.loc[user_id, item_id] > 0:
            return self.user_item_matrix.loc[user_id, item_id]

        # Get similar users
        similar_users = self.get_similar_users(user_id, n=k)

        # If no similar users found, return average rating for the item
        if len(similar_users) == 0:
            return self.user_item_matrix[item_id].mean()

        # Calculate weighted average of ratings from similar users
        numerator = 0
        denominator = 0

        for sim_user, similarity in similar_users.items():
            # Get the rating for this item from the similar user
            rating = self.user_item_matrix.loc[sim_user, item_id]

            # Skip if the similar user hasn't rated this item
            if rating == 0:
                continue

            numerator += similarity * rating
            denominator += similarity

        # If no similar users have rated this item, return average rating for the item
        if denominator == 0:
            return self.user_item_matrix[item_id].mean()

        # Return the weighted average
        return numerator / denominator

    def recommend_items(self, user_id, n=3, exclude_rated=True):
        """
        Recommend top N items for a user

        Parameters:
        user_id (str): ID of the user
        n (int): Number of recommendations to return
        exclude_rated (bool): Whether to exclude items the user has already rated

        Returns:
        DataFrame: Top N recommended items with predicted ratings
        """
        # Get all items
        all_items = self.user_item_matrix.columns

        # Exclude items the user has already rated if requested
        if exclude_rated:
            user_rated_items = self.user_item_matrix.loc[user_id]
            user_rated_items = user_rated_items[user_rated_items > 0].index
            items_to_predict = [item for item in all_items if item not in user_rated_items]
        else:
            items_to_predict = all_items

        # Predict ratings for all items
        predicted_ratings = []
        for item_id in items_to_predict:
            predicted_rating = self.predict_rating(user_id, item_id)
            predicted_ratings.append({
                'item_id': item_id,
                'predicted_rating': predicted_rating
            })

        # Convert to DataFrame and sort by predicted rating (descending)
        recommendations = pd.DataFrame(predicted_ratings)
        recommendations = recommendations.sort_values('predicted_rating', ascending=False).head(n)

        return recommendations

# Create user-based CF recommender
user_cf = UserBasedCF(user_item_matrix)

# Test similar users
test_user = "user1"
similar_users = user_cf.get_similar_users(test_user)
print(f"\nUsers similar to {test_user}:")
print(similar_users)

# Test rating prediction
test_item = "5"  # An item user1 hasn't rated
predicted_rating = user_cf.predict_rating(test_user, test_item)
print(f"\nPredicted rating for {test_user} on item {test_item}: {predicted_rating:.2f}")

# Test recommendations
recommendations = user_cf.recommend_items(test_user)
print(f"\nRecommendations for {test_user}:")
print(recommendations)
```

## Item-Based Collaborative Filtering

Now, let's implement item-based collaborative filtering:

```python
class ItemBasedCF:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.item_similarity_matrix = None
        self._compute_similarity()

    def _compute_similarity(self):
        """Compute similarity between items"""
        # Transpose the matrix to get item-user matrix
        item_user_matrix = self.user_item_matrix.T

        # Convert DataFrame to numpy array
        matrix = item_user_matrix.values

        # Compute cosine similarity
        self.item_similarity_matrix = cosine_similarity(matrix)

        # Convert to DataFrame for easier indexing
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=item_user_matrix.index,
            columns=item_user_matrix.index
        )

    def get_similar_items(self, item_id, n=2):
        """Get top N similar items for a given item"""
        # Get similarity scores for the item
        item_similarities = self.item_similarity_matrix[item_id]

        # Sort by similarity (descending) and exclude the item itself
        similar_items = item_similarities.drop(item_id).sort_values(ascending=False).head(n)

        return similar_items

    def predict_rating(self, user_id, item_id, k=2):
        """
        Predict rating for a user-item pair

        Parameters:
        user_id (str): ID of the user
        item_id (str): ID of the item
        k (int): Number of similar items to consider

        Returns:
        float: Predicted rating
        """
        # Check if the user has already rated the item
        if self.user_item_matrix.loc[user_id, item_id] > 0:
            return self.user_item_matrix.loc[user_id, item_id]

        # Get items the user has rated
        user_rated_items = self.user_item_matrix.loc[user_id]
        user_rated_items = user_rated_items[user_rated_items > 0]

        # If the user hasn't rated any items, return average rating for the item
        if len(user_rated_items) == 0:
            return self.user_item_matrix[item_id].mean()

        # Get similar items to the target item
        similar_items = self.get_similar_items(item_id, n=k)

        # Calculate weighted average of ratings from similar items
        numerator = 0
        denominator = 0

        for sim_item, similarity in similar_items.items():
            # Check if the user has rated this similar item
            if sim_item in user_rated_items.index:
                # Get the user's rating for this similar item
                rating = user_rated_items[sim_item]

                numerator += similarity * rating
                denominator += similarity

        # If no similar items have been rated by the user, return average rating for the item
        if denominator == 0:
            return self.user_item_matrix[item_id].mean()

        # Return the weighted average
        return numerator / denominator

    def recommend_items(self, user_id, n=3, exclude_rated=True):
        """
        Recommend top N items for a user

        Parameters:
        user_id (str): ID of the user
        n (int): Number of recommendations to return
        exclude_rated (bool): Whether to exclude items the user has already rated

        Returns:
        DataFrame: Top N recommended items with predicted ratings
        """
        # Get all items
        all_items = self.user_item_matrix.columns

        # Exclude items the user has already rated if requested
        if exclude_rated:
            user_rated_items = self.user_item_matrix.loc[user_id]
            user_rated_items = user_rated_items[user_rated_items > 0].index
            items_to_predict = [item for item in all_items if item not in user_rated_items]
        else:
            items_to_predict = all_items

        # Predict ratings for all items
        predicted_ratings = []
        for item_id in items_to_predict:
            predicted_rating = self.predict_rating(user_id, item_id)
            predicted_ratings.append({
                'item_id': item_id,
                'predicted_rating': predicted_rating
            })

        # Convert to DataFrame and sort by predicted rating (descending)
        recommendations = pd.DataFrame(predicted_ratings)
        recommendations = recommendations.sort_values('predicted_rating', ascending=False).head(n)

        return recommendations

# Create item-based CF recommender
item_cf = ItemBasedCF(user_item_matrix)

# Test similar items
test_item = "1"
similar_items = item_cf.get_similar_items(test_item)
print(f"\nItems similar to {test_item}:")
print(similar_items)

# Test rating prediction
test_user = "user1"
test_item = "5"  # An item user1 hasn't rated
predicted_rating = item_cf.predict_rating(test_user, test_item)
print(f"\nPredicted rating for {test_user} on item {test_item} (item-based): {predicted_rating:.2f}")

# Test recommendations
recommendations = item_cf.recommend_items(test_user)
print(f"\nRecommendations for {test_user} (item-based):")
print(recommendations)
```

## Handling Cold Start Problems

Collaborative filtering suffers from the "cold start" problem when we have new users or items with no ratings:

```python
def handle_cold_start(user_id, item_id, user_cf, item_cf, content_recommender=None):
    """
    Handle cold start problem for new users or items

    Parameters:
    user_id (str): ID of the user
    item_id (str): ID of the item
    user_cf (UserBasedCF): User-based CF recommender
    item_cf (ItemBasedCF): Item-based CF recommender
    content_recommender: Content-based recommender (optional)

    Returns:
    float: Predicted rating
    """
    # Check if the user has any ratings
    user_ratings = user_cf.user_item_matrix.loc[user_id]
    user_has_ratings = (user_ratings > 0).any()

    # Check if the item has any ratings
    item_ratings = user_cf.user_item_matrix[item_id]
    item_has_ratings = (item_ratings > 0).any()

    if user_has_ratings and item_has_ratings:
        # If both user and item have ratings, use collaborative filtering
        # Average predictions from user-based and item-based CF
        user_pred = user_cf.predict_rating(user_id, item_id)
        item_pred = item_cf.predict_rating(user_id, item_id)
        return (user_pred + item_pred) / 2

    elif user_has_ratings and not item_has_ratings:
        # If only the user has ratings, use user-based CF
        return user_cf.predict_rating(user_id, item_id)

    elif not user_has_ratings and item_has_ratings:
        # If only the item has ratings, use item-based CF
        return item_cf.predict_rating(user_id, item_id)

    else:
        # If neither user nor item has ratings, use content-based filtering if available
        if content_recommender is not None:
            # This would depend on your content-based recommender implementation
            return content_recommender.predict_rating(user_id, item_id)
        else:
            # Otherwise, return the global average rating
            return user_cf.user_item_matrix.values.mean()

# Add a new user with no ratings
new_user_id = "new_user"
user_item_matrix.loc[new_user_id] = 0

# Test cold start handling
```

I'll continue with the rest of Lesson 5 on Collaborative Filtering. Here's the completion of the file:

```markdown:c:\Users\dario\projects\ai-recommender\docs\lessons\05-collaborative-filtering.md
# Add a new user with no ratings
new_user_id = "new_user"
user_item_matrix.loc[new_user_id] = 0

# Test cold start handling
test_item = "3"
predicted_rating = handle_cold_start(new_user_id, test_item, user_cf, item_cf)
print(f"\nPredicted rating for new user on item {test_item}: {predicted_rating:.2f}")

# Add a new item with no ratings
new_item_id = "new_item"
user_item_matrix[new_item_id] = 0

# Test cold start handling for new item
predicted_rating = handle_cold_start(test_user, new_item_id, user_cf, item_cf)
print(f"\nPredicted rating for {test_user} on new item: {predicted_rating:.2f}")
```

## Matrix Factorization

Matrix factorization is a more advanced collaborative filtering technique that decomposes the user-item matrix into lower-dimensional matrices:

```python
from sklearn.decomposition import NMF

class MatrixFactorizationCF:
    def __init__(self, user_item_matrix, n_factors=3):
        self.user_item_matrix = user_item_matrix
        self.n_factors = n_factors
        self.user_features = None
        self.item_features = None
        self._factorize_matrix()

    def _factorize_matrix(self):
        """Factorize the user-item matrix using NMF"""
        # Create NMF model
        model = NMF(n_components=self.n_factors, init='random', random_state=42)

        # Fit the model to the user-item matrix
        self.user_features = model.fit_transform(self.user_item_matrix)

        # Get item features
        self.item_features = model.components_.T

        # Convert to DataFrames for easier indexing
        self.user_features = pd.DataFrame(
            self.user_features,
            index=self.user_item_matrix.index,
            columns=[f'factor_{i}' for i in range(self.n_factors)]
        )

        self.item_features = pd.DataFrame(
            self.item_features,
            index=self.user_item_matrix.columns,
            columns=[f'factor_{i}' for i in range(self.n_factors)]
        )

    def predict_rating(self, user_id, item_id):
        """
        Predict rating for a user-item pair

        Parameters:
        user_id (str): ID of the user
        item_id (str): ID of the item

        Returns:
        float: Predicted rating
        """
        # Check if the user and item exist in the matrix
        if user_id not in self.user_features.index or item_id not in self.item_features.index:
            # Return the global average rating if user or item is new
            return self.user_item_matrix.values.mean()

        # Get user and item features
        user_vec = self.user_features.loc[user_id].values
        item_vec = self.item_features.loc[item_id].values

        # Calculate predicted rating (dot product)
        predicted_rating = np.dot(user_vec, item_vec)

        return predicted_rating

    def recommend_items(self, user_id, n=3, exclude_rated=True):
        """
        Recommend top N items for a user

        Parameters:
        user_id (str): ID of the user
        n (int): Number of recommendations to return
        exclude_rated (bool): Whether to exclude items the user has already rated

        Returns:
        DataFrame: Top N recommended items with predicted ratings
        """
        # Check if the user exists in the matrix
        if user_id not in self.user_features.index:
            # Return empty DataFrame if user is new
            return pd.DataFrame(columns=['item_id', 'predicted_rating'])

        # Get all items
        all_items = self.item_features.index

        # Exclude items the user has already rated if requested
        if exclude_rated:
            user_rated_items = self.user_item_matrix.loc[user_id]
            user_rated_items = user_rated_items[user_rated_items > 0].index
            items_to_predict = [item for item in all_items if item not in user_rated_items]
        else:
            items_to_predict = all_items

        # Predict ratings for all items
        predicted_ratings = []
        for item_id in items_to_predict:
            predicted_rating = self.predict_rating(user_id, item_id)
            predicted_ratings.append({
                'item_id': item_id,
                'predicted_rating': predicted_rating
            })

        # Convert to DataFrame and sort by predicted rating (descending)
        recommendations = pd.DataFrame(predicted_ratings)
        recommendations = recommendations.sort_values('predicted_rating', ascending=False).head(n)

        return recommendations

# Create matrix factorization CF recommender
mf_cf = MatrixFactorizationCF(user_item_matrix)

# Test rating prediction
test_user = "user1"
test_item = "5"  # An item user1 hasn't rated
predicted_rating = mf_cf.predict_rating(test_user, test_item)
print(f"\nPredicted rating for {test_user} on item {test_item} (matrix factorization): {predicted_rating:.2f}")

# Test recommendations
recommendations = mf_cf.recommend_items(test_user)
print(f"\nRecommendations for {test_user} (matrix factorization):")
print(recommendations)

# Visualize user and item features
import matplotlib.pyplot as plt

def plot_latent_factors(user_features, item_features, n_factors=2):
    """
    Plot users and items in the latent factor space

    Parameters:
    user_features (DataFrame): User features
    item_features (DataFrame): Item features
    n_factors (int): Number of factors to plot (must be 2 or 3)
    """
    if n_factors not in [2, 3]:
        raise ValueError("n_factors must be 2 or 3")

    if n_factors == 2:
        # 2D plot
        plt.figure(figsize=(10, 8))

        # Plot users
        plt.scatter(
            user_features['factor_0'],
            user_features['factor_1'],
            color='blue',
            label='Users'
        )

        # Add user labels
        for user_id in user_features.index:
            plt.annotate(
                user_id,
                (user_features.loc[user_id, 'factor_0'], user_features.loc[user_id, 'factor_1']),
                xytext=(5, 5),
                textcoords='offset points'
            )

        # Plot items
        plt.scatter(
            item_features['factor_0'],
            item_features['factor_1'],
            color='red',
            label='Items'
        )

        # Add item labels
        for item_id in item_features.index:
            plt.annotate(
                f"Item {item_id}",
                (item_features.loc[item_id, 'factor_0'], item_features.loc[item_id, 'factor_1']),
                xytext=(5, 5),
                textcoords='offset points'
            )

        plt.xlabel('Factor 0')
        plt.ylabel('Factor 1')
        plt.title('Users and Items in 2D Latent Factor Space')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        # 3D plot
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot users
        ax.scatter(
            user_features['factor_0'],
            user_features['factor_1'],
            user_features['factor_2'],
            color='blue',
            label='Users'
        )

        # Add user labels
        for user_id in user_features.index:
            ax.text(
                user_features.loc[user_id, 'factor_0'],
                user_features.loc[user_id, 'factor_1'],
                user_features.loc[user_id, 'factor_2'],
                user_id
            )

        # Plot items
        ax.scatter(
            item_features['factor_0'],
            item_features['factor_1'],
            item_features['factor_2'],
            color='red',
            label='Items'
        )

        # Add item labels
        for item_id in item_features.index:
            ax.text(
                item_features.loc[item_id, 'factor_0'],
                item_features.loc[item_id, 'factor_1'],
                item_features.loc[item_id, 'factor_2'],
                f"Item {item_id}"
            )

        ax.set_xlabel('Factor 0')
        ax.set_ylabel('Factor 1')
        ax.set_zlabel('Factor 2')
        ax.set_title('Users and Items in 3D Latent Factor Space')
        ax.legend()
        plt.show()

# Plot the first 2 latent factors
plot_latent_factors(mf_cf.user_features, mf_cf.item_features, n_factors=2)
```

## Hybrid Collaborative Filtering

Let's create a hybrid recommender that combines user-based, item-based, and matrix factorization approaches:

```python
class HybridCF:
    def __init__(self, user_item_matrix, weights=None):
        self.user_item_matrix = user_item_matrix

        # Set default weights if not provided
        if weights is None:
            self.weights = {
                'user_cf': 0.3,
                'item_cf': 0.3,
                'mf_cf': 0.4
            }
        else:
            self.weights = weights

        # Initialize recommenders
        self.user_cf = UserBasedCF(user_item_matrix)
        self.item_cf = ItemBasedCF(user_item_matrix)
        self.mf_cf = MatrixFactorizationCF(user_item_matrix)

    def predict_rating(self, user_id, item_id):
        """
        Predict rating for a user-item pair using a weighted average of different CF approaches

        Parameters:
        user_id (str): ID of the user
        item_id (str): ID of the item

        Returns:
        float: Predicted rating
        """
        # Handle new users or items
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return handle_cold_start(user_id, item_id, self.user_cf, self.item_cf)

        # Get predictions from each recommender
        user_cf_pred = self.user_cf.predict_rating(user_id, item_id)
        item_cf_pred = self.item_cf.predict_rating(user_id, item_id)
        mf_cf_pred = self.mf_cf.predict_rating(user_id, item_id)

        # Calculate weighted average
        weighted_pred = (
            self.weights['user_cf'] * user_cf_pred +
            self.weights['item_cf'] * item_cf_pred +
            self.weights['mf_cf'] * mf_cf_pred
        )

        return weighted_pred

    def recommend_items(self, user_id, n=3, exclude_rated=True):
        """
        Recommend top N items for a user

        Parameters:
        user_id (str): ID of the user
        n (int): Number of recommendations to return
        exclude_rated (bool): Whether to exclude items the user has already rated

        Returns:
        DataFrame: Top N recommended items with predicted ratings
        """
        # Get all items
        all_items = self.user_item_matrix.columns

        # Exclude items the user has already rated if requested
        if exclude_rated and user_id in self.user_item_matrix.index:
            user_rated_items = self.user_item_matrix.loc[user_id]
            user_rated_items = user_rated_items[user_rated_items > 0].index
            items_to_predict = [item for item in all_items if item not in user_rated_items]
        else:
            items_to_predict = all_items

        # Predict ratings for all items
        predicted_ratings = []
        for item_id in items_to_predict:
            predicted_rating = self.predict_rating(user_id, item_id)
            predicted_ratings.append({
                'item_id': item_id,
                'predicted_rating': predicted_rating
            })

        # Convert to DataFrame and sort by predicted rating (descending)
        recommendations = pd.DataFrame(predicted_ratings)
        recommendations = recommendations.sort_values('predicted_rating', ascending=False).head(n)

        return recommendations

# Create hybrid CF recommender
hybrid_cf = HybridCF(user_item_matrix)

# Test recommendations
test_user = "user1"
recommendations = hybrid_cf.recommend_items(test_user)
print(f"\nHybrid recommendations for {test_user}:")
print(recommendations)

# Test with new user
recommendations = hybrid_cf.recommend_items(new_user_id)
print(f"\nHybrid recommendations for new user:")
print(recommendations)
```

## Evaluating Collaborative Filtering

Let's implement some evaluation metrics for our collaborative filtering recommenders:

````python
def evaluate_recommender(recommender, test_data, k=5):
    """
    Evaluate a recommender system

    Parameters:
    recommender: Recommender object with predict_rating method
    test_data (list): List of (user_id, item_id, rating) tuples
    k (int): Number of recommendations to consider

    Returns:
    dict: Dictionary with evaluation metrics
    """
    # Calculate prediction error metrics
    mae_sum = 0
    rmse_sum = 0
    count = 0

    for user_id, item_id, true_rating in test_data:
        predicted_rating = recommender.predict_rating(user_id, item_id)
        error = true_rating - predicted_rating

        mae_sum += abs(error)
        rmse_sum += error ** 2
        count += 1

    mae = mae_sum / count if count > 0 else float('inf')
    rmse = (rmse_sum / count) ** 0.5 if count > 0 else float('inf')

    # Calculate recommendation metrics
    precision_sum = 0
    recall_sum = 0
    ndcg_sum = 0
    user_count = 0

    # Group test data by user
    user_test_items = {}
    for user_id, item_id, rating in test_data:
        if user_id not in user_test_items:
            user_test_items[user_id] = []
        user_test_items[user_id].append((item_id, rating))

    # Calculate metrics for each user
    for user_id, items in user_test_items.items():
        # Get relevant items (items with high ratings)
        relevant_items = [item_id for item_id, rating in items if rating >= 4]

        if not relevant_items:
            continue

        # Get recommendations for the user
        recommendations = recommender.recommend_items(user_id, n=k)
        recommended_items = recommendations['item_id'].tolist()

        # Calculate precision and recall
I'll continue with the evaluation metrics implementation for the collaborative filtering recommender. Here's the rest of the code:

```markdown:c:\Users\dario\projects\ai-recommender\docs\lessons\05-collaborative-filtering.md
        # Calculate precision and recall
        relevant_and_recommended = set(relevant_items).intersection(set(recommended_items))

        precision = len(relevant_and_recommended) / len(recommended_items) if recommended_items else 0
        recall = len(relevant_and_recommended) / len(relevant_items) if relevant_items else 0

        # Calculate NDCG (Normalized Discounted Cumulative Gain)
        # DCG = sum(rel_i / log2(i+1)) where rel_i is the relevance of item at position i
        dcg = 0
        idcg = 0

        # Calculate DCG
        for i, item_id in enumerate(recommended_items):
            if item_id in relevant_items:
                # Relevance is 1 if the item is relevant
                dcg += 1 / np.log2(i + 2)  # i+2 because i starts from 0

        # Calculate IDCG (Ideal DCG)
        for i in range(min(len(relevant_items), k)):
            idcg += 1 / np.log2(i + 2)

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0

        precision_sum += precision
        recall_sum += recall
        ndcg_sum += ndcg
        user_count += 1

    # Calculate average metrics
    avg_precision = precision_sum / user_count if user_count > 0 else 0
    avg_recall = recall_sum / user_count if user_count > 0 else 0
    avg_ndcg = ndcg_sum / user_count if user_count > 0 else 0

    # Calculate F1 score
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return {
        'MAE': mae,
        'RMSE': rmse,
        'Precision@K': avg_precision,
        'Recall@K': avg_recall,
        'F1@K': f1,
        'NDCG@K': avg_ndcg
    }

# Create test data (hold-out set)
test_data = [
    ("user1", "4", 4),  # User1 would rate item 4 as 4
    ("user2", "2", 3),  # User2 would rate item 2 as 3
    ("user3", "5", 4),  # User3 would rate item 5 as 4
    ("user4", "3", 5),  # User4 would rate item 3 as 5
    ("user5", "1", 4)   # User5 would rate item 1 as 4
]

# Evaluate different recommenders
print("\nEvaluating User-Based CF:")
user_cf_metrics = evaluate_recommender(user_cf, test_data)
print(user_cf_metrics)

print("\nEvaluating Item-Based CF:")
item_cf_metrics = evaluate_recommender(item_cf, test_data)
print(item_cf_metrics)

print("\nEvaluating Matrix Factorization CF:")
mf_cf_metrics = evaluate_recommender(mf_cf, test_data)
print(mf_cf_metrics)

print("\nEvaluating Hybrid CF:")
hybrid_cf_metrics = evaluate_recommender(hybrid_cf, test_data)
print(hybrid_cf_metrics)

# Compare metrics visually
metrics = ['MAE', 'RMSE', 'Precision@K', 'Recall@K', 'F1@K', 'NDCG@K']
recommenders = ['User-Based CF', 'Item-Based CF', 'Matrix Factorization', 'Hybrid CF']
results = pd.DataFrame({
    'User-Based CF': list(user_cf_metrics.values()),
    'Item-Based CF': list(item_cf_metrics.values()),
    'Matrix Factorization': list(mf_cf_metrics.values()),
    'Hybrid CF': list(hybrid_cf_metrics.values())
}, index=metrics)

# Plot results
plt.figure(figsize=(12, 8))

# Plot error metrics (lower is better)
plt.subplot(2, 1, 1)
results.loc[['MAE', 'RMSE']].plot(kind='bar')
plt.title('Error Metrics (lower is better)')
plt.ylabel('Error')
plt.grid(axis='y')

# Plot recommendation metrics (higher is better)
plt.subplot(2, 1, 2)
results.loc[['Precision@K', 'Recall@K', 'F1@K', 'NDCG@K']].plot(kind='bar')
plt.title('Recommendation Metrics (higher is better)')
plt.ylabel('Score')
plt.grid(axis='y')

plt.tight_layout()
plt.show()
````

## Optimizing Collaborative Filtering

Let's optimize our collaborative filtering recommenders by tuning hyperparameters:

```python
def optimize_cf_parameters(user_item_matrix, test_data):
    """
    Find optimal parameters for collaborative filtering

    Parameters:
    user_item_matrix (DataFrame): User-item matrix
    test_data (list): List of (user_id, item_id, rating) tuples

    Returns:
    dict: Dictionary with optimal parameters
    """
    # Parameters to try
    k_values = [1, 2, 3, 5, 10]  # Number of neighbors
    n_factors_values = [2, 3, 5, 10]  # Number of latent factors

    # Initialize best parameters
    best_params = {
        'user_cf_k': 2,
        'item_cf_k': 2,
        'mf_n_factors': 3,
        'hybrid_weights': {'user_cf': 0.3, 'item_cf': 0.3, 'mf_cf': 0.4}
    }

    best_rmse = float('inf')

    # Try different k values for user-based CF
    print("\nOptimizing User-Based CF...")
    for k in k_values:
        # Create recommender with this k
        recommender = UserBasedCF(user_item_matrix)

        # Override the predict_rating method to use this k
        def predict_with_k(user_id, item_id):
            return recommender.predict_rating(user_id, item_id, k=k)

        recommender.predict_rating = predict_with_k

        # Evaluate
        metrics = evaluate_recommender(recommender, test_data)
        rmse = metrics['RMSE']

        print(f"k={k}, RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params['user_cf_k'] = k

    # Try different k values for item-based CF
    print("\nOptimizing Item-Based CF...")
    best_rmse = float('inf')

    for k in k_values:
        # Create recommender with this k
        recommender = ItemBasedCF(user_item_matrix)

        # Override the predict_rating method to use this k
        def predict_with_k(user_id, item_id):
            return recommender.predict_rating(user_id, item_id, k=k)

        recommender.predict_rating = predict_with_k

        # Evaluate
        metrics = evaluate_recommender(recommender, test_data)
        rmse = metrics['RMSE']

        print(f"k={k}, RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params['item_cf_k'] = k

    # Try different n_factors values for matrix factorization
    print("\nOptimizing Matrix Factorization...")
    best_rmse = float('inf')

    for n_factors in n_factors_values:
        # Create recommender with this n_factors
        recommender = MatrixFactorizationCF(user_item_matrix, n_factors=n_factors)

        # Evaluate
        metrics = evaluate_recommender(recommender, test_data)
        rmse = metrics['RMSE']

        print(f"n_factors={n_factors}, RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params['mf_n_factors'] = n_factors

    # Try different weight combinations for hybrid CF
    print("\nOptimizing Hybrid CF...")
    best_rmse = float('inf')

    weight_combinations = [
        {'user_cf': 0.6, 'item_cf': 0.2, 'mf_cf': 0.2},
        {'user_cf': 0.2, 'item_cf': 0.6, 'mf_cf': 0.2},
        {'user_cf': 0.2, 'item_cf': 0.2, 'mf_cf': 0.6},
        {'user_cf': 0.4, 'item_cf': 0.4, 'mf_cf': 0.2},
        {'user_cf': 0.4, 'item_cf': 0.2, 'mf_cf': 0.4},
        {'user_cf': 0.2, 'item_cf': 0.4, 'mf_cf': 0.4},
        {'user_cf': 0.33, 'item_cf': 0.33, 'mf_cf': 0.34}
    ]

    for weights in weight_combinations:
        # Create recommender with these weights
        recommender = HybridCF(user_item_matrix, weights=weights)

        # Evaluate
        metrics = evaluate_recommender(recommender, test_data)
        rmse = metrics['RMSE']

        print(f"weights={weights}, RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params['hybrid_weights'] = weights

    return best_params

# Find optimal parameters
optimal_params = optimize_cf_parameters(user_item_matrix, test_data)
print("\nOptimal parameters:")
print(optimal_params)

# Create optimized hybrid recommender
optimized_hybrid_cf = HybridCF(
    user_item_matrix,
    weights=optimal_params['hybrid_weights']
)

# Evaluate optimized recommender
print("\nEvaluating Optimized Hybrid CF:")
optimized_metrics = evaluate_recommender(optimized_hybrid_cf, test_data)
print(optimized_metrics)
```

## Exercise

1. Implement a function to handle implicit feedback (e.g., clicks, views) in collaborative filtering
2. Extend the matrix factorization approach to include user and item biases
3. Implement a time-aware collaborative filtering algorithm that considers the recency of ratings
4. Create a visualization that shows how recommendations change as a user rates more items
5. Implement a cross-validation procedure to evaluate your recommender more robustly

## Next Steps

In the next lesson, we'll explore hybrid recommendation systems that combine content-based and collaborative filtering approaches for even better recommendations.

## Resources

- [Collaborative Filtering - Wikipedia](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- [Evaluating Recommendation Systems](https://www.microsoft.com/en-us/research/publication/evaluating-recommendation-systems/)
- [Scikit-learn NMF Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [Surprise Library for Recommender Systems](https://surprise.readthedocs.io/en/stable/)
