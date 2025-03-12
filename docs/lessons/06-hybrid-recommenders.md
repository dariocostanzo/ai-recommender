# Lesson 6: Hybrid Recommendation Systems

## Introduction

In this lesson, we'll explore hybrid recommendation systems that combine multiple recommendation techniques to overcome the limitations of individual approaches. Hybrid systems can provide more accurate and robust recommendations by leveraging the strengths of different methods.

## Why Hybrid Recommenders?

Each recommendation approach has its own strengths and weaknesses:

- **Content-based filtering**: Good at capturing user preferences for specific item attributes but suffers from overspecialization
- **Collaborative filtering**: Good at discovering patterns across users but suffers from cold-start problems
- **Knowledge-based**: Good at incorporating domain knowledge but difficult to scale

Hybrid recommenders combine these approaches to address their individual limitations.

## Types of Hybrid Recommendation Systems

There are several ways to combine recommendation techniques:

1. **Weighted**: Combines the scores of different recommenders using a weighted average
2. **Switching**: Selects the most appropriate recommender based on the context
3. **Cascading**: Uses one recommender to refine the recommendations of another
4. **Feature Combination**: Uses features from one technique as input to another
5. **Feature Augmentation**: Uses the output of one technique as input features to another
6. **Meta-level**: Uses the model learned by one technique as input to another

Let's implement some of these approaches.

## Weighted Hybrid Recommender

A weighted hybrid recommender combines the scores from multiple recommenders using a weighted average:

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class WeightedHybridRecommender:
    def __init__(self, content_recommender, collaborative_recommender, weights=None):
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender

        # Set default weights if not provided
        if weights is None:
            self.weights = {
                'content': 0.4,
                'collaborative': 0.6
            }
        else:
            self.weights = weights

    def recommend_items(self, user_id, n=5):
        """
        Recommend items for a user using a weighted combination of recommenders

        Parameters:
        user_id (str): ID of the user
        n (int): Number of recommendations to return

        Returns:
        DataFrame: Top N recommended items with scores
        """
        # Get recommendations from each recommender
        content_recs = self.content_recommender.recommend_items(user_id, n=n*2)
        collab_recs = self.collaborative_recommender.recommend_items(user_id, n=n*2)

        # Normalize scores to [0, 1] range for each recommender
        if not content_recs.empty:
            content_max = content_recs['score'].max()
            content_min = content_recs['score'].min()
            if content_max > content_min:
                content_recs['normalized_score'] = (content_recs['score'] - content_min) / (content_max - content_min)
            else:
                content_recs['normalized_score'] = 1.0

        if not collab_recs.empty:
            collab_max = collab_recs['predicted_rating'].max()
            collab_min = collab_recs['predicted_rating'].min()
            if collab_max > collab_min:
                collab_recs['normalized_score'] = (collab_recs['predicted_rating'] - collab_min) / (collab_max - collab_min)
            else:
                collab_recs['normalized_score'] = 1.0

        # Combine recommendations
        all_items = set()
        if not content_recs.empty:
            all_items.update(content_recs['item_id'])
        if not collab_recs.empty:
            all_items.update(collab_recs['item_id'])

        # Calculate weighted scores for each item
        hybrid_scores = []
        for item_id in all_items:
            # Get content-based score
            content_score = 0
            if not content_recs.empty and item_id in content_recs['item_id'].values:
                content_score = content_recs.loc[content_recs['item_id'] == item_id, 'normalized_score'].values[0]

            # Get collaborative score
            collab_score = 0
            if not collab_recs.empty and item_id in collab_recs['item_id'].values:
                collab_score = collab_recs.loc[collab_recs['item_id'] == item_id, 'normalized_score'].values[0]

            # Calculate weighted score
            weighted_score = (
                self.weights['content'] * content_score +
                self.weights['collaborative'] * collab_score
            )

            hybrid_scores.append({
                'item_id': item_id,
                'score': weighted_score,
                'content_score': content_score,
                'collab_score': collab_score
            })

        # Convert to DataFrame and sort by score (descending)
        recommendations = pd.DataFrame(hybrid_scores)
        recommendations = recommendations.sort_values('score', ascending=False).head(n)

        return recommendations
## Switching Hybrid Recommender

A switching hybrid recommender selects the most appropriate recommender based on the context:

```python
class SwitchingHybridRecommender:
    def __init__(self, content_recommender, collaborative_recommender, knowledge_recommender=None):
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.knowledge_recommender = knowledge_recommender

    def recommend_items(self, user_id, n=5, user_item_matrix=None, item_metadata=None):
        """
        Recommend items for a user by switching between recommenders

        Parameters:
        user_id (str): ID of the user
        n (int): Number of recommendations to return
        user_item_matrix (DataFrame): User-item interaction matrix
        item_metadata (DataFrame): Item metadata

        Returns:
        DataFrame: Top N recommended items with scores
        """
        # Check if the user is new (cold start)
        is_new_user = False
        if user_item_matrix is not None and user_id not in user_item_matrix.index:
            is_new_user = True
        elif user_item_matrix is not None:
            user_ratings = user_item_matrix.loc[user_id]
            if (user_ratings > 0).sum() < 3:  # Less than 3 ratings
                is_new_user = True

        # Check if we have item metadata
        has_metadata = item_metadata is not None and not item_metadata.empty

        # Select the appropriate recommender
        if is_new_user and has_metadata:
            # Use content-based for new users if we have metadata
            print(f"Using content-based recommender for new user {user_id}")
            return self.content_recommender.recommend_items(user_id, n=n)
        elif is_new_user and self.knowledge_recommender is not None:
            # Use knowledge-based for new users if available
            print(f"Using knowledge-based recommender for new user {user_id}")
            return self.knowledge_recommender.recommend_items(user_id, n=n)
        else:
            # Use collaborative filtering for existing users
            print(f"Using collaborative recommender for existing user {user_id}")
            return self.collaborative_recommender.recommend_items(user_id, n=n)
````

## Cascading Hybrid Recommender

A cascading hybrid recommender uses one recommender to refine the recommendations of another:

```python
class CascadingHybridRecommender:
    def __init__(self, primary_recommender, secondary_recommender):
        self.primary_recommender = primary_recommender
        self.secondary_recommender = secondary_recommender

    def recommend_items(self, user_id, n=5, candidates_multiplier=3):
        """
        Recommend items for a user using a cascading approach

        Parameters:
        user_id (str): ID of the user
        n (int): Number of recommendations to return
        candidates_multiplier (int): Multiplier for the number of candidates from primary recommender

        Returns:
        DataFrame: Top N recommended items with scores
        """
        # Get candidate items from primary recommender
        candidates = self.primary_recommender.recommend_items(user_id, n=n*candidates_multiplier)

        if candidates.empty:
            return candidates

        # Refine candidates using secondary recommender
        refined_scores = []
        for _, row in candidates.iterrows():
            item_id = row['item_id']
            primary_score = row['score'] if 'score' in row else row['predicted_rating']

            # Get score from secondary recommender
            secondary_score = self.secondary_recommender.predict_rating(user_id, item_id)

            # Combine scores (in this case, multiply them)
            combined_score = primary_score * secondary_score

            refined_scores.append({
                'item_id': item_id,
                'score': combined_score,
                'primary_score': primary_score,
                'secondary_score': secondary_score
            })

        # Convert to DataFrame and sort by score (descending)
        recommendations = pd.DataFrame(refined_scores)
        recommendations = recommendations.sort_values('score', ascending=False).head(n)

        return recommendations
```

## Feature Combination Hybrid Recommender

A feature combination hybrid recommender uses features from multiple sources:

```python
class FeatureCombinationRecommender:
    def __init__(self, user_item_matrix, item_metadata):
        self.user_item_matrix = user_item_matrix
        self.item_metadata = item_metadata
        self.user_profiles = None
        self.item_features = None
        self._prepare_features()

    def _prepare_features(self):
        """Prepare combined features for users and items"""
        # Process item metadata to extract features
        # For text features, use TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')

        # Combine all text fields into a single document for each item
        text_fields = ['title', 'description', 'genre']  # Adjust based on your metadata
        text_features = []

        for item_id in self.item_metadata.index:
            doc = ""
            for field in text_fields:
                if field in self.item_metadata.columns:
                    value = self.item_metadata.loc[item_id, field]
                    if isinstance(value, str):
                        doc += value + " "
            text_features.append(doc.strip())

        # Create TF-IDF features
        if text_features:
            tfidf_matrix = tfidf.fit_transform(text_features)
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                index=self.item_metadata.index,
                columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
            )
        else:
            tfidf_df = pd.DataFrame(index=self.item_metadata.index)

        # Add numerical features
        numerical_fields = ['year', 'rating']  # Adjust based on your metadata
        numerical_features = pd.DataFrame(index=self.item_metadata.index)

        for field in numerical_fields:
            if field in self.item_metadata.columns:
                numerical_features[field] = self.item_metadata[field]

        # Combine TF-IDF and numerical features
        self.item_features = pd.concat([tfidf_df, numerical_features], axis=1)

        # Create user profiles based on their ratings and item features
        self.user_profiles = {}

        for user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0]

            if len(rated_items) == 0:
                # New user with no ratings
                self.user_profiles[user_id] = None
                continue

            # Calculate weighted average of item features based on ratings
            weighted_features = pd.DataFrame(0, index=[0], columns=self.item_features.columns)
            weights_sum = 0

            for item_id, rating in rated_items.items():
                if item_id in self.item_features.index:
                    item_feature_vector = self.item_features.loc[item_id]
                    weighted_features += rating * item_feature_vector
                    weights_sum += rating

            if weights_sum > 0:
                user_profile = weighted_features / weights_sum
                self.user_profiles[user_id] = user_profile.iloc[0]
            else:
                self.user_profiles[user_id] = None

    def recommend_items(self, user_id, n=5):
        """
        Recommend items for a user using combined features

        Parameters:
        user_id (str): ID of the user
        n (int): Number of recommendations to return

        Returns:
        DataFrame: Top N recommended items with scores
        """
        # Check if user has a profile
        if user_id not in self.user_profiles or self.user_profiles[user_id] is None:
            # Return empty recommendations for new users
            return pd.DataFrame(columns=['item_id', 'score'])

        user_profile = self.user_profiles[user_id]

        # Calculate similarity between user profile and all items
        similarities = []

        for item_id, item_features in self.item_features.iterrows():
            # Skip items the user has already rated
            if user_id in self.user_item_matrix.index and item_id in self.user_item_matrix.columns:
                if self.user_item_matrix.loc[user_id, item_id] > 0:
                    continue

            # Calculate cosine similarity
            similarity = cosine_similarity(
                user_profile.values.reshape(1, -1),
                item_features.values.reshape(1, -1)
            )[0][0]

            similarities.append({
                'item_id': item_id,
                'score': similarity
            })

        # Convert to DataFrame and sort by similarity (descending)
        recommendations = pd.DataFrame(similarities)
        if not recommendations.empty:
            recommendations = recommendations.sort_values('score', ascending=False).head(n)

        return recommendations
```

## Meta-Level Hybrid Recommender

A meta-level hybrid recommender uses the model learned by one technique as input to another:

```python
from sklearn.ensemble import RandomForestRegressor

class MetaLevelHybridRecommender:
    def __init__(self, base_recommenders, meta_model=None):
        self.base_recommenders = base_recommenders
        self.meta_model = meta_model if meta_model is not None else RandomForestRegressor(n_estimators=100)
        self.is_trained = False

    def train(self, training_data):
        """
        Train the meta-level recommender

        Parameters:
        training_data (list): List of (user_id, item_id, rating) tuples
        """
        # Prepare features and target for meta-model
        X = []
        y = []

        for user_id, item_id, rating in training_data:
            # Get predictions from all base recommenders
            base_predictions = []
            for recommender in self.base_recommenders:
                pred = recommender.predict_rating(user_id, item_id)
                base_predictions.append(pred)

            X.append(base_predictions)
            y.append(rating)

        # Train meta-model
        self.meta_model.fit(X, y)
        self.is_trained = True

        print(f"Meta-level recommender trained on {len(y)} examples")

    def predict_rating(self, user_id, item_id):
        """
        Predict rating for a user-item pair

        Parameters:
        user_id (str): ID of the user
        item_id (str): ID of the item

        Returns:
        float: Predicted rating
        """
        if not self.is_trained:
            raise ValueError("Meta-level recommender is not trained yet")

        # Get predictions from all base recommenders
        base_predictions = []
        for recommender in self.base_recommenders:
            pred = recommender.predict_rating(user_id, item_id)
            base_predictions.append(pred)

        # Use meta-model to make final prediction
        final_prediction = self.meta_model.predict([base_predictions])[0]

        return final_prediction

    def recommend_items(self, user_id, n=5, all_items=None):
        """
        Recommend items for a user

        Parameters:
        user_id (str): ID of the user
        n (int): Number of recommendations to return
        all_items (list): List of all item IDs to consider

        Returns:
        DataFrame: Top N recommended items with scores
        """
        if not self.is_trained:
            raise ValueError("Meta-level recommender is not trained yet")

        if all_items is None:
            # Get all items from the first base recommender
            # This assumes the first recommender has access to all items
            all_items = self.base_recommenders[0].get_all_items()

        # Predict ratings for all items
        predictions = []
        for item_id in all_items:
            try:
                rating = self.predict_rating(user_id, item_id)
                predictions.append({
                    'item_id': item_id,
                    'score': rating
                })
            except Exception as e:
                # Skip items that cause errors
                print(f"Error predicting for item {item_id}: {e}")

        # Convert to DataFrame and sort by score (descending)
        recommendations = pd.DataFrame(predictions)
        if not recommendations.empty:
            recommendations = recommendations.sort_values('score', ascending=False).head(n)

        return recommendations
```

## Implementing a Complete Hybrid Recommender System

Now let's implement a complete hybrid recommender system that combines content-based, collaborative filtering, and knowledge-based approaches:

```python
# Import recommenders from previous lessons
from content_based import ContentBasedRecommender
from collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorizationCF
from knowledge_based import KnowledgeBasedRecommender

class CompleteHybridRecommender:
    def __init__(self, user_item_matrix, item_metadata, user_preferences=None):
        self.user_item_matrix = user_item_matrix
        self.item_metadata = item_metadata
        self.user_preferences = user_preferences

        # Initialize individual recommenders
        self.content_recommender = ContentBasedRecommender(item_metadata)
        self.user_cf = UserBasedCF(user_item_matrix)
        self.item_cf = ItemBasedCF(user_item_matrix)
        self.mf_cf = MatrixFactorizationCF(user_item_matrix)
        self.knowledge_recommender = KnowledgeBasedRecommender(item_metadata, user_preferences)

        # Initialize hybrid recommenders
        self.weighted_hybrid = WeightedHybridRecommender(
            self.content_recommender,
            self.user_cf
        )

        self.switching_hybrid = SwitchingHybridRecommender(
            self.content_recommender,
            self.user_cf,
            self.knowledge_recommender
        )

        self.cascading_hybrid = CascadingHybridRecommender(
            self.user_cf,
            self.content_recommender
        )

        self.feature_hybrid = FeatureCombinationRecommender(
            user_item_matrix,
            item_metadata
        )

        # Meta-level hybrid requires training
        self.meta_hybrid = None

    def train_meta_hybrid(self, training_data):
        """Train the meta-level hybrid recommender"""
        base_recommenders = [
            self.content_recommender,
            self.user_cf,
            self.item_cf,
            self.mf_cf
        ]

        self.meta_hybrid = MetaLevelHybridRecommender(base_recommenders)
        self.meta_hybrid.train(training_data)

    def recommend_items(self, user_id, method='weighted', n=5):
        """
        Recommend items for a user using the specified hybrid method

        Parameters:
        user_id (str): ID of the user
        method (str): Hybrid method to use ('weighted', 'switching', 'cascading', 'feature', 'meta')
        n (int): Number of recommendations to return

        Returns:
        DataFrame: Top N recommended items with scores
        """
        if method == 'weighted':
            return self.weighted_hybrid.recommend_items(user_id, n=n)

        elif method == 'switching':
            return self.switching_hybrid.recommend_items(
                user_id,
                n=n,
                user_item_matrix=self.user_item_matrix,
                item_metadata=self.item_metadata
            )

        elif method == 'cascading':
            return self.cascading_hybrid.recommend_items(user_id, n=n)

        elif method == 'feature':
            return self.feature_hybrid.recommend_items(user_id, n=n)

        elif method == 'meta':
            if self.meta_hybrid is None or not self.meta_hybrid.is_trained:
                raise ValueError("Meta-level hybrid recommender is not trained yet")
            return self.meta_hybrid.recommend_items(user_id, n=n)

        else:
            raise ValueError(f"Unknown hybrid method: {method}")
```

## Evaluating Hybrid Recommenders

Let's evaluate our hybrid recommenders using the metrics we defined in previous lessons:

```python
def evaluate_hybrid_recommenders(hybrid_recommender, test_data, methods=None):
    """
    Evaluate different hybrid recommendation methods

    Parameters:
    hybrid_recommender: CompleteHybridRecommender instance
    test_data (list): List of (user_id, item_id, rating) tuples
    methods (list): List of hybrid methods to evaluate

    Returns:
    DataFrame: Evaluation metrics for each method
    """
    if methods is None:
        methods = ['weighted', 'switching', 'cascading', 'feature']
        if hybrid_recommender.meta_hybrid is not None and hybrid_recommender.meta_hybrid.is_trained:
            methods.append('meta')

    # Define evaluation metrics
    metrics = ['MAE', 'RMSE', 'Precision@5', 'Recall@5', 'F1@5', 'NDCG@5']

    # Initialize results dictionary
    results = {method: {metric: 0 for metric in metrics} for method in methods}

    # Evaluate each method
    for method in methods:
        print(f"\nEvaluating {method} hybrid recommender...")

        # Calculate prediction error metrics
        mae_sum = 0
        rmse_sum = 0
        count = 0

        for user_id, item_id, true_rating in test_data:
            try:
                if method == 'weighted':
                    # For weighted hybrid, we need to combine predictions from base recommenders
                    content_pred = hybrid_recommender.content_recommender.predict_rating(user_id, item_id)
                    collab_pred = hybrid_recommender.user_cf.predict_rating(user_id, item_id)

                    weights = hybrid_recommender.weighted_hybrid.weights
                    predicted_rating = weights['content'] * content_pred + weights['collaborative'] * collab_pred

                elif method == 'meta':
                    predicted_rating = hybrid_recommender.meta_hybrid.predict_rating(user_id, item_id)

                else:
                    # For other methods, we'll use the user-based CF prediction as an approximation
                    predicted_rating = hybrid_recommender.user_cf.predict_rating(user_id, item_id)

                error = true_rating - predicted_rating

                mae_sum += abs(error)
                rmse_sum += error ** 2
                count += 1

            except Exception as e:
                print(f"Error evaluating {method} for {user_id}, {item_id}: {e}")

        # Calculate average error metrics
        results[method]['MAE'] = mae_sum / count if count > 0 else float('inf')
        results[method]['RMSE'] = (rmse_sum / count) ** 0.5 if count > 0 else float('inf')

        # Calculate recommendation metrics
        # Group test data by user
        user_test_items = {}
        for user_id, item_id, rating in test_data:
            if user_id not in user_test_items:
                user_test_items[user_id] = []
            user_test_items[user_id].append((item_id, rating))

        precision_sum = 0
        recall_sum = 0
        ndcg_sum = 0
        user_count = 0

        for user_id, items in user_test_items.items():
            # Get relevant items (items with high ratings)
            relevant_items = [item_id for item_id, rating in items if rating >= 4]

            if not relevant_items:
                continue

            # Get recommendations for the user
            try:
                recommendations = hybrid_recommender.recommend_items(user_id, method=method, n=5)
                recommended_items = recommendations['item_id'].tolist() if not recommendations.empty else []

                # Calculate precision and recall
                relevant_and_recommended = set(relevant_items).intersection(set(recommended_items))

                precision = len(relevant_and_recommended) / len(recommended_items) if recommended_items else 0
                recall = len(relevant_and_recommended) / len(relevant_items) if relevant_items else 0

                # Calculate NDCG
                dcg = 0
                idcg = 0

                for i, item_id in enumerate(recommended_items):
                    if item_id in relevant_items:
                        dcg += 1 / np.log2(i + 2)

                for i in range(min(len(relevant_items), 5)):
                    idcg += 1 / np.log2(i + 2)

                ndcg = dcg / idcg if idcg > 0 else 0

                precision_sum += precision
                recall_sum += recall
                ndcg_sum += ndcg
                user_count += 1

            except Exception as e:
                print(f"Error getting recommendations for {user_id} with {method}: {e}")

        # Calculate average recommendation metrics
        results[method]['Precision@5'] = precision_sum / user_count if user_count > 0 else 0
        results[method]['Recall@5'] = recall_sum / user_count if user_count > 0 else 0

        # Calculate F1 score
        precision = results[method]['Precision@5']
        recall = results[method]['Recall@5']
        results[method]['F1@5'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results[method]['NDCG@5'] = ndcg_sum / user_count if user_count > 0 else 0

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

# Example usage
# Create sample data
user_item_matrix = pd.DataFrame({
    '1': [5, 4, 0, 5, 0],
    '2': [3, 0, 4, 2, 0],
    '3': [4, 5, 3, 0, 4],
    '4': [0, 0, 5, 0, 5],
    '5': [0, 3, 0, 4, 3]
}, index=['user1', 'user2', 'user3', 'user4', 'user5'])

item_metadata = pd.DataFrame({
    'title': ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5'],
    'genre': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy'],
    'year': [2020, 2019, 2021, 2018, 2022],
    'rating': [4.5, 3.8, 4.2, 3.9, 4.0]
}, index=['1', '2', '3', '4', '5'])

user_preferences = {
    'user1': {'genre': ['Action', 'Drama']},
    'user2': {'genre': ['Action']},
    'user3': {'genre': ['Comedy', 'Drama']},
    'user4': {'genre': ['Drama']},
    'user5': {'genre': ['Comedy']}
}

# Create test data
test_data = [
    ('user1', '4', 4),
    ('user2', '1', 5),
    ('user3', '2', 4),
    ('user4', '3', 5),
    ('user5', '4', 3)
]

# Create training data for meta-level hybrid
training_data = [
    ('user1', '1', 5),
    ('user1', '2', 3),
    ('user1', '3', 4),
    ('user2', '1', 4),
    ('user2', '3', 5),
    ('user3', '2', 4),
    ('user3', '3', 3),
    ('user4', '1', 5),
    ('user4', '5', 4),
    ('user5', '3', 4),
    ('user5', '5', 3)
]

# Create hybrid recommender
hybrid_recommender = CompleteHybridRecommender(user_item_matrix, item_metadata, user_preferences)

# Train meta-level hybrid
hybrid_recommender.train_meta_hybrid(training_data)

# Evaluate hybrid recommenders
results = evaluate_hybrid_recommenders(hybrid_recommender, test_data)

print("\nEvaluation Results:")
print(results)

# Visualize results
import matplotlib.pyplot as plt

# Plot error metrics (lower is better)
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
results.loc[['MAE', 'RMSE']].plot(kind='bar')
plt.title('Error Metrics (lower is better)')
plt.ylabel('Error')
plt.grid(axis='y')

# Plot recommendation metrics (higher is better)
plt.subplot(2, 1, 2)
results.loc[['Precision@5', 'Recall@5', 'F1@5', 'NDCG@5']].plot(kind='bar')
plt.title('Recommendation Metrics (higher is better)')
plt.ylabel('Score')
plt.grid(axis='y')

plt.tight_layout()
plt.show()
```

## Optimizing Hybrid Recommenders

Let's implement a function to optimize the weights for our weighted hybrid recommender:

```python
def optimize_hybrid_weights(hybrid_recommender, validation_data):
    """
    Find optimal weights for the weighted hybrid recommender

    Parameters:
    hybrid_recommender: CompleteHybridRecommender instance
    validation_data (list): List of (user_id, item_id, rating) tuples

    Returns:
    dict: Dictionary with optimal weights
    """
    # Define weight combinations to try
    weight_combinations = [
        {'content': 0.0, 'collaborative': 1.0},
        {'content': 0.1, 'collaborative': 0.9},
        {'content': 0.2, 'collaborative': 0.8},
        {'content': 0.3, 'collaborative': 0.7},
        {'content': 0.4, 'collaborative': 0.6},
        {'content': 0.5, 'collaborative': 0.5},
        {'content': 0.6, 'collaborative': 0.4},
        {'content': 0.7, 'collaborative': 0.3},
        {'content': 0.8, 'collaborative': 0.2},
        {'content': 0.9, 'collaborative': 0.1},
        {'content': 1.0, 'collaborative': 0.0}
    ]

    best_weights = {'content': 0.5, 'collaborative': 0.5}
    best_rmse = float('inf')

    print("\nOptimizing hybrid weights...")

    for weights in weight_combinations:
        # Update weights in the weighted hybrid recommender
        hybrid_recommender.weighted_hybrid.weights = weights

        # Calculate RMSE on validation data
        mae_sum = 0
        rmse_sum = 0
        count = 0

        for user_id, item_id, true_rating in validation_data:
            try:
                # Get predictions from base recommenders
                content_pred = hybrid_recommender.content_recommender.predict_rating(user_id, item_id)
                collab_pred = hybrid_recommender.user_cf.predict_rating(user_id, item_id)

                # Calculate weighted prediction
                predicted_rating = weights['content'] * content_pred + weights['collaborative'] * collab_pred

                error = true_rating - predicted_rating

                mae_sum += abs(error)
                rmse_sum += error ** 2
                count += 1

            except Exception as e:
                print(f"Error evaluating weights {weights} for {user_id}, {item_id}: {e}")

        # Calculate average RMSE
        rmse = (rmse_sum / count) ** 0.5 if count > 0 else float('inf')
        mae = mae_sum / count if count > 0 else float('inf')

        print(f"Weights: {weights}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = weights

    print(f"\nBest weights: {best_weights}, RMSE: {best_rmse:.4f}")

    return best_weights

# Example usage
# Create validation data
validation_data = [
    ('user1', '5', 3),
    ('user2', '4', 4),
    ('user3', '1', 5),
    ('user4', '2', 4),
    ('user5', '3', 3)
]

# Optimize weights
optimal_weights = optimize_hybrid_weights(hybrid_recommender, validation_data)

# Update weights in the hybrid recommender
hybrid_recommender.weighted_hybrid.weights = optimal_weights

# Evaluate optimized weighted hybrid
print("\nEvaluating optimized weighted hybrid recommender:")
optimized_metrics = evaluate_hybrid_recommenders(
    hybrid_recommender,
    test_data,
    methods=['weighted']
)
print(optimized_metrics)
```

## Exercise

1. Implement a feature augmentation hybrid recommender that uses the output of one recommender as input features to another
2. Create a switching hybrid recommender that uses different criteria for switching between recommenders
3. Implement a stacking ensemble method for combining multiple recommenders
4. Design a hybrid recommender that adapts its weights based on user feedback
5. Create a visualization that shows how different hybrid methods perform for different types of users

## Next Steps

In the next lesson, we'll explore deep learning approaches for recommendation systems, including neural collaborative filtering and sequence-based recommenders.

## Resources

- [Hybrid Recommender Systems: Survey and Experiments](https://dl.acm.org/doi/10.1145/505248.505253)
- [A Survey of Hybrid Recommendation Systems](https://www.sciencedirect.com/science/article/pii/S1319157818300034)
- [Netflix Recommender System: Algorithms, Business Value, and Innovation](https://dl.acm.org/doi/10.1145/2843948)
- [Hybrid Recommendation Systems in Python](https://towardsdatascience.com/recommendation-systems-models-and-evaluation-84944a84fb8e)
- [LightFM: A Hybrid Recommendation Framework](https://github.com/lyst/lightfm)
