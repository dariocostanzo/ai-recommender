# Lesson 8: Deploying Recommendation Systems in Production

## Introduction

In previous lessons, we've explored various recommendation algorithms from collaborative filtering to deep learning approaches. However, building a great recommendation model is only part of the challenge. In this lesson, we'll focus on how to effectively deploy recommendation systems in production environments.

## Challenges in Deploying Recommendation Systems

Deploying recommendation systems comes with unique challenges:

1. **Scale**: Production systems often need to handle millions of users and items
2. **Latency**: Recommendations need to be generated quickly, often in real-time
3. **Freshness**: Models need to incorporate new user interactions and items
4. **Cold start**: Handling new users and items with limited data
5. **Infrastructure**: Managing compute resources efficiently
6. **Monitoring**: Tracking recommendation quality and system health

## Architecture of Production Recommendation Systems

A typical production recommendation system architecture includes:

```

┌───────────────┐     ┌───────────────┐    ┌───────────────┐
│ Data Sources  │───▶│ Data Pipeline │───▶│ Feature Store │
└───────────────┘     └───────────────┘    └───────────────┘
│
▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Monitoring    │◀───│ Serving Layer │◀───│ Model Training│
└───────────────┘    └───────────────┘    └───────────────┘
▲ │
│ ▼
│            ┌───────────────┐
└────────────│ Clients       │
             └───────────────┘

```

### Components

1. **Data Sources**: User interactions, item metadata, contextual information
2. **Data Pipeline**: ETL processes, data cleaning, feature engineering
3. **Feature Store**: Centralized repository for features used in training and serving
4. **Model Training**: Offline training of recommendation models
5. **Serving Layer**: API for generating recommendations in real-time
6. **Monitoring**: Tracking system performance and recommendation quality
7. **Clients**: Web, mobile, or other interfaces consuming recommendations

## Offline vs. Online Recommendation Generation

### Offline Generation (Batch Processing)

```python
def generate_offline_recommendations(recommender, users, n=10):
    """
    Generate recommendations for all users in batch mode

    Parameters:
    recommender: Trained recommender model
    users: List of user IDs
    n: Number of recommendations per user

    Returns:
    Dictionary mapping user IDs to their recommendations
    """
    recommendations = {}

    for user_id in users:
        try:
            user_recs = recommender.recommend_items(user_id, n=n)
            if not user_recs.empty:
                recommendations[user_id] = user_recs.to_dict('records')
            else:
                recommendations[user_id] = []
        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {e}")
            recommendations[user_id] = []

    return recommendations

# Example usage
all_users = user_item_matrix.index.tolist()
batch_recommendations = generate_offline_recommendations(hybrid_recommender, all_users)

# Save to database or file
import json
with open('recommendations.json', 'w') as f:
    json.dump(batch_recommendations, f)
```

### Online Generation (Real-time)

```python
def get_real_time_recommendations(recommender, user_id, context=None, n=5):
    """
    Generate real-time recommendations for a user

    Parameters:
    recommender: Trained recommender model
    user_id: ID of the user
    context: Optional contextual information
    n: Number of recommendations

    Returns:
    List of recommended items
    """
    try:
        # Get base recommendations
        recommendations = recommender.recommend_items(user_id, n=n)

        # Apply contextual filtering if context is provided
        if context and not recommendations.empty:
            # Example: Filter by category if specified in context
            if 'category' in context:
                category_filter = context['category']
                # This assumes we have item_categories data
                matching_items = [item for item in recommendations['item_id']
                                 if item in item_categories and
                                 category_filter in item_categories[item]]
                recommendations = recommendations[recommendations['item_id'].isin(matching_items)]

            # Example: Time-based filtering
            if 'time_of_day' in context:
                time_of_day = context['time_of_day']
                # Apply time-based business logic
                # ...

        if recommendations.empty:
            return []

        return recommendations.to_dict('records')

    except Exception as e:
        print(f"Error generating real-time recommendations: {e}")
        return []

# Example usage
user_id = 'user1'
context = {'category': 'Action', 'time_of_day': 'evening'}
real_time_recs = get_real_time_recommendations(hybrid_recommender, user_id, context)
```

## Scaling Recommendation Systems

### Preprocessing and Caching

```python
import redis
import pickle
import hashlib

class RecommendationCache:
    def __init__(self, host='localhost', port=6379, db=0, expiration=3600):
        """
        Initialize Redis cache for recommendations

        Parameters:
        host: Redis host
        port: Redis port
        db: Redis database number
        expiration: Cache expiration time in seconds (default: 1 hour)
        """
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.expiration = expiration

    def _generate_key(self, user_id, context=None, n=10):
        """Generate a unique cache key based on parameters"""
        key_parts = [str(user_id), str(n)]

        if context:
            # Sort context keys to ensure consistent key generation
            sorted_context = sorted(context.items())
            context_str = ','.join([f"{k}:{v}" for k, v in sorted_context])
            key_parts.append(context_str)

        key_string = '_'.join(key_parts)
        return f"rec:{hashlib.md5(key_string.encode()).hexdigest()}"

    def get_recommendations(self, user_id, context=None, n=10):
        """Get recommendations from cache if available"""
        cache_key = self._generate_key(user_id, context, n)
        cached_data = self.redis_client.get(cache_key)

        if cached_data:
            return pickle.loads(cached_data)
        return None

    def store_recommendations(self, user_id, recommendations, context=None, n=10):
        """Store recommendations in cache"""
        cache_key = self._generate_key(user_id, context, n)
        self.redis_client.setex(
            cache_key,
            self.expiration,
            pickle.dumps(recommendations)
        )

# Example usage
cache = RecommendationCache()

def get_cached_recommendations(recommender, user_id, context=None, n=5):
    """Get recommendations with caching"""
    # Try to get from cache first
    cached_recs = cache.get_recommendations(user_id, context, n)

    if cached_recs:
        print("Cache hit!")
        return cached_recs

    # Generate new recommendations
    print("Cache miss, generating recommendations...")
    recommendations = get_real_time_recommendations(recommender, user_id, context, n)

    # Store in cache for future requests
    cache.store_recommendations(user_id, recommendations, context, n)

    return recommendations
```

### Distributed Processing

For large-scale systems, distributed processing frameworks like Apache Spark can be used:

```python
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

def train_spark_als(spark, ratings_df, rank=10, max_iter=10, reg_param=0.01):
    """
    Train ALS model using Spark

    Parameters:
    spark: SparkSession
    ratings_df: DataFrame with columns (user_id, item_id, rating)
    rank: Number of latent factors
    max_iter: Maximum iterations
    reg_param: Regularization parameter

    Returns:
    Trained ALS model
    """
    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(ratings_df)

    # Initialize ALS model
    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        coldStartStrategy="drop"
    )

    # Train model
    model = als.fit(spark_df)

    return model

def generate_recommendations_spark(spark, model, users, n=10):
    """
    Generate recommendations for users using Spark

    Parameters:
    spark: SparkSession
    model: Trained ALS model
    users: List of user IDs
    n: Number of recommendations per user

    Returns:
    DataFrame with user recommendations
    """
    # Create DataFrame of users
    users_df = spark.createDataFrame([(user,) for user in users], ["user_id"])

    # Generate recommendations
    recommendations = model.recommendForUserSubset(users_df, n)

    return recommendations

# Example usage
spark = SparkSession.builder \
    .appName("RecommendationSystem") \
    .getOrCreate()

# Prepare ratings data
ratings_data = []
for user in user_item_matrix.index:
    for item in user_item_matrix.columns:
        rating = user_item_matrix.loc[user, item]
        if rating > 0:
            ratings_data.append((user, item, float(rating)))

ratings_df = pd.DataFrame(ratings_data, columns=["user_id", "item_id", "rating"])

# Train model
model = train_spark_als(spark, ratings_df)

# Generate recommendations
all_users = user_item_matrix.index.tolist()
recommendations = generate_recommendations_spark(spark, model, all_users)
```

## Building a Recommendation API

Let's create a simple Flask API for serving recommendations:

```python
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load pre-trained models
def load_models():
    models = {}
    model_dir = "models"

    # Load NCF model
    with open(os.path.join(model_dir, "ncf_model.pkl"), "rb") as f:
        models["ncf"] = pickle.load(f)

    # Load content-based model
    with open(os.path.join(model_dir, "content_model.pkl"), "rb") as f:
        models["content"] = pickle.load(f)

    # Load hybrid model
    with open(os.path.join(model_dir, "hybrid_model.pkl"), "rb") as f:
        models["hybrid"] = pickle.load(f)

    return models

# Initialize cache
recommendation_cache = RecommendationCache()

# Load models
models = load_models()

@app.route("/api/recommendations", methods=["GET"])
def get_recommendations():
    user_id = request.args.get("user_id")
    model_type = request.args.get("model", "hybrid")  # Default to hybrid
    n = int(request.args.get("n", 5))  # Default to 5 recommendations

    # Parse context if provided
    context = {}
    for key in request.args:
        if key.startswith("context_"):
            context_key = key[8:]  # Remove "context_" prefix
            context[context_key] = request.args.get(key)

    # Check if we have recommendations in cache
    cached_recs = recommendation_cache.get_recommendations(user_id, context, n)
    if cached_recs:
        return jsonify({"recommendations": cached_recs, "source": "cache"})

    # Get appropriate model
    if model_type not in models:
        return jsonify({"error": f"Model type '{model_type}' not found"}), 400

    recommender = models[model_type]

    # Generate recommendations
    try:
        recommendations = get_real_time_recommendations(recommender, user_id, context, n)

        # Store in cache
        recommendation_cache.store_recommendations(user_id, recommendations, context, n)

        return jsonify({"recommendations": recommendations, "source": "model"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({"available_models": list(models.keys())})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
```

## A/B Testing Recommendation Algorithms

A/B testing is crucial for evaluating recommendation algorithms in production:

```python
import random
import uuid
from datetime import datetime

class ABTest:
    def __init__(self, test_name, variants, traffic_split=None):
        """
        Initialize A/B test

        Parameters:
        test_name: Name of the A/B test
        variants: List of variant names
        traffic_split: Dictionary mapping variant names to traffic percentages
                      (if None, traffic is split evenly)
        """
        self.test_name = test_name
        self.variants = variants

        if traffic_split is None:
            # Split traffic evenly
            split_percentage = 1.0 / len(variants)
            self.traffic_split = {variant: split_percentage for variant in variants}
        else:
            # Validate traffic split
            total = sum(traffic_split.values())
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"Traffic split must sum to 1.0, got {total}")
            self.traffic_split = traffic_split

        # Track assignments and conversions
        self.assignments = {variant: 0 for variant in variants}
        self.conversions = {variant: 0 for variant in variants}

    def assign_variant(self, user_id):
        """
        Assign a user to a variant

        Parameters:
        user_id: User ID

        Returns:
        Assigned variant name
        """
        # Deterministic assignment based on user_id
        # This ensures the same user always gets the same variant
        random.seed(user_id)
        rand_val = random.random()

        cumulative = 0
        for variant, percentage in self.traffic_split.items():
            cumulative += percentage
            if rand_val <= cumulative:
                self.assignments[variant] += 1
                return variant

        # Fallback to last variant
        last_variant = self.variants[-1]
        self.assignments[last_variant] += 1
        return last_variant

    def record_conversion(self, variant):
        """Record a conversion for a variant"""
        if variant in self.conversions:
            self.conversions[variant] += 1

    def get_statistics(self):
        """Get test statistics"""
        stats = {}

        for variant in self.variants:
            assignments = self.assignments[variant]
            conversions = self.conversions[variant]

            stats[variant] = {
                "assignments": assignments,
                "conversions": conversions,
                "conversion_rate": conversions / assignments if assignments > 0 else 0
            }

        return stats

# Example usage in the API
ab_test = ABTest(
    "recommendation_algorithms",
    ["ncf", "content", "hybrid"],
    {"ncf": 0.3, "content": 0.3, "hybrid": 0.4}
)

# Modified API endpoint with A/B testing
@app.route("/api/recommendations_ab", methods=["GET"])
def get_recommendations_ab():
    user_id = request.args.get("user_id")
    n = int(request.args.get("n", 5))

    # Parse context if provided
    context = {}
    for key in request.args:
        if key.startswith("context_"):
            context_key = key[8:]
            context[context_key] = request.args.get(key)

    # Assign user to a variant
    model_type = ab_test.assign_variant(user_id)

    # Check if we have recommendations in cache
    cached_recs = recommendation_cache.get_recommendations(user_id, context, n)
    if cached_recs:
        return jsonify({
            "recommendations": cached_recs,
            "source": "cache",
            "model": model_type
        })

    # Get appropriate model
    if model_type not in models:
        return jsonify({"error": f"Model type '{model_type}' not found"}), 400

    recommender = models[model_type]

    # Generate recommendations
    try:
        recommendations = get_real_time_recommendations(recommender, user_id, context, n)

        # Store in cache
        recommendation_cache.store_recommendations(user_id, recommendations, context, n)

        return jsonify({
            "recommendations": recommendations,
            "source": "model",
            "model": model_type
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to record conversions
@app.route("/api/record_conversion", methods=["POST"])
def record_conversion():
    data = request.json
    user_id = data.get("user_id")
    item_id = data.get("item_id")

    # Get the variant assigned to this user
    variant = ab_test.assign_variant(user_id)

    # Record conversion
    ab_test.record_conversion(variant)

    return jsonify({"success": True})

# Endpoint to get A/B test statistics
@app.route("/api/ab_test_stats", methods=["GET"])
def get_ab_test_stats():
    stats = ab_test.get_statistics()
    return jsonify(stats)
```

## Model Retraining and Updating

Recommendation models need to be regularly updated to incorporate new data:

```python
import schedule
import time
import threading
import pandas as pd
from datetime import datetime

class ModelManager:
    def __init__(self, models_dir="models", update_interval_hours=24):
        """
        Initialize model manager

        Parameters:
        models_dir: Directory to store models
        update_interval_hours: How often to update models (in hours)
        """
        self.models_dir = models_dir
        self.update_interval_hours = update_interval_hours
        self.models = {}
        self.last_updated = {}
        self.lock = threading.RLock()

        # Load initial models
        self.load_models()

        # Schedule regular updates
        self.schedule_updates()

    def load_models(self):
        """Load all models from disk"""
        with self.lock:
            try:
                # Load NCF model
                with open(os.path.join(self.models_dir, "ncf_model.pkl"), "rb") as f:
                    self.models["ncf"] = pickle.load(f)
                    self.last_updated["ncf"] = datetime.now()

                # Load content-based model
                with open(os.path.join(self.models_dir, "content_model.pkl"), "rb") as f:
                    self.models["content"] = pickle.load(f)
                    self.last_updated["content"] = datetime.now()

                # Load hybrid model
                with open(os.path.join(self.models_dir, "hybrid_model.pkl"), "rb") as f:
                    self.models["hybrid"] = pickle.load(f)
                    self.last_updated["hybrid"] = datetime.now()

                print("Models loaded successfully")
            except Exception as e:
                print(f"Error loading models: {e}")

    def get_model(self, model_type):
        """Get a model by type"""
        with self.lock:
            if model_type in self.models:
                return self.models[model_type]
            return None

    def update_model(self, model_type):
        """Update a specific model"""
        print(f"Updating {model_type} model...")

        try:
            # Load latest data
            if model_type == "ncf":
                # Get latest user-item interactions
                user_item_matrix = self._load_latest_interactions()

                # Train new model
                new_model = NCFRecommender(user_item_matrix, num_epochs=5)
                new_model.train()

            elif model_type == "content":
                # Get latest user-item interactions and item features
                user_item_matrix = self._load_latest_interactions()
                item_features = self._load_latest_item_features()

                # Train new model
                new_model = DeepContentRecommender(item_features, user_item_matrix, num_epochs=5)
                new_model.train()

            elif model_type == "hybrid":
                # Get latest data
                user_item_matrix = self._load_latest_interactions()
                item_features = self._load_latest_item_features()
                user_sequences = self._load_latest_user_sequences()

                # Train new model
                new_model = DeepHybridRecommender(user_item_matrix, item_features, user_sequences)
                new_model.train()

            # Save and update model
            with self.lock:
                # Save to disk
                with open(os.path.join(self.models_dir, f"{model_type}_model.pkl"), "wb") as f:
                    pickle.dump(new_model, f)

                # Update in-memory model
                self.models[model_type] = new_model
                self.last_updated[model_type] = datetime.now()

            print(f"{model_type} model updated successfully")

        except Exception as e:
            print(f"Error updating {model_type} model: {e}")

    def _load_latest_interactions(self):
        """Load latest user-item interactions from database"""
        # In a real system, this would query a database
        # For this example, we'll simulate it
        print("Loading latest user-item interactions...")
        return pd.DataFrame({
            '1': [5, 4, 0, 5, 0],
            '2': [3, 0, 4, 2, 0],
            '3': [4, 5, 3, 0, 4],
            '4': [0, 0, 5, 0, 5],
            '5': [0, 3, 0, 4, 3]
        }, index=['user1', 'user2', 'user3', 'user4', 'user5'])

    def _load_latest_item_features(self):
        """Load latest item features from database"""
        # In a real system, this would query a database
        print("Loading latest item features...")

        # Create item features (one-hot encoded genres and normalized year)
        genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance']
        item_features = pd.DataFrame(0, index=['1', '2', '3', '4', '5'], columns=genres + ['year_norm'])

        # Set genres for each item
        item_features.loc['1', 'Action'] = 1
        item_features.loc['2', 'Comedy'] = 1
        item_features.loc['3', 'Drama'] = 1
        item_features.loc['4', 'Action'] = 1
        item_features.loc['4', 'Sci-Fi'] = 1
        item_features.loc['5', 'Comedy'] = 1
        item_features.loc['5', 'Romance'] = 1

        # Set normalized years (between 0 and 1)
        item_features['year_norm'] = [0.2, 0.4, 0.6, 0.8, 1.0]

        return item_features

    def _load_latest_user_sequences(self):
        """Load latest user interaction sequences from database"""
        # In a real system, this would query a database
        print("Loading latest user sequences...")
        return {
            'user1': [1, 3, 4],
            'user2': [2, 3, 1],
            'user3': [1, 2, 5],
            'user4': [3, 5],
            'user5': [2, 4, 5]
        }

    def schedule_updates(self):
        """Schedule regular model updates"""
        def update_all_models():
            for model_type in self.models.keys():
                self.update_model(model_type)

        # Schedule daily updates
        schedule.every(self.update_interval_hours).hours.do(update_all_models)

        # Run the scheduler in a background thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()

# Example usage
model_manager = ModelManager()

# Modified API to use model manager
@app.route("/api/recommendations", methods=["GET"])
def get_recommendations_v2():
    user_id = request.args.get("user_id")
    model_type = request.args.get("model", "hybrid")
    n = int(request.args.get("n", 5))

    # Parse context if provided
    context = {}
    for key in request.args:
        if key.startswith("context_"):
            context_key = key[8:]
            context[context_key] = request.args.get(key)

    # Check if we have recommendations in cache
    cached_recs = recommendation_cache.get_recommendations(user_id, context, n)
    if cached_recs:
        return jsonify({"recommendations": cached_recs, "source": "cache"})

    # Get appropriate model
    recommender = model_manager.get_model(model_type)
    if recommender is None:
        return jsonify({"error": f"Model type '{model_type}' not found"}), 400

    # Generate recommendations
    try:
        recommendations = get_real_time_recommendations(recommender, user_id, context, n)

        # Store in cache
        recommendation_cache.store_recommendations(user_id, recommendations, context, n)

        return jsonify({
            "recommendations": recommendations,
            "source": "model",
            "model_last_updated": model_manager.last_updated.get(model_type).isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

## Monitoring Recommendation Systems

Monitoring is crucial for maintaining recommendation quality:

```python
import time
from collections import deque
import threading
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class RecommendationMonitor:
    def __init__(self, window_size=1000):
        """
        Initialize recommendation system monitor

        Parameters:
        window_size: Number of requests to keep in history
        """
        self.window_size = window_size

        # Metrics
        self.response_times = deque(maxlen=window_size)
        self.cache_hits = deque(maxlen=window_size)
        self.error_rates = deque(maxlen=window_size)
        self.recommendation_counts = deque(maxlen=window_size)

        # User engagement metrics
        self.click_through_rates = deque(maxlen=window_size)
        self.conversion_rates = deque(maxlen=window_size)

        # Lock for thread safety
        self.lock = threading.RLock()

    def record_request(self, response_time, cache_hit, error, num_recommendations):
        """Record metrics for a recommendation request"""
        with self.lock:
            self.response_times.append(response_time)
            self.cache_hits.append(1 if cache_hit else 0)
            self.error_rates.append(1 if error else 0)
            self.recommendation_counts.append(num_recommendations)

    def record_engagement(self, click_through, conversion):
        """Record user engagement metrics"""
        with self.lock:
            self.click_through_rates.append(1 if click_through else 0)
            self.conversion_rates.append(1 if conversion else 0)

    def get_metrics(self):
        """Get current metrics"""
        with self.lock:
            metrics = {
                "avg_response_time": np.mean(self.response_times) if self.response_times else 0,
                "p95_response_time": np.percentile(self.response_times, 95) if self.response_times else 0,
                "cache_hit_rate": np.mean(self.cache_hits) if self.cache_hits else 0,
                "error_rate": np.mean(self.error_rates) if self.error_rates else 0,
                "avg_recommendations": np.mean(self.recommendation_counts) if self.recommendation_counts else 0,
                "click_through_rate": np.mean(self.click_through_rates) if self.click_through_rates else 0,
                "conversion_rate": np.mean(self.conversion_rates) if self.conversion_rates else 0
            }

            return metrics

    def generate_dashboard(self):
        """Generate a simple dashboard with metrics visualizations"""
        metrics = self.get_metrics()

        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Response time histogram
        if self.response_times:
            axs[0, 0].hist(self.response_times, bins=20)
            axs[0, 0].set_title('Response Time Distribution')
            axs[0, 0].set_xlabel('Response Time (ms)')
            axs[0, 0].set_ylabel('Frequency')

        # Cache hit rate over time
        if self.cache_hits:
            window = min(100, len(self.cache_hits))
            rolling_avg = [np.mean(list(self.cache_hits)[max(0, i-window):i+1])
                          for i in range(len(self.cache_hits))]
            axs[0, 1].plot(rolling_avg)
            axs[0, 1].set_title('Cache Hit Rate (Rolling Average)')
            axs[0, 1].set_xlabel('Request Number')
            axs[0, 1].set_ylabel('Hit Rate')

        # User engagement metrics
        labels = ['Click-through', 'Conversion']
        values = [metrics['click_through_rate'], metrics['conversion_rate']]
        axs[1, 0].bar(labels, values)
        axs[1, 0].set_title('User Engagement Metrics')
        axs[1, 0].set_ylim(0, 1)

        # Error rate over time
        if self.error_rates:
            window = min(100, len(self.error_rates))
            rolling_avg = [np.mean(list(self.error_rates)[max(0, i-window):i+1])
                          for i in range(len(self.error_rates))]
            axs[1, 1].plot(rolling_avg)
            axs[1, 1].set_title('Error Rate (Rolling Average)')
            axs[1, 1].set_xlabel('Request Number')
            axs[1, 1].set_ylabel('Error Rate')

        plt.tight_layout()

        # Convert plot to base64 image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        return f"<img src='data:image/png;base64,{img_str}'/>"

# Initialize monitor
recommendation_monitor = RecommendationMonitor()

# Middleware to track request metrics
def track_recommendation_request(func):
    """Decorator to track recommendation request metrics"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        error = False
        cache_hit = False
        num_recommendations = 0

        try:
            response = func(*args, **kwargs)

            # Extract metrics from response
            if isinstance(response, tuple):
                result = response[0]
            else:
                result = response

            if hasattr(result, 'json'):
                data = result.json()
                cache_hit = data.get('source') == 'cache'
                recommendations = data.get('recommendations', [])
                num_recommendations = len(recommendations)

            return response

        except Exception as e:
            error = True
            raise e

        finally:
            # Record metrics
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            recommendation_monitor.record_request(
                response_time, cache_hit, error, num_recommendations
            )

    return wrapper

# Apply middleware to recommendation endpoints
app.view_functions['get_recommendations'] = track_recommendation_request(app.view_functions['get_recommendations'])
app.view_functions['get_recommendations_ab'] = track_recommendation_request(app.view_functions['get_recommendations_ab'])

# Add monitoring dashboard endpoint
@app.route("/api/monitoring/dashboard", methods=["GET"])
def monitoring_dashboard():
    """Generate and return monitoring dashboard"""
    dashboard_html = recommendation_monitor.generate_dashboard()

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recommendation System Monitoring</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
            .metric {{
                background-color: #f5f5f5;
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                min-width: 200px;
            }}
            .metric h3 {{ margin-top: 0; }}
            .value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
        </style>
    </head>
    <body>
        <h1>Recommendation System Monitoring</h1>

        <div class="metrics">
            <div class="metric">
                <h3>Avg Response Time</h3>
                <div class="value">{recommendation_monitor.get_metrics()['avg_response_time']:.2f} ms</div>
            </div>
            <div class="metric">
                <h3>P95 Response Time</h3>
                <div class="value">{recommendation_monitor.get_metrics()['p95_response_time']:.2f} ms</div>
            </div>
            <div class="metric">
                <h3>Cache Hit Rate</h3>
                <div class="value">{recommendation_monitor.get_metrics()['cache_hit_rate']*100:.1f}%</div>
            </div>
            <div class="metric">
                <h3>Error Rate</h3>
                <div class="value">{recommendation_monitor.get_metrics()['error_rate']*100:.1f}%</div>
            </div>
            <div class="metric">
                <h3>Click-through Rate</h3>
                <div class="value">{recommendation_monitor.get_metrics()['click_through_rate']*100:.1f}%</div>
            </div>
            <div class="metric">
                <h3>Conversion Rate</h3>
                <div class="value">{recommendation_monitor.get_metrics()['conversion_rate']*100:.1f}%</div>
            </div>
        </div>

        <h2>Visualizations</h2>
        {dashboard_html}

        <script>
            // Auto-refresh every 30 seconds
            setTimeout(function() {{
                location.reload();
            }}, 30000);
        </script>
    </body>
    </html>
    """

# Endpoint to record user engagement
@app.route("/api/record_engagement", methods=["POST"])
def record_engagement():
    """Record user engagement with recommendations"""
    data = request.json
    user_id = data.get("user_id")
    item_id = data.get("item_id")
    click_through = data.get("click_through", False)
    conversion = data.get("conversion", False)

    # Record engagement metrics
    recommendation_monitor.record_engagement(click_through, conversion)

    return jsonify({"success": True})
```

## Handling Cold Start Problems in Production

Cold start is a common challenge in recommendation systems. Here are strategies to address it:

```python
class ColdStartHandler:
    def __init__(self, item_features=None, popularity_model=None):
        """
        Initialize cold start handler

        Parameters:
        item_features: DataFrame of item features
        popularity_model: Model for popularity-based recommendations
        """
        self.item_features = item_features
        self.popularity_model = popularity_model

        # Cache for similar items
        self.similar_items_cache = {}

    def recommend_for_new_user(self, n=5):
        """Recommend items for a new user"""
        if self.popularity_model is not None:
            # Use popularity model
            return self.popularity_model.recommend_items(n=n)
        else:
            # Fallback to random popular items
            return self._get_random_popular_items(n)

    def recommend_for_new_item(self, item_id, item_features, n=5):
        """Recommend similar items for a new item"""
        if item_id in self.similar_items_cache:
            return self.similar_items_cache[item_id]

        if self.item_features is not None:
            # Find similar items based on features
            similar_items = self._find_similar_items(item_features, n)
            self.similar_items_cache[item_id] = similar_items
            return similar_items
        else:
            # Fallback to random popular items
            return self._get_random_popular_items(n)

    def _find_similar_items(self, item_features, n=5):
        """Find items with similar features"""
        from sklearn.metrics.pairwise import cosine_similarity

        # Convert item features to array
        item_features_array = np.array(item_features).reshape(1, -1)

        # Calculate similarity with all items
        all_features = self.item_features.values
        similarities = cosine_similarity(item_features_array, all_features)[0]

        # Get top N similar items
        similar_indices = np.argsort(similarities)[::-1][:n]
        similar_items = [
            {
                'item_id': self.item_features.index[idx],
                'similarity': similarities[idx]
            }
            for idx in similar_indices
        ]

        return similar_items

    def _get_random_popular_items(self, n=5):
        """Get random popular items as fallback"""
        # In a real system, this would query a database for popular items
        # For this example, we'll return dummy data
        return [
            {'item_id': '1', 'popularity': 0.9},
            {'item_id': '3', 'popularity': 0.8},
            {'item_id': '5', 'popularity': 0.7},
            {'item_id': '2', 'popularity': 0.6},
            {'item_id': '4', 'popularity': 0.5}
        ][:n]

# Initialize cold start handler
cold_start_handler = ColdStartHandler(item_features=item_features)

# Modified API endpoint to handle cold start
@app.route("/api/recommendations_with_cold_start", methods=["GET"])
def get_recommendations_with_cold_start():
    """Get recommendations with cold start handling"""
    user_id = request.args.get("user_id")
    n = int(request.args.get("n", 5))

    # Check if this is a new user
    is_new_user = user_id not in user_item_matrix.index

    if is_new_user:
        # Handle cold start for new user
        recommendations = cold_start_handler.recommend_for_new_user(n)
        return jsonify({
            "recommendations": recommendations,
            "source": "cold_start_handler",
            "new_user": True
        })

    # For existing users, proceed with normal recommendation flow
    model_type = request.args.get("model", "hybrid")

    # Parse context if provided
    context = {}
    for key in request.args:
        if key.startswith("context_"):
            context_key = key[8:]
            context[context_key] = request.args.get(key)

    # Check cache
    cached_recs = recommendation_cache.get_recommendations(user_id, context, n)
    if cached_recs:
        return jsonify({
            "recommendations": cached_recs,
            "source": "cache",
            "new_user": False
        })

    # Get appropriate model
    recommender = model_manager.get_model(model_type)
    if recommender is None:
        return jsonify({"error": f"Model type '{model_type}' not found"}), 400

    # Generate recommendations
    try:
        recommendations = get_real_time_recommendations(recommender, user_id, context, n)

        # Store in cache
        recommendation_cache.store_recommendations(user_id, recommendations, context, n)

        return jsonify({
            "recommendations": recommendations,
            "source": "model",
            "new_user": False
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

## Deploying with Docker

Containerization makes deployment more consistent and scalable. Here's a sample Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p models

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
```

And a docker-compose.yml file for orchestrating services:

```yaml
version: "3"

services:
  recommender-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    environment:
      - FLASK_ENV=production
      - REDIS_HOST=redis
    depends_on:
      - redis

  redis:
    image: redis:6
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

volumes:
  redis-data:
```

## Conclusion

Deploying recommendation systems in production involves addressing several challenges:

1. **Scalability**: Using techniques like caching, distributed processing, and efficient algorithms to handle large user bases.

2. **Real-time vs. Batch Processing**: Choosing the right approach based on your application's needs.

3. **Model Updating**: Implementing strategies for regular model retraining to incorporate new data.

4. **Cold Start**: Handling new users and items with limited data.

5. **A/B Testing**: Continuously evaluating and improving recommendation algorithms.

6. **Monitoring**: Tracking system performance and recommendation quality.

By addressing these challenges, you can build robust recommendation systems that provide value to users and achieve business objectives.

In the next lesson, we'll explore advanced topics in recommendation systems, including ethical considerations, privacy concerns, and emerging trends in the field.
