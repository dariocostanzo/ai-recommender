# Lesson 9: Advanced Topics in Recommendation Systems

## Introduction

In previous lessons, we've covered the fundamentals of recommendation systems, from basic collaborative filtering to deep learning approaches, and how to deploy them in production. In this final lesson, we'll explore advanced topics in recommendation systems, including ethical considerations, privacy concerns, and emerging trends in the field.

## Ethical Considerations in Recommendation Systems

Recommendation systems have become ubiquitous in our digital lives, but they also raise important ethical questions:

### Filter Bubbles and Echo Chambers

Recommendation systems can inadvertently create "filter bubbles" where users are only exposed to content that aligns with their existing preferences and beliefs.

```python
class DiversityEnhancer:
    """
    Enhances diversity in recommendations to combat filter bubbles
    """
    def __init__(self, diversity_weight=0.3, similarity_metric=None):
        """
        Initialize diversity enhancer

        Parameters:
        diversity_weight: Weight given to diversity (0-1)
        similarity_metric: Function to compute similarity between items
        """
        self.diversity_weight = diversity_weight
        self.similarity_metric = similarity_metric or self._default_similarity

    def _default_similarity(self, item1, item2):
        """Default similarity metric based on item features"""
        # This is a placeholder - in a real system, you would use item features
        return 0.5  # Return a default similarity

    def rerank_recommendations(self, recommendations, top_n=10):
        """
        Rerank recommendations to increase diversity

        Parameters:
        recommendations: List of recommended items with scores
        top_n: Number of items to return

        Returns:
        Reranked list of recommendations
        """
        if len(recommendations) <= 1:
            return recommendations[:top_n]

        # Start with the highest-scored item
        ranked_list = [recommendations[0]]
        remaining = recommendations[1:]

        # Iteratively add items that maximize the objective function
        while remaining and len(ranked_list) < top_n:
            max_score = -float('inf')
            best_item_idx = -1

            for i, item in enumerate(remaining):
                # Original recommendation score (normalized)
                original_score = item.get('score', 0.5)

                # Diversity score (average dissimilarity with items in ranked list)
                diversity_score = 0
                for ranked_item in ranked_list:
                    similarity = self.similarity_metric(item, ranked_item)
                    diversity_score += (1 - similarity)

                if ranked_list:
                    diversity_score /= len(ranked_list)

                # Combined score with diversity weight
                combined_score = (1 - self.diversity_weight) * original_score + \
                                self.diversity_weight * diversity_score

                if combined_score > max_score:
                    max_score = combined_score
                    best_item_idx = i

            if best_item_idx >= 0:
                ranked_list.append(remaining.pop(best_item_idx))
            else:
                break

        return ranked_list
```

### Fairness and Bias

Recommendation systems can perpetuate or amplify existing biases in the data:

```python
class FairnessAwareRecommender:
    """
    Recommender that considers fairness constraints
    """
    def __init__(self, base_recommender, sensitive_attributes=None, fairness_constraints=None):
        """
        Initialize fairness-aware recommender

        Parameters:
        base_recommender: Base recommendation algorithm
        sensitive_attributes: Dictionary mapping items to sensitive attributes
        fairness_constraints: Dictionary with fairness constraints
        """
        self.base_recommender = base_recommender
        self.sensitive_attributes = sensitive_attributes or {}
        self.fairness_constraints = fairness_constraints or {}

    def recommend_items(self, user_id, n=10):
        """
        Generate recommendations with fairness constraints

        Parameters:
        user_id: User ID
        n: Number of recommendations

        Returns:
        Fair recommendations
        """
        # Get base recommendations (more than needed)
        base_recs = self.base_recommender.recommend_items(user_id, n=n*3)

        # Apply fairness constraints
        fair_recs = self._apply_fairness_constraints(base_recs, n)

        return fair_recs

    def _apply_fairness_constraints(self, recommendations, n):
        """Apply fairness constraints to recommendations"""
        if not self.fairness_constraints or not self.sensitive_attributes:
            return recommendations[:n]

        # Count items by sensitive attribute
        attribute_counts = {}

        # Initialize with zeros
        for attr_name, attr_values in self.fairness_constraints.items():
            attribute_counts[attr_name] = {value: 0 for value in attr_values.keys()}

        # Select items while satisfying constraints
        selected_items = []

        for item in recommendations:
            item_id = item.get('item_id')

            # Check if adding this item would violate constraints
            violates_constraint = False

            for attr_name, attr_values in self.fairness_constraints.items():
                if attr_name in self.sensitive_attributes.get(item_id, {}):
                    item_attr_value = self.sensitive_attributes[item_id][attr_name]

                    # Check if we've reached the maximum for this attribute value
                    if item_attr_value in attr_values:
                        current_ratio = attribute_counts[attr_name].get(item_attr_value, 0) / max(1, len(selected_items))
                        max_ratio = attr_values[item_attr_value]

                        if current_ratio >= max_ratio and len(selected_items) > 0:
                            violates_constraint = True
                            break

            if not violates_constraint:
                selected_items.append(item)

                # Update counts
                for attr_name in self.fairness_constraints.keys():
                    if attr_name in self.sensitive_attributes.get(item_id, {}):
                        item_attr_value = self.sensitive_attributes[item_id][attr_name]
                        attribute_counts[attr_name][item_attr_value] = attribute_counts[attr_name].get(item_attr_value, 0) + 1

            if len(selected_items) >= n:
                break

        # If we don't have enough items, add more from the original list
        if len(selected_items) < n:
            remaining = [item for item in recommendations if item not in selected_items]
            selected_items.extend(remaining[:n - len(selected_items)])

        return selected_items[:n]
```

### Transparency and Explainability

Users should understand why certain recommendations are made:

```python
class ExplainableRecommender:
    """
    Wrapper for making recommendations explainable
    """
    def __init__(self, base_recommender, item_features=None, user_item_matrix=None):
        """
        Initialize explainable recommender

        Parameters:
        base_recommender: Base recommendation algorithm
        item_features: Features of items
        user_item_matrix: User-item interaction matrix
        """
        self.base_recommender = base_recommender
        self.item_features = item_features
        self.user_item_matrix = user_item_matrix

    def recommend_with_explanations(self, user_id, n=5):
        """
        Generate recommendations with explanations

        Parameters:
        user_id: User ID
        n: Number of recommendations

        Returns:
        Recommendations with explanations
        """
        # Get base recommendations
        recommendations = self.base_recommender.recommend_items(user_id, n=n)

        # Add explanations
        for rec in recommendations:
            rec['explanation'] = self._generate_explanation(user_id, rec['item_id'])

        return recommendations

    def _generate_explanation(self, user_id, item_id):
        """Generate explanation for a recommendation"""
        explanation_types = [
            self._content_based_explanation,
            self._collaborative_explanation,
            self._popularity_explanation
        ]

        # Try different explanation types
        for explanation_func in explanation_types:
            explanation = explanation_func(user_id, item_id)
            if explanation:
                return explanation

        # Fallback explanation
        return "This item might interest you based on your preferences."

    def _content_based_explanation(self, user_id, item_id):
        """Generate content-based explanation"""
        if self.item_features is None or item_id not in self.item_features.index:
            return None

        # Get item features
        item_features = self.item_features.loc[item_id]

        # Find most prominent features
        top_features = item_features.nlargest(3).index.tolist()

        if top_features:
            return f"This item has {', '.join(top_features)} that you might enjoy."

        return None

    def _collaborative_explanation(self, user_id, item_id):
        """Generate collaborative filtering explanation"""
        if self.user_item_matrix is None:
            return None

        # Find similar users who liked this item
        if user_id in self.user_item_matrix.index and item_id in self.user_item_matrix.columns:
            # Get users who rated this item highly
            item_ratings = self.user_item_matrix[item_id]
            high_ratings = item_ratings[item_ratings > 3].index.tolist()

            if high_ratings:
                return f"Users with similar tastes have enjoyed this item."

        return None

    def _popularity_explanation(self, user_id, item_id):
        """Generate popularity-based explanation"""
        if self.user_item_matrix is None or item_id not in self.user_item_matrix.columns:
            return None

        # Calculate item popularity
        item_ratings = self.user_item_matrix[item_id]
        num_ratings = (item_ratings > 0).sum()
        avg_rating = item_ratings[item_ratings > 0].mean()

        if num_ratings > 10 and avg_rating > 3.5:
            return f"This is a popular item with an average rating of {avg_rating:.1f} from {num_ratings} users."

        return None
```

## Privacy in Recommendation Systems

Recommendation systems rely on user data, raising privacy concerns:

### Federated Learning

Federated learning allows training models without sharing raw user data:

```python
class FederatedRecommender:
    """
    Recommender using federated learning principles
    """
    def __init__(self, model_architecture, num_clients=10):
        """
        Initialize federated recommender

        Parameters:
        model_architecture: Neural network architecture for recommendations
        num_clients: Number of simulated clients
        """
        self.model_architecture = model_architecture
        self.num_clients = num_clients
        self.global_model = self._initialize_model()
        self.client_data = {}  # In a real system, this would be distributed

    def _initialize_model(self):
        """Initialize global model with the given architecture"""
        # This is a simplified example
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(50,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10)  # Output layer
        ])

        model.compile(
            optimizer='adam',
            loss='mse'
        )

        return model

    def _get_model_weights(self):
        """Get current model weights"""
        return self.global_model.get_weights()

    def _set_model_weights(self, weights):
        """Set model weights"""
        self.global_model.set_weights(weights)

    def federated_training(self, num_rounds=5, local_epochs=1):
        """
        Perform federated training

        Parameters:
        num_rounds: Number of federated rounds
        local_epochs: Number of local epochs per client
        """
        import numpy as np

        for round_num in range(num_rounds):
            print(f"Federated Round {round_num + 1}/{num_rounds}")

            # Get current global model weights
            global_weights = self._get_model_weights()

            # List to store updated weights from clients
            client_weights = []

            # Simulate client updates
            for client_id in range(self.num_clients):
                # In a real system, this would happen on client devices
                client_model = self._initialize_model()
                client_model.set_weights(global_weights)

                # Get client data (in a real system, this would be local to the client)
                X, y = self._get_client_data(client_id)

                if len(X) > 0:
                    # Train on client data
                    client_model.fit(X, y, epochs=local_epochs, verbose=0)

                    # Get updated weights
                    updated_weights = client_model.get_weights()
                    client_weights.append(updated_weights)

            # Aggregate client weights (simple average in this example)
            # In a real system, you might use more sophisticated aggregation
            new_global_weights = []
            for weight_list in zip(*client_weights):
                new_global_weights.append(np.mean(weight_list, axis=0))

            # Update global model
            self._set_model_weights(new_global_weights)

            # Evaluate global model
            test_loss = self._evaluate_global_model()
            print(f"Round {round_num + 1} completed. Test loss: {test_loss:.4f}")

    def _get_client_data(self, client_id):
        """Get data for a specific client"""
        # In a real system, this data would be on the client device
        if client_id in self.client_data:
            return self.client_data[client_id]

        # Simulate empty data
        return np.array([]), np.array([])

    def _evaluate_global_model(self):
        """Evaluate global model on test data"""
        # This is a simplified example
        # In a real system, you would have a separate test set
        return 0.1  # Placeholder loss value

    def recommend_items(self, user_features, n=5):
        """
        Generate recommendations for a user

        Parameters:
        user_features: Features representing the user
        n: Number of recommendations

        Returns:
        List of recommended item IDs
        """
        # This is a simplified example
        # In a real system, you would convert user features to model input
        import numpy as np

        # Ensure user features have the right shape
        if len(user_features.shape) == 1:
            user_features = np.expand_dims(user_features, axis=0)

        # Get model predictions
        predictions = self.global_model.predict(user_features)

        # Get top N items
        top_indices = np.argsort(predictions[0])[-n:][::-1]

        # Convert indices to item IDs (in a real system)
        item_ids = [f"item_{idx}" for idx in top_indices]

        return item_ids
```

### Differential Privacy

Differential privacy adds noise to data to protect individual privacy:

```python
class DifferentialPrivacyRecommender:
    """
    Recommender with differential privacy guarantees
    """
    def __init__(self, base_recommender, epsilon=1.0, delta=1e-5):
        """
        Initialize differentially private recommender

        Parameters:
        base_recommender: Base recommendation algorithm
        epsilon: Privacy parameter (smaller = more privacy)
        delta: Probability of privacy failure
        """
        self.base_recommender = base_recommender
        self.epsilon = epsilon
        self.delta = delta

    def add_noise_to_ratings(self, user_item_matrix):
        """
        Add Laplacian noise to user-item matrix for differential privacy

        Parameters:
        user_item_matrix: Original user-item interaction matrix

        Returns:
        Noisy user-item matrix
        """
        import numpy as np

        # Calculate sensitivity (assuming ratings are bounded)
        sensitivity = 1.0  # Assuming ratings are normalized or bounded

        # Calculate scale parameter for Laplace noise
        scale = sensitivity / self.epsilon

        # Create copy of matrix
        noisy_matrix = user_item_matrix.copy()

        # Add Laplace noise to non-zero entries
        for user in noisy_matrix.index:
            for item in noisy_matrix.columns:
                if noisy_matrix.loc[user, item] > 0:
                    noise = np.random.laplace(0, scale)
                    noisy_matrix.loc[user, item] += noise

                    # Ensure ratings stay within bounds (e.g., 1-5)
                    noisy_matrix.loc[user, item] = max(1, min(5, noisy_matrix.loc[user, item]))

        return noisy_matrix

    def recommend_items(self, user_id, n=5):
        """
        Generate differentially private recommendations

        Parameters:
        user_id: User ID
        n: Number of recommendations

        Returns:
        Differentially private recommendations
        """
        # In a real system, you would apply differential privacy
        # to the training process or the recommendation algorithm

        # For this example, we'll use a simple approach:
        # 1. Get recommendations from base recommender
        base_recs = self.base_recommender.recommend_items(user_id, n=n*2)

        # 2. Add noise to scores
        import numpy as np

        sensitivity = 1.0
        scale = sensitivity / self.epsilon

        for rec in base_recs:
            if 'score' in rec:
                noise = np.random.laplace(0, scale)
                rec['score'] += noise
                rec['score'] = max(0, rec['score'])  # Ensure non-negative

        # 3. Re-rank and return top N
        noisy_recs = sorted(base_recs, key=lambda x: x.get('score', 0), reverse=True)[:n]

        return noisy_recs
```

## Emerging Trends in Recommendation Systems

### Reinforcement Learning for Recommendations

Reinforcement learning optimizes recommendations for long-term user engagement:

```python
class RLRecommender:
    """
    Recommender using reinforcement learning
    """
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95):
        """
        Initialize RL-based recommender

        Parameters:
        state_size: Size of state representation
        action_size: Number of possible actions (items)
        learning_rate: Learning rate for model updates
        gamma: Discount factor for future rewards
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize Q-network
        self.model = self._build_model()

        # Memory for experience replay
        self.memory = []
        self.max_memory_size = 2000

    def _build_model(self):
        """Build neural network model for Q-learning"""
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )

        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0)

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=0.1):
        """
        Choose action based on epsilon-greedy policy

        Parameters:
        state: Current state
        epsilon: Exploration rate

        Returns:
        Selected action (item)
        """
        import numpy as np

        if np.random.random() < epsilon:
            # Explore: choose random action
            return np.random.randint(self.action_size)
        else:
            # Exploit: choose best action
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size=32):
        """Train model using experience replay"""
        import numpy as np

        if len(self.memory) < batch_size:
            return

        # Sample random batch from memory
        minibatch = np.random.choice(self.memory, batch_size, replace=False)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

    def get_state_representation(self, user_id, user_history):
        """
        Convert user history to state representation

        Parameters:
        user_id: User ID
        user_history: List of user interactions

        Returns:
        State representation as numpy array
        """
        import numpy as np

        # This is a simplified example
        # In a real system, you would create a more sophisticated state representation

        # Create a vector of zeros
        state = np.zeros(self.state_size)

        # Fill in based on user history
        for i, item in enumerate(user_history[-self.state_size:]):
            if i < self.state_size:
                state[i] = item

        return np.reshape(state, [1, self.state_size])

    def recommend_items(self, user_id, user_history, n=5, epsilon=0.05):
        """
        Generate recommendations using RL

        Parameters:
        user_id: User ID
        user_history: User interaction history
        n: Number of recommendations
        epsilon: Exploration rate

        Returns:
        List of recommended items
        """
        import numpy as np

        # Get state representation
        state = self.get_state_representation(user_id, user_history)

        # Get Q-values for all actions
        q_values = self.model.predict(state)[0]

        # Get top N actions (items)
        recommended_items = []

        # Mix of exploitation and exploration
        if np.random.random() < epsilon:
            # Include some random items
            random_items = np.random.choice(self.action_size, size=int(n/5), replace=False)
            for item in random_items:
                recommended_items.append({
                    'item_id': int(item),
                    'score': float(q_values[item])
                })

        # Fill the rest with highest Q-value items
        top_items = np.argsort(q_values)[::-1]
        for item in top_items:
            if len(recommended_items) >= n:
                break

            if item not in [rec['item_id'] for rec in recommended_items]:
                recommended_items.append({
                    'item_id': int(item),
                    'score': float(q_values[item])
                })

        return recommended_items
```

### Multi-Modal Recommendations

Multi-modal recommendations incorporate different types of data:

```python
class MultiModalRecommender:
    """
    Recommender using multiple modalities (text, images, etc.)
    """
    def __init__(self, text_encoder=None, image_encoder=None, fusion_model=None):
        """
        Initialize multi-modal recommender

        Parameters:
        text_encoder: Model to encode text features
        image_encoder: Model to encode image features
        fusion_model: Model to fuse different modalities
        """
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion_model = fusion_model

    def encode_item(self, item_id, item_data):
        """
        Encode item using multiple modalities

        Parameters:
        item_id: Item ID
        item_data: Dictionary with item data (text, images, etc.)

        Returns:
        Combined item embedding
        """
        import numpy as np

        embeddings = []

        # Encode text if available
        if 'text' in item_data and self.text_encoder:
            text_embedding = self.text_encoder.encode(item_data['text'])
            embeddings.append(text_embedding)

        # Encode image if available
        if 'image' in item_data and self.image_encoder:
            image_embedding = self.image_encoder.encode(item_data['image'])
            embeddings.append(image_embedding)

        # Fuse embeddings if multiple modalities are available
        if len(embeddings) > 1 and self.fusion_model:
            return self.fusion_model.fuse(embeddings)
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            # Fallback to empty embedding
            return np.zeros(100)  # Assuming 100-dim embedding

    def encode_user(self, user_id, user_history):
        """
        Encode user based on interaction history

        Parameters:
        user_id: User ID
        user_history: List of items the user has interacted with

        Returns:
        User embedding
        """
        import numpy as np

        # This is a simplified example
        # In a real system, you would aggregate item embeddings

        # Get item embeddings
        item_embeddings = []
        for item_id in user_history:
            if item_id in self.item_embeddings:
                item_embeddings.append(self.item_embeddings[item_id])

        if item_embeddings:
            # Average item embeddings
            return np.mean(item_embeddings, axis=0)
        else:
            # Fallback to empty embedding
            return np.zeros(100)  # Assuming 100-dim embedding

    def recommend_items(self, user_id, user_history, n=5):
        """
        Generate multi-modal recommendations

        Parameters:
        user_id: User ID
        user_history: User interaction history
        n: Number of recommendations

        Returns:
        List of recommended items
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # Encode user
        user_embedding = self.encode_user(user_id, user_history)

        # Calculate similarity with all items
        similarities = {}
        for item_id, item_embedding in self.item_embeddings.items():
            if item_id not in user_history:  # Exclude items the user has already interacted with
                sim = cosine_similarity([user_embedding], [item_embedding])[0][0]
                similarities[item_id] = sim

        # Sort by similarity
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Return top N items
        recommendations = []
        for item_id, score in sorted_items[:n]:
            recommendations.append({
                'item_id': item_id,
                'score': float(score)
            })

        return recommendations
```

### Graph-Based Recommendations

Graph-based approaches model complex relationships between users and items:

```python
class GraphRecommender:
    """
    Recommender using graph neural networks
    """
    def __init__(self, num_users, num_items, embedding_dim=64):
        """
        Initialize graph-based recommender

        Parameters:
        num_users: Number of users
        num_items: Number of items
        embedding_dim: Dimension of embeddings
        """
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Initialize model
        self.model = self._build_model()

        # Adjacency matrix (user-item interactions)
        self.adjacency_matrix = None

    def _build_model(self):
        """Build graph neural network model"""
        import tensorflow as tf

        # User embeddings
        user_embeddings = tf.keras.layers.Embedding(
            self.num_users, self.embedding_dim, name='user_embeddings'
        )

        # Item embeddings
        item_embeddings = tf.keras.layers.Embedding(
            self.num_items, self.embedding_dim, name='item_embeddings'
        )

        # Graph convolutional layers
        graph_conv_1 = tf.keras.layers.Dense(self.embedding_dim, activation='relu')
        graph_conv_2 = tf.keras.layers.Dense(self.embedding_dim, activation='relu')

        # Prediction layer
        prediction = tf.keras.layers.Dense(1)

        # Define model
        user_input = tf.keras.layers.Input(shape=(1,))
        item_input = tf.keras.layers.Input(shape=(1,))

        user_embedding = user_embeddings(user_input)
        item_embedding = item_embeddings(item_input)

        # Concatenate embeddings
        concat_embedding = tf.keras.layers.Concatenate()([user_embedding, item_embedding])

        # Apply graph convolutions
        x = graph_conv_1(concat_embedding)
        x = graph_conv_2(x)

        # Prediction
        output = prediction(x)

        # Create model
        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)

        # Compile model
        model.compile(
            loss='mse',
            optimizer='adam'
        )

        return model

    def build_adjacency_matrix(self, user_item_matrix):
        """
        Build adjacency matrix from user-item interactions

        Parameters:
        user_item_matrix: User-item interaction matrix
        """
        import numpy as np
        import scipy.sparse as sp

        # Convert to binary matrix (1 for interaction, 0 otherwise)
        binary_matrix = (user_item_matrix > 0).astype(np.float32)

        # Create sparse matrix
        self.adjacency_matrix = sp.csr_matrix(binary_matrix)

    def train(self, user_item_matrix, epochs=10, batch_size=64):
        """
        Train graph-based recommender

        Parameters:
        user_item_matrix: User-item interaction matrix
        epochs: Number of training epochs
        batch_size: Batch size
        """
        import numpy as np

        # Build adjacency matrix
        self.build_adjacency_matrix(user_item_matrix)

        # Prepare training data
        users, items, ratings = [], [], []

        for user in range(user_item_matrix.shape[0]):
            for item in range(user_item_matrix.shape[1]):
                if user_item_matrix.iloc[user, item] > 0:
                    users.append(user)
                    items.append(item)
                    ratings.append(user_item_matrix.iloc[user, item])

        # Convert to numpy arrays
        users = np.array(users)
        items = np.array(items)
        ratings = np.array(ratings)

        # Train model
        self.model.fit(
            [users, items],
            ratings,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )

    def recommend_items(self, user_id, n=5):
        """
        Generate recommendations using graph neural network

        Parameters:
        user_id: User ID
        n: Number of recommendations

        Returns:
        List of recommended items
        """
        import numpy as np

        # Generate predictions for all items
        items = np.array(range(self.num_items))
        users = np.array([user_id] * self.num_items)

        # Predict ratings
        predictions = self.model.predict([users, items]).flatten()

        # Get top N items
        top_indices = np.argsort(predictions)[::-1][:n]

        # Create recommendation list
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'item_id': int(idx),
                'score': float(predictions[idx])
            })

        return recommendations
```

### Self-Supervised Learning for Recommendations

Self-supervised learning leverages unlabeled data to improve recommendation quality:

```python
class SelfSupervisedRecommender:
    """
    Recommender using self-supervised learning techniques
    """
    def __init__(self, num_users, num_items, embedding_dim=64):
        """
        Initialize self-supervised recommender

        Parameters:
        num_users: Number of users
        num_items: Number of items
        embedding_dim: Dimension of embeddings
        """
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Initialize models
        self.encoder = self._build_encoder()
        self.predictor = self._build_predictor()

    def _build_encoder(self):
        """Build encoder model for self-supervised learning"""
        import tensorflow as tf

        # Input layer
        input_layer = tf.keras.layers.Input(shape=(self.num_items,))

        # Encoder layers
        x = tf.keras.layers.Dense(256, activation='relu')(input_layer)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # Embedding layer
        embedding = tf.keras.layers.Dense(self.embedding_dim, activation=None)(x)

        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=embedding)

        return model

    def _build_predictor(self):
        """Build predictor model for recommendations"""
        import tensorflow as tf

        # User embedding input
        user_embedding = tf.keras.layers.Input(shape=(self.embedding_dim,))

        # Item embedding input
        item_embedding = tf.keras.layers.Input(shape=(self.embedding_dim,))

        # Concatenate embeddings
        concat = tf.keras.layers.Concatenate()([user_embedding, item_embedding])

        # Prediction layers
        x = tf.keras.layers.Dense(64, activation='relu')(concat)
        x = tf.keras.layers.Dense(32, activation='relu')(x)

        # Output layer
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        # Create model
        model = tf.keras.Model(inputs=[user_embedding, item_embedding], outputs=output)

        # Compile model
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    def pretrain_encoder(self, user_item_matrix, epochs=10, batch_size=64):
        """
        Pretrain encoder using self-supervised learning

        Parameters:
        user_item_matrix: User-item interaction matrix
        epochs: Number of training epochs
        batch_size: Batch size
        """
        import tensorflow as tf
        import numpy as np

        # Create self-supervised task: predict masked items
        X_orig = user_item_matrix.values

        # Create pretraining model
        input_layer = tf.keras.layers.Input(shape=(self.num_items,))
        embedding = self.encoder(input_layer)
        decoder = tf.keras.layers.Dense(self.num_items, activation='sigmoid')(embedding)

        pretrain_model = tf.keras.Model(inputs=input_layer, outputs=decoder)
        pretrain_model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # Train with masking (randomly mask 20% of non-zero entries)
        for epoch in range(epochs):
            print(f"Pretraining epoch {epoch+1}/{epochs}")

            # Create masked version of data
            X_masked = X_orig.copy()

            # Mask 20% of non-zero entries
            for i in range(X_orig.shape[0]):
                non_zero = np.where(X_orig[i] > 0)[0]
                if len(non_zero) > 0:
                    mask_indices = np.random.choice(
                        non_zero,
                        size=max(1, int(0.2 * len(non_zero))),
                        replace=False
                    )
                    X_masked[i, mask_indices] = 0

            # Train for one epoch
            pretrain_model.fit(
                X_masked,
                X_orig,
                batch_size=batch_size,
                epochs=1,
                verbose=1
            )

    def train_predictor(self, user_item_matrix, epochs=10, batch_size=64):
        """
        Train predictor model using pretrained encoder

        Parameters:
        user_item_matrix: User-item interaction matrix
        epochs: Number of training epochs
        batch_size: Batch size
        """
        import numpy as np

        # Generate user embeddings
        user_embeddings = self.encoder.predict(user_item_matrix.values)

        # Create item embeddings (simplified approach)
        item_embeddings = np.random.normal(0, 0.1, (self.num_items, self.embedding_dim))

        # Prepare training data
        users, items, labels = [], [], []

        for user in range(user_item_matrix.shape[0]):
            # Positive examples (items the user has interacted with)
            pos_items = np.where(user_item_matrix.iloc[user] > 0)[0]

            for item in pos_items:
                users.append(user)
                items.append(item)
                labels.append(1)

            # Negative examples (items the user has not interacted with)
            neg_items = np.where(user_item_matrix.iloc[user] == 0)[0]
            neg_samples = np.random.choice(
                neg_items,
                size=min(len(pos_items), len(neg_items)),
                replace=False
            )

            for item in neg_samples:
                users.append(user)
                items.append(item)
                labels.append(0)

        # Convert to numpy arrays
        users = np.array(users)
        items = np.array(items)
        labels = np.array(labels)

        # Get embeddings for training
        user_embs = user_embeddings[users]
        item_embs = item_embeddings[items]

        # Train predictor model
        self.predictor.fit(
            [user_embs, item_embs],
            labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )

    def recommend_items(self, user_id, user_item_matrix, n=5):
        """
        Generate recommendations using self-supervised model

        Parameters:
        user_id: User ID
        user_item_matrix: User-item interaction matrix
        n: Number of recommendations

        Returns:
        List of recommended items
        """
        import numpy as np

        # Get user vector
        user_vector = user_item_matrix.iloc[user_id].values.reshape(1, -1)

        # Get user embedding
        user_embedding = self.encoder.predict(user_vector)

        # Generate predictions for all items
        predictions = []

        # Create item embeddings (simplified approach)
        item_embeddings = np.random.normal(0, 0.1, (self.num_items, self.embedding_dim))

        # Get items the user has not interacted with
        non_interacted = np.where(user_vector[0] == 0)[0]

        # Predict scores for non-interacted items
        for item in non_interacted:
            item_embedding = item_embeddings[item].reshape(1, -1)
            score = self.predictor.predict([user_embedding, item_embedding])[0][0]
            predictions.append((item, score))

        # Sort by predicted score
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Return top N recommendations
        recommendations = []
        for item, score in predictions[:n]:
            recommendations.append({
                'item_id': int(item),
                'score': float(score)
            })

        return recommendations
```

## Conclusion

In this lesson, we've explored advanced topics in recommendation systems, including:

1. **Ethical Considerations**:

   - Filter bubbles and echo chambers
   - Fairness and bias in recommendations
   - Transparency and explainability

2. **Privacy Concerns**:

   - Federated learning for privacy-preserving recommendations
   - Differential privacy techniques

3. **Emerging Trends**:
   - Reinforcement learning for optimizing long-term engagement
   - Multi-modal recommendations incorporating different data types
   - Graph-based approaches for modeling complex relationships
   - Self-supervised learning for leveraging unlabeled data

These advanced techniques represent the cutting edge of recommendation systems research and practice. As recommendation systems continue to evolve, they will become more personalized, more privacy-preserving, and more ethically responsible.

The field of recommendation systems is rapidly advancing, with new techniques and approaches being developed to address the challenges of scale, privacy, and personalization. By staying informed about these developments and incorporating them into your recommendation systems, you can create more effective and responsible recommenders that provide value to users while respecting their privacy and autonomy.

## Further Reading

1. Jannach, D., Zanker, M., Felfernig, A., & Friedrich, G. (2010). Recommender systems: an introduction. Cambridge University Press.

2. Aggarwal, C. C. (2016). Recommender systems: The textbook. Springer.

3. Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep learning based recommender system: A survey and new perspectives. ACM Computing Surveys, 52(1), 1-38.

4. Ekstrand, M. D., Tian, M., Azpiazu, I. M., Ekstrand, J. D., Anuyah, O., McNeill, D., & Pera, M. S. (2018). All the cool kids, how do they fit in?: Popularity and demographic biases in recommender evaluation and effectiveness. In Conference on Fairness, Accountability and Transparency.

5. Chen, J., Chang, Y., Hobbs, B., Castaldi, P., Cho, M. H., Silverman, E. K., & Dy, J. (2016). Interpretable clustering via discriminative rectangle mixture model. In IEEE 16th International Conference on Data Mining (ICDM).

This concludes our series on recommendation systems. You now have a comprehensive understanding of the fundamentals, advanced techniques, and practical considerations for building effective recommendation systems. Good luck with your recommendation system projects!
