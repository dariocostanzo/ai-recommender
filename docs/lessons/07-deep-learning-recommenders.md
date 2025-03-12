# Lesson 7: Deep Learning for Recommendation Systems

## Introduction

In this lesson, we'll explore how deep learning techniques can be applied to recommendation systems. Deep learning models have shown impressive results in various domains, including recommendations, by capturing complex patterns and non-linear relationships in data.

## Why Deep Learning for Recommendations?

Traditional recommendation approaches have several limitations:

- **Linear models**: Most traditional methods rely on linear combinations of features
- **Manual feature engineering**: Significant effort is required to design effective features
- **Limited expressiveness**: Difficulty capturing complex user-item interactions
- **Sparse data handling**: Challenges with extremely sparse user-item matrices

Deep learning models address these limitations by:

- Learning hierarchical representations automatically
- Capturing non-linear relationships between users and items
- Integrating heterogeneous data sources more effectively
- Modeling sequential patterns in user behavior

## Neural Collaborative Filtering (NCF)

Neural Collaborative Filtering extends traditional matrix factorization by using neural networks to learn the user-item interaction function.

### Basic NCF Architecture

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class NCFDataset(Dataset):
    def __init__(self, user_item_interactions, all_item_ids, negative_samples=4):
        self.user_ids = []
        self.item_ids = []
        self.labels = []

        # Create positive samples
        for user_id, item_id, rating in user_item_interactions:
            self.user_ids.append(user_id)
            self.item_ids.append(item_id)
            self.labels.append(1)  # Positive interaction

            # Generate negative samples
            negative_count = 0
            while negative_count < negative_samples:
                # Sample a random item
                neg_item = np.random.choice(all_item_ids)

                # Check if this is a positive interaction
                if not any((user_id == u and neg_item == i) for u, i, _ in user_item_interactions):
                    self.user_ids.append(user_id)
                    self.item_ids.append(neg_item)
                    self.labels.append(0)  # Negative interaction
                    negative_count += 1

        self.user_ids = torch.tensor(self.user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(self.item_ids, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'label': self.labels[idx]
        }

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, layers=[64, 32, 16, 8]):
        super(NeuralCollaborativeFiltering, self).__init__()

        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        self.fc_layers = nn.ModuleList()
        layer_dims = [2 * embedding_dim] + layers

        for i in range(len(layer_dims) - 1):
            self.fc_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.fc_layers.append(nn.ReLU())

        # Output layer
        self.output_layer = nn.Linear(layer_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_id, item_id):
        # Get embeddings
        user_embedded = self.user_embedding(user_id)
        item_embedded = self.item_embedding(item_id)

        # Concatenate user and item embeddings
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Forward through MLP layers
        for layer in self.fc_layers:
            vector = layer(vector)

        # Output prediction
        prediction = self.sigmoid(self.output_layer(vector))

        return prediction.squeeze()

class NCFRecommender:
    def __init__(self, user_item_matrix, embedding_dim=32, layers=[64, 32, 16, 8],
                 learning_rate=0.001, batch_size=256, num_epochs=20):
        self.user_item_matrix = user_item_matrix
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Create user and item mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(user_item_matrix.index)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}

        self.item_to_idx = {item: idx for idx, item in enumerate(user_item_matrix.columns)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        # Initialize model
        self.model = NeuralCollaborativeFiltering(
            num_users=len(self.user_to_idx),
            num_items=len(self.item_to_idx),
            embedding_dim=embedding_dim,
            layers=layers
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

    def _prepare_training_data(self):
        interactions = []

        for user in self.user_item_matrix.index:
            user_idx = self.user_to_idx[user]
            for item in self.user_item_matrix.columns:
                rating = self.user_item_matrix.loc[user, item]
                if rating > 0:  # Consider only positive interactions
                    item_idx = self.item_to_idx[item]
                    interactions.append((user_idx, item_idx, rating))

        all_item_indices = list(self.item_to_idx.values())

        return interactions, all_item_indices

    def train(self):
        interactions, all_item_indices = self._prepare_training_data()

        # Create dataset and dataloader
        dataset = NCFDataset(interactions, all_item_indices)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in dataloader:
                user_id = batch['user_id'].to(self.device)
                item_id = batch['item_id'].to(self.device)
                label = batch['label'].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                prediction = self.model(user_id, item_id)
                loss = self.criterion(prediction, label)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

    def predict_rating(self, user_id, item_id):
        """Predict rating for a user-item pair"""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 0.0

        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]

        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
            prediction = self.model(user_tensor, item_tensor)

            # Scale prediction from [0,1] to [1,5]
            scaled_rating = 1 + prediction.item() * 4

            return scaled_rating

    def recommend_items(self, user_id, n=5):
        """Recommend top N items for a user"""
        if user_id not in self.user_to_idx:
            return pd.DataFrame(columns=['item_id', 'predicted_rating'])

        # Get items the user hasn't rated yet
        user_idx = self.user_to_idx[user_id]
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = [item for item in self.user_item_matrix.columns if user_ratings[item] == 0]

        if not unrated_items:
            return pd.DataFrame(columns=['item_id', 'predicted_rating'])

        # Predict ratings for unrated items
        predictions = []
        for item in unrated_items:
            item_idx = self.item_to_idx[item]
            predicted_rating = self.predict_rating(user_id, item)
            predictions.append({
                'item_id': item,
                'predicted_rating': predicted_rating
            })

        # Sort by predicted rating and return top N
        recommendations = pd.DataFrame(predictions)
        recommendations = recommendations.sort_values('predicted_rating', ascending=False).head(n)

        return recommendations
```

### GMF: Generalized Matrix Factorization

GMF is a special case of NCF that generalizes matrix factorization using neural networks:

```python
class GMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(GMF, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Output layer
        self.output_layer = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_id, item_id):
        # Get embeddings
        user_embedded = self.user_embedding(user_id)
        item_embedded = self.item_embedding(item_id)

        # Element-wise product
        element_product = torch.mul(user_embedded, item_embedded)

        # Output prediction
        prediction = self.sigmoid(self.output_layer(element_product))

        return prediction.squeeze()
```

### NeuMF: Neural Matrix Factorization

NeuMF combines GMF and MLP to better model user-item interactions:

```python
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, mlp_layers=[64, 32, 16, 8], alpha=0.5):
        super(NeuMF, self).__init__()

        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        self.fc_layers = nn.ModuleList()
        layer_dims = [2 * embedding_dim] + mlp_layers

        for i in range(len(layer_dims) - 1):
            self.fc_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.fc_layers.append(nn.ReLU())

        # Output layer
        self.output_layer = nn.Linear(mlp_layers[-1] + embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_id, item_id):
        # GMF part
        gmf_user_embedded = self.gmf_user_embedding(user_id)
        gmf_item_embedded = self.gmf_item_embedding(item_id)
        gmf_vector = torch.mul(gmf_user_embedded, gmf_item_embedded)

        # MLP part
        mlp_user_embedded = self.mlp_user_embedding(user_id)
        mlp_item_embedded = self.mlp_item_embedding(item_id)
        mlp_vector = torch.cat([mlp_user_embedded, mlp_item_embedded], dim=-1)

        for layer in self.fc_layers:
            mlp_vector = layer(mlp_vector)

        # Concatenate GMF and MLP vectors
        vector = torch.cat([gmf_vector, mlp_vector], dim=-1)

        # Output prediction
        prediction = self.sigmoid(self.output_layer(vector))

        return prediction.squeeze()
```

## Sequence-Based Recommenders

Sequence-based recommenders model the sequential patterns in user behavior, which is particularly important for capturing evolving user preferences.

### Recurrent Neural Networks (RNN) for Recommendations

```python
class RNNRecommender(nn.Module):
    def __init__(self, num_items, embedding_dim=32, hidden_dim=64, num_layers=1, dropout=0.2):
        super(RNNRecommender, self).__init__()

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)  # +1 for padding
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(self, item_sequences, sequence_lengths):
        # Embed the item sequences
        embedded = self.item_embedding(item_sequences)

        # Pack padded sequence for variable length inputs
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, sequence_lengths, batch_first=True, enforce_sorted=False
        )

        # Process with RNN
        output, hidden = self.rnn(packed)

        # Unpack the sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Get the last output for each sequence
        last_outputs = []
        for i, length in enumerate(sequence_lengths):
            last_outputs.append(output[i, length-1])
        last_outputs = torch.stack(last_outputs)

        # Predict next item
        logits = self.fc(last_outputs)

        return logits

class SequentialRecommender:
    def __init__(self, user_sequences, num_items, embedding_dim=32, hidden_dim=64,
                 num_layers=1, dropout=0.2, learning_rate=0.001, batch_size=32, num_epochs=20):
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Create user and item mappings
        self.users = list(user_sequences.keys())
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}

        # Initialize model
        self.model = RNNRecommender(
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def _prepare_sequences(self, sequences, max_len=10):
        """Prepare sequences for training"""
        X = []
        y = []
        lengths = []

        for sequence in sequences:
            if len(sequence) < 2:
                continue

            # Use all possible subsequences for training
            for end_idx in range(2, len(sequence) + 1):
                seq = sequence[:end_idx-1]
                target = sequence[end_idx-1]

                # Pad sequence if needed
                if len(seq) < max_len:
                    seq = [0] * (max_len - len(seq)) + seq
                else:
                    seq = seq[-max_len:]

                X.append(seq)
                y.append(target)
                lengths.append(min(end_idx-1, max_len))

        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    def train(self, max_sequence_length=10):
        """Train the sequential recommender"""
        # Prepare all sequences for training
        all_sequences = []
        for user, sequence in self.user_sequences.items():
            all_sequences.append(sequence)

        X, y, lengths = self._prepare_sequences(all_sequences, max_len=max_sequence_length)

        # Create dataset
        dataset = torch.utils.data.TensorDataset(X, y, lengths)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_X, batch_y, batch_lengths in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_lengths = batch_lengths.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(batch_X, batch_lengths)
                loss = self.criterion(logits, batch_y)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

    def recommend_next_items(self, user_id, n=5):
        """Recommend next items for a user based on their sequence"""
        if user_id not in self.user_sequences:
            return pd.DataFrame(columns=['item_id', 'score'])

        sequence = self.user_sequences[user_id]
        if not sequence:
            return pd.DataFrame(columns=['item_id', 'score'])

        # Prepare sequence
        max_len = 10
        if len(sequence) < max_len:
            padded_sequence = [0] * (max_len - len(sequence)) + sequence
            sequence_length = len(sequence)
        else:
            padded_sequence = sequence[-max_len:]
            sequence_length = max_len

        # Convert to tensor
        sequence_tensor = torch.tensor([padded_sequence], dtype=torch.long).to(self.device)
        length_tensor = torch.tensor([sequence_length], dtype=torch.long).to(self.device)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            logits = self.model(sequence_tensor, length_tensor)
            probabilities = torch.softmax(logits, dim=1)

        # Convert to recommendations
        item_scores = probabilities[0].cpu().numpy()

        # Filter out items already in the sequence
        for item in sequence:
            if item > 0 and item < len(item_scores):
                item_scores[item] = 0

        # Get top N items
        top_items = np.argsort(item_scores)[::-1][:n]

        recommendations = []
        for item_id in top_items:
            if item_id > 0:  # Skip padding item
                recommendations.append({
                    'item_id': int(item_id),
                    'score': float(item_scores[item_id])
                })

        return pd.DataFrame(recommendations)
```

### Transformer-Based Recommenders

Transformers have revolutionized sequence modeling with their self-attention mechanism. Let's implement a simple transformer-based recommender:

```python
class TransformerRecommender(nn.Module):
    def __init__(self, num_items, embedding_dim=32, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerRecommender, self).__init__()

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)  # +1 for padding
        self.position_embedding = nn.Embedding(50, embedding_dim)  # Support sequences up to length 50

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, num_items)

    def forward(self, item_sequences, attention_mask=None):
        # Create position indices
        batch_size, seq_len = item_sequences.size()
        positions = torch.arange(seq_len, device=item_sequences.device).unsqueeze(0).repeat(batch_size, 1)

        # Get embeddings
        item_embedded = self.item_embedding(item_sequences)
        position_embedded = self.position_embedding(positions)

        # Combine embeddings
        embedded = item_embedded + position_embedded

        # Apply transformer encoder
        if attention_mask is not None:
            # Convert boolean mask to float
            attention_mask = attention_mask.float()
            # Convert to attention mask format expected by transformer
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))

        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=attention_mask)

        # Use the last position output for prediction
        last_output = transformer_output[:, -1, :]

        # Predict next item
        logits = self.fc(last_output)

        return logits
```

## Deep Learning for Content-Based Recommendations

Deep learning can also enhance content-based filtering by learning better representations of item features.

### Deep Content-Based Recommender

```python
class DeepContentRecommender:
    def __init__(self, item_features, user_item_matrix, embedding_dim=32, hidden_dims=[64, 32],
                 learning_rate=0.001, batch_size=64, num_epochs=20):
        self.item_features = item_features
        self.user_item_matrix = user_item_matrix

        # Create user and item mappings
        self.users = list(user_item_matrix.index)
        self.items = list(item_features.index)

        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.items)}

        # Model parameters
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Build the model
        self._build_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        # Define the model architecture
        input_dim = self.item_features.shape[1]

        # Item feature encoder
        item_encoder_layers = []
        current_dim = input_dim
        for hidden_dim in self.hidden_dims:
            item_encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim

        item_encoder_layers.append(nn.Linear(current_dim, self.embedding_dim))
        self.item_encoder = nn.Sequential(*item_encoder_layers)

        # User preference model
        self.user_embedding = nn.Embedding(len(self.users), self.embedding_dim)

        # Rating prediction layer
        self.rating_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], 1)
        )

        # Combine into a single model
        class ContentBasedModel(nn.Module):
            def __init__(self, item_encoder, user_embedding, rating_predictor):
                super(ContentBasedModel, self).__init__()
                self.item_encoder = item_encoder
                self.user_embedding = user_embedding
                self.rating_predictor = rating_predictor

            def forward(self, user_idx, item_features):
                # Encode item features
                item_embedding = self.item_encoder(item_features)

                # Get user embedding
                user_embedding = self.user_embedding(user_idx)

                # Concatenate and predict rating
                combined = torch.cat([user_embedding, item_embedding], dim=1)
                rating = self.rating_predictor(combined)

                return rating.squeeze()

        self.model = ContentBasedModel(self.item_encoder, self.user_embedding, self.rating_predictor)

    def _prepare_training_data(self):
        # Prepare training data from user-item matrix
        user_indices = []
        item_features_list = []
        ratings = []

        for user in self.users:
            user_idx = self.user_to_idx[user]
            user_ratings = self.user_item_matrix.loc[user]

            for item, rating in user_ratings.items():
                if rating > 0 and item in self.item_features.index:
                    user_indices.append(user_idx)
                    item_features_list.append(self.item_features.loc[item].values)
                    ratings.append(rating)

        return (
            torch.tensor(user_indices, dtype=torch.long),
            torch.tensor(item_features_list, dtype=torch.float),
            torch.tensor(ratings, dtype=torch.float)
        )

    def train(self):
        # Prepare training data
        user_indices, item_features, ratings = self._prepare_training_data()

        # Create dataset
        dataset = torch.utils.data.TensorDataset(user_indices, item_features, ratings)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_users, batch_items, batch_ratings in dataloader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch_users, batch_items)
                loss = self.criterion(predictions, batch_ratings)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

    def predict_rating(self, user_id, item_id):
        """Predict rating for a user-item pair"""
        if user_id not in self.user_to_idx or item_id not in self.item_features.index:
            return 0.0

        user_idx = self.user_to_idx[user_id]
        item_features = self.item_features.loc[item_id].
        item_features = self.item_features.loc[item_id].values

        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            item_tensor = torch.tensor([item_features], dtype=torch.float).to(self.device)
            prediction = self.model(user_tensor, item_tensor)

            return prediction.item()

    def recommend_items(self, user_id, n=5):
        """Recommend top N items for a user"""
        if user_id not in self.user_to_idx:
            return pd.DataFrame(columns=['item_id', 'predicted_rating'])

        # Get items the user hasn't rated yet
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = [item for item in self.items if item in user_ratings.index and user_ratings[item] == 0]

        if not unrated_items:
            return pd.DataFrame(columns=['item_id', 'predicted_rating'])

        # Predict ratings for unrated items
        predictions = []
        for item in unrated_items:
            predicted_rating = self.predict_rating(user_id, item)
            predictions.append({
                'item_id': item,
                'predicted_rating': predicted_rating
            })

        # Sort by predicted rating and return top N
        recommendations = pd.DataFrame(predictions)
        recommendations = recommendations.sort_values('predicted_rating', ascending=False).head(n)

        return recommendations
```

## Deep Learning for Hybrid Recommendations

We can combine multiple deep learning models to create powerful hybrid recommenders.

### Deep Hybrid Recommender

```python
class DeepHybridRecommender:
    def __init__(self, user_item_matrix, item_features, user_sequences=None):
        self.user_item_matrix = user_item_matrix
        self.item_features = item_features
        self.user_sequences = user_sequences

        # Create user and item mappings
        self.users = list(user_item_matrix.index)
        self.items = list(set(user_item_matrix.columns) | set(item_features.index))

        # Initialize component recommenders
        self.ncf = NCFRecommender(user_item_matrix)
        self.content = DeepContentRecommender(item_features, user_item_matrix)

        if user_sequences:
            self.sequential = SequentialRecommender(
                user_sequences,
                num_items=len(self.items)
            )

    def train(self):
        """Train all component recommenders"""
        print("Training NCF recommender...")
        self.ncf.train()

        print("\nTraining content-based recommender...")
        self.content.train()

        if hasattr(self, 'sequential'):
            print("\nTraining sequential recommender...")
            self.sequential.train()

    def predict_rating(self, user_id, item_id, weights=None):
        """Predict rating using a weighted combination of models"""
        if weights is None:
            weights = {'ncf': 0.6, 'content': 0.4, 'sequential': 0.0}

        predictions = {}

        # Get predictions from each model
        try:
            predictions['ncf'] = self.ncf.predict_rating(user_id, item_id)
        except:
            predictions['ncf'] = 0

        try:
            predictions['content'] = self.content.predict_rating(user_id, item_id)
        except:
            predictions['content'] = 0

        if hasattr(self, 'sequential') and 'sequential' in weights and weights['sequential'] > 0:
            try:
                # For sequential models, we need to get the score differently
                recommendations = self.sequential.recommend_next_items(user_id, n=100)
                if not recommendations.empty and item_id in recommendations['item_id'].values:
                    score = recommendations.loc[recommendations['item_id'] == item_id, 'score'].values[0]
                    # Convert score to rating scale
                    predictions['sequential'] = 1 + score * 4
                else:
                    predictions['sequential'] = 0
            except:
                predictions['sequential'] = 0

        # Calculate weighted prediction
        weighted_sum = 0
        weight_sum = 0

        for model, weight in weights.items():
            if model in predictions and weight > 0:
                weighted_sum += predictions[model] * weight
                weight_sum += weight

        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return 0

    def recommend_items(self, user_id, n=5, weights=None):
        """Recommend items using a weighted combination of models"""
        if weights is None:
            weights = {'ncf': 0.6, 'content': 0.4, 'sequential': 0.0}

        # Get recommendations from each model
        recommendations = {}

        try:
            recommendations['ncf'] = self.ncf.recommend_items(user_id, n=n*2)
        except:
            recommendations['ncf'] = pd.DataFrame(columns=['item_id', 'predicted_rating'])

        try:
            recommendations['content'] = self.content.recommend_items(user_id, n=n*2)
        except:
            recommendations['content'] = pd.DataFrame(columns=['item_id', 'predicted_rating'])

        if hasattr(self, 'sequential') and 'sequential' in weights and weights['sequential'] > 0:
            try:
                recommendations['sequential'] = self.sequential.recommend_next_items(user_id, n=n*2)
                # Rename score column to match other recommenders
                if not recommendations['sequential'].empty and 'score' in recommendations['sequential'].columns:
                    recommendations['sequential'] = recommendations['sequential'].rename(
                        columns={'score': 'predicted_rating'}
                    )
            except:
                recommendations['sequential'] = pd.DataFrame(columns=['item_id', 'predicted_rating'])

        # Combine recommendations
        all_items = set()
        for model, recs in recommendations.items():
            if not recs.empty:
                all_items.update(recs['item_id'].tolist())

        # Calculate weighted scores for all items
        combined_recommendations = []
        for item_id in all_items:
            weighted_score = 0
            weight_sum = 0

            for model, weight in weights.items():
                if model in recommendations and weight > 0:
                    recs = recommendations[model]
                    if not recs.empty and item_id in recs['item_id'].values:
                        score = recs.loc[recs['item_id'] == item_id, 'predicted_rating'].values[0]
                        weighted_score += score * weight
                        weight_sum += weight

            if weight_sum > 0:
                combined_recommendations.append({
                    'item_id': item_id,
                    'predicted_rating': weighted_score / weight_sum
                })

        # Sort and return top N
        result = pd.DataFrame(combined_recommendations)
        if not result.empty:
            result = result.sort_values('predicted_rating', ascending=False).head(n)

        return result
```

## Example Usage

Let's see how to use these deep learning recommenders with a sample dataset:

```python
# Create sample data
user_item_matrix = pd.DataFrame({
    '1': [5, 4, 0, 5, 0],
    '2': [3, 0, 4, 2, 0],
    '3': [4, 5, 3, 0, 4],
    '4': [0, 0, 5, 0, 5],
    '5': [0, 3, 0, 4, 3]
}, index=['user1', 'user2', 'user3', 'user4', 'user5'])

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

# Create user sequences
user_sequences = {
    'user1': [1, 3, 4],
    'user2': [2, 3, 1],
    'user3': [1, 2, 5],
    'user4': [3, 5],
    'user5': [2, 4, 5]
}

# Initialize and train NCF recommender
print("Training Neural Collaborative Filtering model...")
ncf = NCFRecommender(user_item_matrix, num_epochs=5)
ncf.train()

# Get recommendations for a user
user_id = 'user1'
recommendations = ncf.recommend_items(user_id)
print(f"\nNCF Recommendations for {user_id}:")
print(recommendations)

# Initialize and train deep content-based recommender
print("\nTraining Deep Content-Based model...")
content_recommender = DeepContentRecommender(item_features, user_item_matrix, num_epochs=5)
content_recommender.train()

# Get recommendations
recommendations = content_recommender.recommend_items(user_id)
print(f"\nDeep Content-Based Recommendations for {user_id}:")
print(recommendations)

# Initialize and train sequential recommender
print("\nTraining Sequential model...")
sequential_recommender = SequentialRecommender(user_sequences, num_items=6, num_epochs=5)
sequential_recommender.train()

# Get recommendations
recommendations = sequential_recommender.recommend_next_items(user_id)
print(f"\nSequential Recommendations for {user_id}:")
print(recommendations)

# Initialize and train hybrid recommender
print("\nTraining Deep Hybrid model...")
hybrid_recommender = DeepHybridRecommender(user_item_matrix, item_features, user_sequences)
hybrid_recommender.train()

# Get recommendations with different weightings
weights = {'ncf': 0.4, 'content': 0.3, 'sequential': 0.3}
recommendations = hybrid_recommender.recommend_items(user_id, weights=weights)
print(f"\nHybrid Recommendations for {user_id}:")
print(recommendations)
```

## Evaluating Deep Learning Recommenders

Let's implement a function to evaluate our deep learning recommenders:

```python
def evaluate_deep_recommenders(user_item_matrix, item_features, user_sequences, test_data):
    """
    Evaluate different deep learning recommenders

    Parameters:
    user_item_matrix (DataFrame): User-item rating matrix
    item_features (DataFrame): Item features
    user_sequences (dict): User interaction sequences
    test_data (list): List of (user_id, item_id, rating) tuples for testing

    Returns:
    DataFrame: Evaluation metrics for each recommender
    """
    # Initialize recommenders
    ncf = NCFRecommender(user_item_matrix, num_epochs=5)
    content = DeepContentRecommender(item_features, user_item_matrix, num_epochs=5)
    sequential = SequentialRecommender(user_sequences, num_items=len(item_features), num_epochs=5)
    hybrid = DeepHybridRecommender(user_item_matrix, item_features, user_sequences)

    # Train recommenders
    print("Training recommenders...")
    ncf.train()
    content.train()
    sequential.train()
    hybrid.train()

    # Define recommenders to evaluate
    recommenders = {
        'NCF': ncf,
        'DeepContent': content,
        'Sequential': sequential,
        'Hybrid': hybrid
    }

    # Initialize results dictionary
    results = {name: {'MAE': 0, 'RMSE': 0, 'Precision@5': 0, 'Recall@5': 0, 'F1@5': 0, 'NDCG@5': 0}
               for name in recommenders.keys()}

    # Evaluate rating prediction
    print("\nEvaluating rating prediction...")
    for name, recommender in recommenders.items():
        mae_sum = 0
        rmse_sum = 0
        count = 0

        for user_id, item_id, true_rating in test_data:
            try:
                if name == 'Sequential':
                    # For sequential models, we need to get the score differently
                    recommendations = recommender.recommend_next_items(user_id, n=100)
                    if not recommendations.empty and item_id in recommendations['item_id'].values:
                        score = recommendations.loc[recommendations['item_id'] == item_id, 'score'].values[0]
                        # Convert score to rating scale
                        predicted_rating = 1 + score * 4
                    else:
                        predicted_rating = 0
                elif name == 'Hybrid':
                    predicted_rating = recommender.predict_rating(user_id, item_id)
                else:
                    predicted_rating = recommender.predict_rating(user_id, item_id)

                error = true_rating - predicted_rating
                mae_sum += abs(error)
                rmse_sum += error ** 2
                count += 1

            except Exception as e:
                print(f"Error evaluating {name} for {user_id}, {item_id}: {e}")

        # Calculate average error metrics
        results[name]['MAE'] = mae_sum / count if count > 0 else float('inf')
        results[name]['RMSE'] = (rmse_sum / count) ** 0.5 if count > 0 else float('inf')

    # Evaluate recommendation metrics
    print("\nEvaluating recommendation quality...")

    # Group test data by user
    user_test_items = {}
    for user_id, item_id, rating in test_data:
        if user_id not in user_test_items:
            user_test_items[user_id] = []
        user_test_items[user_id].append((item_id, rating))

    for name, recommender in recommenders.items():
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
                if name == 'Sequential':
                    recommendations = recommender.recommend_next_items(user_id, n=5)
                    recommended_items = recommendations['item_id'].tolist() if not recommendations.empty else []
                elif name == 'Hybrid':
                    recommendations = recommender.recommend_items(user_id, n=5)
                    recommended_items = recommendations['item_id'].tolist() if not recommendations.empty else []
                else:
                    recommendations = recommender.recommend_items(user_id, n=5)
                    recommended_items = recommendations['item_id'].tolist() if not recommendations.empty else []

                # Calculate precision and recall
                hits = len(set(recommended_items) & set(relevant_items))
                precision = hits / len(recommended_items) if recommended_items else 0
                recall = hits / len(relevant_items) if relevant_items else 0

                # Calculate F1 score
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                # Calculate NDCG
                dcg = 0
                idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_items), 5)))

                for i, item in enumerate(recommended_items):
                    if item in relevant_items:
                        dcg += 1 / np.log2(i + 2)

                ndcg = dcg / idcg if idcg > 0 else 0

                # Update sums
                precision_sum += precision
                recall_sum += recall
                ndcg_sum += ndcg
                user_count += 1

                # Update results
                results[name]['Precision@5'] = precision_sum / user_count if user_count > 0 else 0
                results[name]['Recall@5'] = recall_sum / user_count if user_count > 0 else 0
                results[name]['F1@5'] = 2 * results[name]['Precision@5'] * results[name]['Recall@5'] / (results[name]['Precision@5'] + results[name]['Recall@5']) if (results[name]['Precision@5'] + results[name]['Recall@5']) > 0 else 0
                results[name]['NDCG@5'] = ndcg_sum / user_count if user_count > 0 else 0

            except Exception as e:
                print(f"Error evaluating recommendations for {name}, user {user_id}: {e}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T

    return results_df

# Example usage of the evaluation function
def create_test_data(user_item_matrix, test_ratio=0.2):
    """Create test data by hiding some ratings"""
    test_data = []
    train_matrix = user_item_matrix.copy()

    for user in user_item_matrix.index:
        user_ratings = user_item_matrix.loc[user]
        rated_items = user_ratings[user_ratings > 0].index.tolist()

        # Select random items for testing
        num_test = max(1, int(len(rated_items) * test_ratio))
        test_items = np.random.choice(rated_items, num_test, replace=False)

        for item in test_items:
            # Add to test data
            test_data.append((user, item, user_item_matrix.loc[user, item]))
            # Hide from training data
            train_matrix.loc[user, item] = 0

    return train_matrix, test_data

# Create train/test split
train_matrix, test_data = create_test_data(user_item_matrix)

# Evaluate recommenders
results = evaluate_deep_recommenders(train_matrix, item_features, user_sequences, test_data)
print("\nEvaluation Results:")
print(results)
```

## Conclusion

In this lesson, we've explored how deep learning can enhance recommendation systems:

1. **Neural Collaborative Filtering (NCF)** extends matrix factorization by using neural networks to model complex user-item interactions.

2. **Sequence-based recommenders** like RNNs and Transformers capture temporal patterns in user behavior, providing context-aware recommendations.

3. **Deep content-based recommenders** learn better representations of item features, improving recommendations for new items.

4. **Hybrid approaches** combine multiple deep learning models to leverage their complementary strengths.

Deep learning recommenders offer several advantages:

- They can automatically learn feature representations without manual engineering
- They capture non-linear relationships between users and items
- They can integrate multiple data sources and modalities
- They adapt to evolving user preferences through sequential modeling

However, they also come with challenges:

- They require more data to train effectively
- They have higher computational requirements
- They are less interpretable than traditional methods
- They need careful tuning of hyperparameters

As you implement these models, remember to:

- Start with simpler models and gradually increase complexity
- Use appropriate evaluation metrics for your specific use case
- Consider the trade-off between model complexity and performance
- Combine different approaches in hybrid systems for better results

In the next lesson, we'll explore how to deploy recommendation systems in production environments and address challenges like scalability, real-time updates, and monitoring.
