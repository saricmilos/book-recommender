from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error

class UserBasedCF:
    def __init__(self, min_user_ratings=10, min_book_ratings=10, k_neighbors=10, random_seed=42):
        self.min_user_ratings = min_user_ratings
        self.min_book_ratings = min_book_ratings
        self.k_neighbors = k_neighbors
        self.random_seed = random_seed
        
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        self.train_df = None
        self.test_df = None
        self.train_matrix = None
        self.user_sim_matrix = None
        self.pred_matrix = None
        self.user_means = None

    # -----------------------------
    # 1. Filter dataframe
    # -----------------------------
    def filter_dataframe(self, df: pd.DataFrame):
        """Filter dataframe to remove users/books with few ratings."""
        cf_df = df[['user_id', 'book_title', 'book_rating']].copy()
        cf_df_filtered = cf_df.copy()

        while True:
            user_counts = cf_df_filtered.groupby('user_id')['book_rating'].count()
            book_counts = cf_df_filtered.groupby('book_title')['user_id'].count()

            active_users = user_counts[user_counts >= self.min_user_ratings].index
            active_books = book_counts[book_counts >= self.min_book_ratings].index

            filtered = cf_df_filtered[
                cf_df_filtered['user_id'].isin(active_users) &
                cf_df_filtered['book_title'].isin(active_books)
            ]

            if len(filtered) == len(cf_df_filtered):
                break
            cf_df_filtered = filtered.copy()

        print("Filtered dataframe shape:", cf_df_filtered.shape)
        return cf_df_filtered

    # -----------------------------
    # 2. Train-test split
    # -----------------------------
    def train_test_split(self, df: pd.DataFrame, test_frac=0.2):
        """Split dataframe into train/test while keeping each user in train."""
        rng = np.random.RandomState(self.random_seed)
        train_idxs, test_idxs = [], []

        for user, group in df.groupby('user_id'):
            idxs = group.index.tolist()
            rng.shuffle(idxs)
            n_test = max(1, int(round(test_frac * len(idxs))))
            test_for_user = idxs[:n_test]
            train_for_user = idxs[n_test:]

            if len(train_for_user) == 0 and len(test_for_user) > 1:
                train_for_user.append(test_for_user.pop())

            train_idxs.extend(train_for_user)
            test_idxs.extend(test_for_user)

        self.train_df = df.loc[train_idxs].reset_index(drop=True)
        self.test_df = df.loc[test_idxs].reset_index(drop=True)

        # Encode users and items
        self.train_df['user_idx'] = self.user_encoder.fit_transform(self.train_df['user_id'])
        self.train_df['book_idx'] = self.item_encoder.fit_transform(self.train_df['book_title'])
        self.test_df['user_idx'] = self.user_encoder.transform(self.test_df['user_id'])
        self.test_df['book_idx'] = self.item_encoder.transform(self.test_df['book_title'])

    # -----------------------------
    # 3. Build train matrix
    # -----------------------------
    def build_train_matrix(self):
        """Build sparse user-item matrix from training data."""
        num_users = self.train_df['user_idx'].nunique()
        num_items = self.train_df['book_idx'].nunique()
        self.train_matrix = csr_matrix(
            (self.train_df['book_rating'], 
             (self.train_df['user_idx'], self.train_df['book_idx'])),
            shape=(num_users, num_items)
        )
        return self.train_matrix

    # -----------------------------
    # 4. Compute user similarity
    # -----------------------------
    def compute_user_similarity(self):
        """Compute cosine similarity between users."""
        self.user_sim_matrix = cosine_similarity(self.train_matrix)
        np.fill_diagonal(self.user_sim_matrix, 0)

    # -----------------------------
    # 5. Predict ratings
    # -----------------------------
    def predict_ratings(self):
        """Predict ratings using top-k similar users."""
        num_users, num_items = self.train_matrix.shape
        self.user_means = np.array(self.train_matrix.mean(axis=1)).flatten()
        self.pred_matrix = np.zeros((num_users, num_items))
        top_k_users = np.argsort(-self.user_sim_matrix, axis=1)[:, :self.k_neighbors]

        for u in range(num_users):
            neighbors = top_k_users[u]
            sim_scores = self.user_sim_matrix[u, neighbors]
            neighbor_ratings = self.train_matrix[neighbors, :].toarray()
            neighbor_means = self.user_means[neighbors][:, np.newaxis]
            mean_centered = neighbor_ratings - neighbor_means
            weighted_sum = sim_scores @ mean_centered
            sim_sum = np.sum(np.abs(sim_scores))
            self.pred_matrix[u, :] = self.user_means[u] + (weighted_sum / sim_sum if sim_sum != 0 else 0)

        self.pred_matrix = np.nan_to_num(self.pred_matrix)

    # -----------------------------
    # 6. Recommend books for user
    # -----------------------------
    def recommend_books_by_user(self, user_id, top_k=10):
        """Return top-k book recommendations for a given user."""
        user_idx = self.user_encoder.transform([user_id])[0]
        sims = self.user_sim_matrix[user_idx]
        top_k_idx = np.argsort(-sims)[:top_k+1]
        top_k_idx = [i for i in top_k_idx if i != user_idx][:top_k]
        user_ratings = self.train_matrix[top_k_idx].toarray().mean(axis=0)
        already_rated = self.train_matrix[user_idx].toarray().flatten() > 0
        user_ratings[already_rated] = -np.inf
        top_books_idx = np.argsort(-user_ratings)[:top_k]
        return self.item_encoder.inverse_transform(top_books_idx)

    # -----------------------------
    # 7. Evaluate metrics
    # -----------------------------
    def evaluate_rmse_mae(self):
        """Evaluate prediction matrix on test set."""
        users = self.test_df['user_idx'].to_numpy()
        items = self.test_df['book_idx'].to_numpy()
        y_true = self.test_df['book_rating'].to_numpy()
        y_pred = self.pred_matrix[users, items]

        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

    # -----------------------------
    # 8. Precision@N and Recall@N
    # -----------------------------
    def precision_recall_at_n(self, N=10, threshold=4.0):
        """Compute Precision@N and Recall@N."""
        train_items = self.train_df.groupby('user_idx')['book_idx'].apply(set).to_dict()
        test_items  = (
            self.test_df[self.test_df['book_rating'] >= threshold]
            .groupby('user_idx')['book_idx']
            .apply(set)
            .to_dict()
        )

        precisions, recalls = [], []
        for user, true_items in test_items.items():
            if not true_items:
                continue
            scores = self.pred_matrix[user].copy()
            scores[list(train_items.get(user, []))] = -np.inf
            top_n_items = np.argsort(-scores)[:N]
            hits = len(set(top_n_items) & true_items)
            precisions.append(hits / N)
            recalls.append(hits / len(true_items))

        return np.mean(precisions) if precisions else 0.0, np.mean(recalls) if recalls else 0.0
