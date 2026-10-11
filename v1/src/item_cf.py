# item_cf.py

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# 1. Data filtering functions
# -----------------------------
def filter_users_items(df, user_col='user_id', item_col='book_title', rating_col='book_rating',
                       min_user_ratings=10, min_item_ratings=10):
    """
    Iteratively filter users and items based on minimum ratings threshold.
    """
    filtered_df = df[[user_col, item_col, rating_col]].copy()
    while True:
        user_counts = filtered_df.groupby(user_col)[rating_col].count()
        item_counts = filtered_df.groupby(item_col)[user_col].count()

        active_users = user_counts[user_counts >= min_user_ratings].index
        active_items = item_counts[item_counts >= min_item_ratings].index

        new_df = filtered_df[
            filtered_df[user_col].isin(active_users) &
            filtered_df[item_col].isin(active_items)
        ]

        if len(new_df) == len(filtered_df):
            break
        filtered_df = new_df.copy()
    return filtered_df


# -----------------------------
# 2. Train-test split
# -----------------------------
def train_test_split_by_user(df, test_frac=0.2, random_seed=42, user_col='user_id'):
    """
    Split dataset into train and test sets per user.
    """
    rng = np.random.RandomState(random_seed)
    train_idxs, test_idxs = [], []

    for user, group in df.groupby(user_col):
        idxs = group.index.tolist()
        rng.shuffle(idxs)
        n_test = max(1, int(round(test_frac * len(idxs))))
        test_for_user = idxs[:n_test]
        train_for_user = idxs[n_test:]

        if len(train_for_user) == 0 and len(test_for_user) > 1:
            train_for_user.append(test_for_user.pop())

        train_idxs.extend(train_for_user)
        test_idxs.extend(test_for_user)

    train_df = df.loc[train_idxs].reset_index(drop=True)
    test_df = df.loc[test_idxs].reset_index(drop=True)
    return train_df, test_df


# -----------------------------
# 3. Label encoders
# -----------------------------
class CFEncoders:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def fit(self, train_df, user_col='user_id', item_col='book_title'):
        train_df['user_idx'] = self.user_encoder.fit_transform(train_df[user_col])
        train_df['item_idx'] = self.item_encoder.fit_transform(train_df[item_col])
        return train_df

    def transform(self, df, user_col='user_id', item_col='book_title'):
        df['user_idx'] = self.user_encoder.transform(df[user_col])
        df['item_idx'] = self.item_encoder.transform(df[item_col])
        return df


# -----------------------------
# 4. Item-user matrix
# -----------------------------
def build_item_user_matrix(df, rating_col='book_rating', num_items=None, num_users=None):
    """
    Build a CSR matrix of items x users.
    """
    matrix = csr_matrix(
        (df[rating_col], (df['item_idx'], df['user_idx'])),
        shape=(num_items, num_users)
    )
    return matrix


# -----------------------------
# 5. Item similarity
# -----------------------------
def compute_item_similarity(item_user_matrix):
    """
    Compute cosine similarity between items.
    """
    sim_matrix = cosine_similarity(item_user_matrix)
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix


# -----------------------------
# 6. Top-K items
# -----------------------------
def get_top_k_items(item_sim_matrix, k=10):
    """
    For each item, return indices of top-k similar items.
    """
    return np.argsort(-item_sim_matrix, axis=1)[:, :k]


# -----------------------------
# 7. Prediction
# -----------------------------
def predict_ratings(item_user_matrix, item_sim_matrix, top_k_items):
    """
    Predict ratings using item-based CF with top-k neighbors.
    """
    num_items, num_users = item_user_matrix.shape
    item_means = np.array(item_user_matrix.mean(axis=1)).flatten()
    pred_matrix = np.zeros((num_users, num_items))

    for i in range(num_items):
        neighbors = top_k_items[i]
        sim_scores = item_sim_matrix[i, neighbors]
        neighbor_ratings = item_user_matrix[neighbors, :].toarray()
        neighbor_means = item_means[neighbors][:, np.newaxis]
        mean_centered = neighbor_ratings - neighbor_means
        mask = neighbor_ratings != 0
        weighted_sum = (sim_scores[:, np.newaxis] * mean_centered) * mask
        sim_sum_per_user = np.sum(np.abs(sim_scores[:, np.newaxis] * mask), axis=0)
        pred = np.where(sim_sum_per_user != 0, weighted_sum.sum(axis=0) / sim_sum_per_user, 0)
        pred_matrix[:, i] = item_means[i] + pred

    return np.nan_to_num(pred_matrix)


# -----------------------------
# 8. Evaluation
# -----------------------------
def evaluate_predictions(test_df, pred_matrix, rating_col='book_rating'):
    """
    Compute MAE, MSE, RMSE
    """
    users = test_df['user_idx'].to_numpy()
    items = test_df['item_idx'].to_numpy()
    y_true = test_df[rating_col].to_numpy()
    y_pred = pred_matrix[users, items]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}


# -----------------------------
# 9. Recommendation functions
# -----------------------------
def recommend_for_user(user_id, user_encoder, item_encoder, pred_matrix, n=10):
    user_idx = user_encoder.transform([user_id])[0]
    preds = pred_matrix[user_idx]
    top_idx = np.argsort(-preds)[:n]
    items = item_encoder.inverse_transform(top_idx)
    scores = preds[top_idx]
    return list(zip(items, scores))


def recommend_similar_items(item_title, item_encoder, item_sim_matrix, k=10):
    item_idx = item_encoder.transform([item_title])[0]
    sims = item_sim_matrix[item_idx]
    top_idx = np.argsort(-sims)[:k+1]
    top_idx = [i for i in top_idx if i != item_idx][:k]
    return item_encoder.inverse_transform(top_idx)

def recommend_similar_items_new(item_title, item_encoder, item_sim_matrix, k=10):
    # Get item index
    item_idx = item_encoder.transform([item_title])[0]

    # Get similarity row as dense array
    sims = item_sim_matrix[item_idx].toarray().flatten()
    sims[item_idx] = -1e9

    # Efficient top-k
    if k < len(sims):
        top_idx = np.argpartition(-sims, k)[:k]  # get k largest
        top_idx = top_idx[np.argsort(-sims[top_idx])]  # sort top-k properly
    else:
        top_idx = np.argsort(-sims)[::-1]

    # Return titles
    return item_encoder.inverse_transform(top_idx)