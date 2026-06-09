import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ─── MODEL DEFINITIONS (must match training) ──────────────────────────────────
class GMF(nn.Module):
    def __init__(self, n_u, n_i, dim=128, global_mean=3.5376):
        super().__init__()
        self.u_emb  = nn.Embedding(n_u, dim)
        self.i_emb  = nn.Embedding(n_i, dim)
        self.u_bias = nn.Embedding(n_u, 1)
        self.i_bias = nn.Embedding(n_i, 1)
        self.mu     = nn.Parameter(torch.tensor([global_mean]), requires_grad=False)
    def forward(self, u, i):
        dot = (self.u_emb(u) * self.i_emb(i)).sum(1)
        return dot + self.u_bias(u).squeeze() + self.i_bias(i).squeeze() + self.mu

class DeepNCF(nn.Module):
    def __init__(self, n_u, n_i, dim=64, global_mean=3.5376):
        super().__init__()
        self.u_emb = nn.Embedding(n_u, dim)
        self.i_emb = nn.Embedding(n_i, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim*2, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.mu = global_mean
    def forward(self, u, i):
        x = torch.cat([self.u_emb(u), self.i_emb(i)], dim=1)
        return self.mlp(x).squeeze() + self.mu

# ─── LOAD DATA & BUILD ID MAPS ────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet('netflix_ratings.parquet')

# Rebuild the same ID mappings used during training
user_cat  = pd.Categorical(pd.read_csv('netflix_filtered_ratings.csv', usecols=['UserID'])['UserID'])
movie_cat = pd.Categorical(pd.read_csv('netflix_filtered_ratings.csv', usecols=['MovieID'])['MovieID'])

user_to_idx  = dict(zip(user_cat.categories, range(len(user_cat.categories))))
movie_to_idx = dict(zip(movie_cat.categories, range(len(movie_cat.categories))))
idx_to_movie = {v: k for k, v in movie_to_idx.items()}

num_users  = len(user_to_idx)
num_movies = len(movie_to_idx)
global_mean = float(df['rating'].mean())

print(f"Users: {num_users:,} | Movies: {num_movies:,} | Global Mean: {global_mean:.4f}")

# ─── LOAD BOTH MODELS ─────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

gmf_model = GMF(num_users, num_movies, dim=128, global_mean=global_mean).to(device)
gmf_model.load_state_dict(torch.load('best_GMF_(Matrix_Factorization).pt', map_location=device))
gmf_model.eval()
print("GMF model loaded  (RMSE: 0.7717)")

ncf_model = DeepNCF(num_users, num_movies, dim=64, global_mean=global_mean).to(device)
ncf_model.load_state_dict(torch.load('best_Deep_NCF_(MLP).pt', map_location=device))
ncf_model.eval()
print("NCF model loaded  (RMSE: 0.8123)")

# ─── RECOMMENDATION FUNCTION ──────────────────────────────────────────────────
def recommend(user_id, model, model_name, top_k=10):
    """Generate Top-K recommendations for a given UserID."""
    if user_id not in user_to_idx:
        print(f"UserID {user_id} not found in dataset.")
        return

    u_idx = user_to_idx[user_id]

    # Movies this user has already rated
    rated_movies = set(df[df['user_idx'] == u_idx]['movie_idx'].tolist())
    unseen_idxs  = [i for i in range(num_movies) if i not in rated_movies]

    # Predict ratings for all unseen movies in one batch
    u_tensor = torch.tensor([u_idx] * len(unseen_idxs), dtype=torch.long).to(device)
    m_tensor = torch.tensor(unseen_idxs, dtype=torch.long).to(device)

    with torch.no_grad():
        preds = model(u_tensor, m_tensor).cpu().numpy()

    # Clamp predictions to valid rating range
    preds = np.clip(preds, 1.0, 5.0)

    top_indices = np.argsort(preds)[::-1][:top_k]

    print(f"\n{'='*55}")
    print(f"  [{model_name}] Top-{top_k} Recommendations")
    print(f"  For UserID: {user_id}  |  Already rated: {len(rated_movies)} movies")
    print(f"{'='*55}")
    print(f"  {'Rank':<5} {'MovieID':<12} {'Predicted Rating':>16}")
    print(f"  {'-'*40}")
    for rank, idx in enumerate(top_indices, 1):
        movie_id    = idx_to_movie[unseen_idxs[idx]]
        pred_rating = preds[idx]
        bar = '*' * int(pred_rating * 4)
        print(f"  {rank:<5} {movie_id:<12} {pred_rating:>8.2f} / 5.0   {bar}")
    print(f"{'='*55}")

def recommend_ensemble(user_id, top_k=10, gmf_weight=0.6, ncf_weight=0.4):
    """Blend GMF + NCF predictions for a more robust recommendation."""
    if user_id not in user_to_idx:
        print(f"UserID {user_id} not found in dataset.")
        return

    u_idx = user_to_idx[user_id]
    rated_movies = set(df[df['user_idx'] == u_idx]['movie_idx'].tolist())
    unseen_idxs  = [i for i in range(num_movies) if i not in rated_movies]

    u_tensor = torch.tensor([u_idx] * len(unseen_idxs), dtype=torch.long).to(device)
    m_tensor = torch.tensor(unseen_idxs, dtype=torch.long).to(device)

    with torch.no_grad():
        gmf_preds = gmf_model(u_tensor, m_tensor).cpu().numpy()
        ncf_preds = ncf_model(u_tensor, m_tensor).cpu().numpy()

    blended = gmf_weight * gmf_preds + ncf_weight * ncf_preds
    blended = np.clip(blended, 1.0, 5.0)

    top_indices = np.argsort(blended)[::-1][:top_k]

    print(f"\n{'='*60}")
    print(f"  [ENSEMBLE GMF+NCF] Top-{top_k} Recommendations")
    print(f"  For UserID: {user_id}  (GMF {int(gmf_weight*100)}% + NCF {int(ncf_weight*100)}%)")
    print(f"{'='*60}")
    print(f"  {'Rank':<5} {'MovieID':<12} {'GMF':>6} {'NCF':>6} {'Blended':>8}")
    print(f"  {'-'*45}")
    for rank, idx in enumerate(top_indices, 1):
        movie_id = idx_to_movie[unseen_idxs[idx]]
        g = np.clip(gmf_preds[idx], 1, 5)
        n = np.clip(ncf_preds[idx], 1, 5)
        b = blended[idx]
        print(f"  {rank:<5} {movie_id:<12} {g:>6.2f} {n:>6.2f} {b:>8.2f}/5.0")
    print(f"{'='*60}")

# ─── DEMO: RUN RECOMMENDATIONS ────────────────────────────────────────────────
# Change this to any UserID from your dataset
DEMO_USER = 712664

recommend(DEMO_USER, gmf_model, "GMF  (RMSE 0.7717)")
recommend(DEMO_USER, ncf_model, "Deep NCF  (RMSE 0.8123)")
recommend_ensemble(DEMO_USER)

print("\nDone! To recommend for a different user, change DEMO_USER at the bottom of this file.")
