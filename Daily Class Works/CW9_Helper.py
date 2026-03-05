"""
recommender_utils.py
====================
Helper functions for Matrix Completion Learning Tool
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_movielens_small():
    """
    Load a small subset of MovieLens data for easy visualization
    
    Returns:
        R: ratings matrix (users x movies)
        movie_titles: list of movie titles
        sparsity: percentage of missing entries
    """
    # Load MovieLens 100K dataset
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'
    names = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(url, sep='\t', names=names)
    
    # Load movie titles
    movie_url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.item'
    movie_cols = ['movie_id', 'title'] + ['col_' + str(i) for i in range(2, 24)]
    movies_df = pd.read_csv(movie_url, sep='|', names=movie_cols, 
                            encoding='latin-1', usecols=[0, 1])
    
    # Select subset: top 50 active users and 30 popular movies
    user_counts = ratings_df['user_id'].value_counts()
    movie_counts = ratings_df['movie_id'].value_counts()
    
    top_users = user_counts.head(50).index
    top_movies = movie_counts.head(30).index
    
    # Filter data
    subset = ratings_df[ratings_df['user_id'].isin(top_users) & 
                        ratings_df['movie_id'].isin(top_movies)]
    
    # Create ratings matrix
    n_users = len(top_users)
    n_movies = len(top_movies)
    R = np.full((n_users, n_movies), np.nan)
    
    user_map = {uid: i for i, uid in enumerate(top_users)}
    movie_map = {mid: i for i, mid in enumerate(top_movies)}
    
    for _, row in subset.iterrows():
        u_idx = user_map[row['user_id']]
        m_idx = movie_map[row['movie_id']]
        R[u_idx, m_idx] = row['rating']
    
    # Get movie titles
    movie_titles = []
    for mid in top_movies:
        title = movies_df[movies_df['movie_id'] == mid]['title'].values[0]
        movie_titles.append(title[:30])  # Truncate long titles
    
    sparsity = np.isnan(R).sum() / R.size * 100
    
    return R, movie_titles, sparsity


def plot_ratings_matrix(R, movie_titles, title="Ratings Matrix"):
    """Plot the ratings matrix as a heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(R, cmap='YlOrRd', aspect='auto', vmin=1, vmax=5)
    ax.set_xlabel('Movies', fontsize=12, fontweight='bold')
    ax.set_ylabel('Users', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add movie titles on x-axis
    ax.set_xticks(range(len(movie_titles)))
    ax.set_xticklabels(movie_titles, rotation=90, ha='right', fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rating (1-5 stars)', fontsize=11)
    
    plt.tight_layout()
    plt.show()


def plot_data_exploration(R):
    """Plot basic statistics about the data"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Rating distribution
    observed_ratings = R[~np.isnan(R)]
    axes[0].hist(observed_ratings, bins=5, range=(0.5, 5.5), 
                 color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Rating', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Rating Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Ratings per user
    ratings_per_user = np.sum(~np.isnan(R), axis=1)
    axes[1].hist(ratings_per_user, bins=15, color='coral', 
                 edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Number of Ratings', fontsize=11)
    axes[1].set_ylabel('Number of Users', fontsize=11)
    axes[1].set_title('Ratings per User', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Ratings per movie
    ratings_per_movie = np.sum(~np.isnan(R), axis=0)
    axes[2].hist(ratings_per_movie, bins=15, color='lightgreen', 
                 edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Number of Ratings', fontsize=11)
    axes[2].set_ylabel('Number of Movies', fontsize=11)
    axes[2].set_title('Ratings per Movie', fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def simple_matrix_completion(R, rank=5):
    """
    Simple matrix completion using SVD
    
    Steps:
    1. Fill missing values with column means
    2. Apply SVD
    3. Keep only top 'rank' components
    4. Reconstruct matrix
    """
    # Make a copy
    R_filled = R.copy()
    
    # Step 1: Fill missing with column means
    col_means = np.nanmean(R, axis=0)
    for j in range(R.shape[1]):
        R_filled[np.isnan(R[:, j]), j] = col_means[j]
    
    # Step 2: Apply SVD
    U, s, Vt = np.linalg.svd(R_filled, full_matrices=False)
    
    # Step 3: Keep only top 'rank' components
    U_r = U[:, :rank]
    s_r = s[:rank]
    Vt_r = Vt[:rank, :]
    
    # Step 4: Reconstruct
    R_completed = U_r @ np.diag(s_r) @ Vt_r
    
    # Clip to valid rating range
    R_completed = np.clip(R_completed, 1, 5)
    
    return R_completed


def evaluate_predictions(R_true, R_pred, mask):
    """
    Calculate prediction error only on observed entries
    
    Args:
        R_true: original ratings
        R_pred: predicted ratings
        mask: boolean array (True = observed entry)
    """
    observed_true = R_true[mask]
    observed_pred = R_pred[mask]
    
    mae = np.mean(np.abs(observed_true - observed_pred))
    rmse = np.sqrt(np.mean((observed_true - observed_pred)**2))
    
    return mae, rmse


def plot_svd_analysis(R):
    """Analyze the singular values to understand rank"""
    # Fill NaN with column means for SVD
    R_filled = R.copy()
    col_means = np.nanmean(R, axis=0)
    for j in range(R.shape[1]):
        R_filled[np.isnan(R[:, j]), j] = col_means[j]
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(R_filled, full_matrices=False)
    
    # Plot singular values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Singular values
    axes[0].plot(range(1, len(s)+1), s, 'o-', linewidth=2, 
                 markersize=6, color='steelblue')
    axes[0].set_xlabel('Component', fontsize=11)
    axes[0].set_ylabel('Singular Value', fontsize=11)
    axes[0].set_title('Singular Values', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Cumulative energy
    energy = np.cumsum(s**2) / np.sum(s**2)
    axes[1].plot(range(1, len(energy)+1), energy, 'o-', 
                 linewidth=2, markersize=6, color='coral')
    axes[1].axhline(y=0.9, color='green', linestyle='--', 
                    linewidth=2, label='90% energy')
    axes[1].set_xlabel('Number of Components', fontsize=11)
    axes[1].set_ylabel('Cumulative Energy', fontsize=11)
    axes[1].set_title('Cumulative Energy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return s


def plot_comparison(R_original, R_completed, movie_titles):
    """Compare original and completed matrices side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original
    im1 = axes[0].imshow(R_original, cmap='YlOrRd', aspect='auto', vmin=1, vmax=5)
    axes[0].set_xlabel('Movies', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Users', fontsize=12, fontweight='bold')
    axes[0].set_title('Original Ratings (with missing values)', 
                      fontsize=13, fontweight='bold')
    axes[0].set_xticks(range(len(movie_titles)))
    axes[0].set_xticklabels(movie_titles, rotation=90, ha='right', fontsize=8)
    plt.colorbar(im1, ax=axes[0], label='Rating')
    
    # Completed
    im2 = axes[1].imshow(R_completed, cmap='YlOrRd', aspect='auto', vmin=1, vmax=5)
    axes[1].set_xlabel('Movies', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Users', fontsize=12, fontweight='bold')
    axes[1].set_title('Completed Ratings (all values filled)', 
                      fontsize=13, fontweight='bold')
    axes[1].set_xticks(range(len(movie_titles)))
    axes[1].set_xticklabels(movie_titles, rotation=90, ha='right', fontsize=8)
    plt.colorbar(im2, ax=axes[1], label='Rating')
    
    plt.tight_layout()
    plt.show()


def plot_rank_experiment(R, ranks):
    """Test different ranks and plot error"""
    mask = ~np.isnan(R)
    maes = []
    rmses = []
    
    for rank in ranks:
        R_pred = simple_matrix_completion(R, rank=rank)
        mae, rmse = evaluate_predictions(R, R_pred, mask)
        maes.append(mae)
        rmses.append(rmse)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(ranks, maes, 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[0].set_xlabel('Rank', fontsize=12)
    axes[0].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[0].set_title('Effect of Rank on Prediction Error', 
                      fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(ranks, rmses, 'o-', linewidth=2, markersize=8, color='coral')
    axes[1].set_xlabel('Rank', fontsize=12)
    axes[1].set_ylabel('Root Mean Squared Error', fontsize=12)
    axes[1].set_title('RMSE vs Rank', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return maes, rmses