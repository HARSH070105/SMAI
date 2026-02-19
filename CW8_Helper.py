"""
pca_utils.py
============
Helper functions for PCA learning tool
Contains all visualization and data loading functions
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people


def load_eigenfaces_data():
    """
    Load and return the eigenfaces dataset
    
    Returns:
        faces: sklearn dataset object
        X: feature matrix (n_samples, n_features)
        y: target labels
        target_names: names of people
        h, w: height and width of images
    """
    faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = faces.data
    y = faces.target
    target_names = faces.target_names
    h, w = faces.images.shape[1:]
    
    return faces, X, y, target_names, h, w


def print_dataset_info(X, target_names, h, w):
    """Print basic information about the dataset"""
    n_samples, n_features = X.shape
    n_classes = len(target_names)
    
    print(f"Number of samples: {n_samples}")
    print(f"Number of features (pixels): {n_features}")
    print(f"Image dimensions: {h}x{w}")
    print(f"Number of classes (people): {n_classes}")
    print(f"Classes: {target_names}")


def plot_sample_faces(faces, y, target_names, n_samples=10):
    """
    Plot sample faces from the dataset
    
    Args:
        faces: sklearn dataset object
        y: target labels
        target_names: names of people
        n_samples: number of samples to display
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Sample Faces from Dataset', fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(faces.images[i], cmap='gray')
            ax.set_title(target_names[y[i]], fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_eigenfaces(eigenvectors, h, w, n_components=10):
    """
    Visualize the top eigenfaces (principal components)
    
    Args:
        eigenvectors: matrix of eigenvectors
        h, w: height and width of images
        n_components: number of eigenfaces to display
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Top 10 Eigenfaces (Principal Components)', 
                 fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < n_components:
            eigenface = eigenvectors[:, i].reshape((h, w))
            ax.imshow(eigenface, cmap='gray')
            ax.set_title(f'PC {i+1}', fontsize=11)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_variance_explained(eigenvalues):
    """
    Plot explained variance ratio and cumulative variance
    
    Args:
        eigenvalues: array of eigenvalues
    """
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    ax1.bar(range(1, 21), explained_variance_ratio[:20], 
            alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Variance Explained by Each Component', 
                  fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, 101), cumulative_variance[:100], 
             marker='o', markersize=3, linewidth=2, color='coral')
    ax2.axhline(y=0.90, color='green', linestyle='--', label='90% variance')
    ax2.axhline(y=0.95, color='red', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', 
                  fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return cumulative_variance


def plot_reconstruction_comparison(X, X_centered, X_mean, eigenvectors, 
                                   cumulative_variance, h, w, sample_idx=0):
    """
    Show face reconstruction with different numbers of components
    
    Args:
        X: original data
        X_centered: centered data
        X_mean: mean of data
        eigenvectors: principal components
        cumulative_variance: cumulative explained variance
        h, w: image dimensions
        sample_idx: which sample to reconstruct
    """
    sample_face = X[sample_idx]
    n_components_list = [10, 50, 100, 200]
    
    fig, axes = plt.subplots(1, len(n_components_list) + 1, figsize=(15, 3))
    fig.suptitle('Face Reconstruction with Different Numbers of Components', 
                 fontsize=14, fontweight='bold')
    
    # Original
    axes[0].imshow(sample_face.reshape((h, w)), cmap='gray')
    axes[0].set_title('Original', fontsize=11)
    axes[0].axis('off')
    
    # Reconstructions
    for i, n_comp in enumerate(n_components_list):
        projection = np.dot(X_centered[sample_idx], eigenvectors[:, :n_comp])
        reconstruction = np.dot(projection, eigenvectors[:, :n_comp].T) + X_mean
        
        axes[i+1].imshow(reconstruction.reshape((h, w)), cmap='gray')
        var_explained = cumulative_variance[n_comp-1] * 100
        axes[i+1].set_title(f'{n_comp} components\n({var_explained:.1f}% var)', 
                           fontsize=10)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_sklearn_comparison(X, X_reconstructed, h, w, n_samples=3):
    """
    Compare original faces with sklearn PCA reconstructions
    
    Args:
        X: original data
        X_reconstructed: reconstructed data from PCA
        h, w: image dimensions
        n_samples: number of samples to compare
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(12, 8))
    fig.suptitle('Comparison: Original vs Reconstructed Faces', 
                 fontsize=14, fontweight='bold')
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(X[i].reshape((h, w)), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}', fontsize=11)
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(X_reconstructed[i].reshape((h, w)), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}', fontsize=11)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)