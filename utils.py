import numpy as np
import plotly.graph_objects as go

def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance matrix for a dataset X.
    
    Parameters:
    - X: 2D NumPy array of shape (n, d)
    
    Returns:
    - D: 2D NumPy array of shape (n, n), where D[i,j] = ||X[i] - X[j]||
    """
    return np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)



def k_nearest_neighbors(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the indices of the k nearest neighbors for each point.
    
    Parameters:
    - X: 2D NumPy array of shape (n, d)
    - k: Number of neighbors to return
    
    Returns:
    - knn_indices: 2D NumPy array of shape (n, k)
    - pairwise_distances: 2D NumPy array of shape (n, n)
    """
    D = pairwise_distances(X)
    # Set diagonal to infinity so we don't include self as neighbor
    np.fill_diagonal(D, np.inf)
    
    # Get indices of k smallest distances for each row
    knn_indices = np.argsort(D, axis=1)[:, :k]
    
    return knn_indices, D

def plot_results(X: np.ndarray, y: np.ndarray) -> None:
    # Assume X and y are defined as above

    fig = go.Figure()

    # Add scatter trace, color by y
    fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(color=y, colorscale='Viridis', showscale=True),
        name='Data Points'
    ))

    fig.update_layout(
        title='Scatter Plot of X Colored by y',
        xaxis_title='X1',
        yaxis_title='X2'
    )

    fig.show()