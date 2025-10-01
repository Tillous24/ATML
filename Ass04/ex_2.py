import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple, Dict

def load_mnist_38(train=True, device="cpu", seed=0):
    torch.manual_seed(seed)
    ds = datasets.MNIST(
        root="./data", train=train, download=True, transform=transforms.ToTensor()
    )
    X, y = [], []
    for img, label in ds:
        if label in (3, 8):
            X.append(img.view(-1).numpy())         # 784 vector in [0,1]
            y.append(1 if label == 8 else 0)        # map 8->1, 3->0
    X = np.stack(X).astype(np.float64)
    y = np.array(y, dtype=np.int64)
    return X, y

def stratified_uniform_split(X, y, num_nodes=4, seed=0):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    parts = [[] for _ in range(num_nodes)]

    for k, idx in enumerate(idx0):
        parts[k % num_nodes].append(idx)
    for k, idx in enumerate(idx1):
        parts[k % num_nodes].append(idx)
    node_splits = []
    for p in parts:
        p = np.array(p, dtype=int)
        node_splits.append((X[p], y[p]))
    return node_splits

#Log regression
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_loss_and_grad(theta, X, y, l2=0.0):
    """
    theta = [w; b], w in R^d, b scalar
    Loss: mean cross-entropy; optional L2 on w only.
    """
    n, d = X.shape
    w = theta[:-1]
    b = theta[-1]
    z = X @ w + b
    p = sigmoid(z)
    eps = 1e-12
    #loss over full dataset
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    #gradient
    diff = (p - y)
    grad_w = (X.T @ diff) / n + l2 * w
    grad_b = np.mean(diff)  # no L2 on bias
    grad = np.concatenate([grad_w, np.array([grad_b])])
    return loss + (l2 / 2.0) * np.sum(w * w), grad

#Graphs
def laplacian_from_adjacency(A):
    deg = np.diag(np.sum(A, axis=1))
    return deg - A

def dgd_train(node_splits, A, T=200, alpha=0.15, gamma=0.2, l2=0.0, seed=0):
    """
    node_splits: list of (X_i, y_i) for i=0..3
    A: adjacency (4x4) with 1 for edges (i!=j), 0 otherwise (undirected)
    Returns: history of global loss (evaluated with node 0's model)
    """
    rng = np.random.default_rng(seed)
    n_nodes = len(node_splits)
    d = node_splits[0][0].shape[1]
    #init theta_i^0 (w,b)
    Theta = np.zeros((n_nodes, d + 1), dtype=np.float64)
    X_all = np.concatenate([Xi for Xi, yi in node_splits], axis=0)
    y_all = np.concatenate([yi for Xi, yi in node_splits], axis=0)

    L = laplacian_from_adjacency(A)
    deg_max = np.max(np.diag(L))
    #alpha < 1/deg_max to keep gossip stable
    if alpha >= 1.0 / max(1, deg_max):
        print(f"[warn] alpha={alpha} may be too large for deg_max={deg_max}. "
              f"Consider using alpha < {1.0/deg_max:.3f}")

    global_loss_hist = []
    for t in range(T):
        #gossip (vectorized: Theta_half = Theta - alpha * L @ Theta)
        Theta_half = Theta - alpha * (L @ Theta)

        #local gradient step
        Theta_next = np.empty_like(Theta)
        for i in range(n_nodes):
            Xi, yi = node_splits[i]
            _, grad_i = logistic_loss_and_grad(Theta_half[i], Xi, yi, l2=l2)
            Theta_next[i] = Theta_half[i] - gamma * grad_i

        Theta = Theta_next

        # Evaluate global training loss at node 1's model (index 0)
        loss_global, _ = logistic_loss_and_grad(Theta[0], X_all, y_all, l2=0.0)
        global_loss_hist.append(loss_global)

    return np.array(global_loss_hist), Theta

#path 1-2-3-4
A_path = np.array([
    [0,1,0,0],
    [1,0,1,0],
    [0,1,0,1],
    [0,0,1,0],
], dtype=float)

#ring 1-2-3-4-1
A_ring = np.array([
    [0,1,0,1],
    [1,0,1,0],
    [0,1,0,1],
    [1,0,1,0],
], dtype=float)

#complete graph K4
A_complete = np.ones((4,4), dtype=float) - np.eye(4)

def run_all(topologies: Dict[str, np.ndarray],
            T=250, alpha=None, gamma=0.2, l2=0.0, seed=0):
    X, y = load_mnist_38(train=True, seed=seed)
    splits = stratified_uniform_split(X, y, num_nodes=4, seed=seed)
    losses = {}
    for name, A in topologies.items():
        deg_max = int(np.max(np.sum(A, axis=1)))
        a = alpha if alpha is not None else 0.45 / max(1, deg_max)
        hist, _ = dgd_train(splits, A, T=T, alpha=a, gamma=gamma, l2=l2, seed=seed)
        losses[name] = hist
    return losses


if __name__ == "__main__":
    topologies = {"A (path)": A_path, "B (ring)": A_ring, "C (complete)": A_complete}
    losses = run_all(topologies, T=250, gamma=0.2, l2=0.0, seed=1)
    # Plot
    plt.figure(figsize=(7,4.5))
    for name, hist in losses.items():
        plt.plot(hist, label=name)
    plt.xlabel("Iteration t")
    plt.ylabel("Global training loss (node 1 model)")
    plt.title("DGD: global training loss at node 1 vs. iteration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()