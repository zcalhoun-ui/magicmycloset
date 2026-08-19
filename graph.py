import os
import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker  
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt

# 1. Get the script directory and build file paths
def get_script_directory():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return script_dir

# 2. Load the CSV file
script_dir = get_script_directory()
matrix_path = os.path.join(script_dir, "matrix_numerical.csv")
converted_matrix_df = pd.read_csv(matrix_path)
# remove bad outfit connections
converted_matrix_df = converted_matrix_df[converted_matrix_df['Rating'] != -1]

# 3. Convert columns to integer coordinates   
converted_matrix_df['Top'] = converted_matrix_df['Top'].astype(int)
converted_matrix_df['Bottom'] = converted_matrix_df['Bottom'].astype(int)
converted_matrix_df['A/L'] = converted_matrix_df['A/L'].astype(int)

print("Success to here")

# 8. Allocate the dense tensor (initialized with 0.0 for matrix math compatibility)
shape = (
    converted_matrix_df['Top'].max() + 1, 
    converted_matrix_df['Bottom'].max() + 1, 
    converted_matrix_df['A/L'].max() + 1
)
scorer_tensor = np.zeros(shape, dtype=float)  # Use float for standard linear algebra operations

print("I allocated the tensor")
scorer_tensor[
    converted_matrix_df['Top'].values, 
    converted_matrix_df['Bottom'].values, 
    converted_matrix_df['A/L'].values
] = 1.0

print("Successfully populated the tensor")

original_tensor = scorer_tensor

# === Build LRA Matrix ===
# 1. Keep NumPy as the backend since you are using pandas values
tl.set_backend('numpy')

# 2. Collapse the 22 layers into a 2D bipartite matrix (1043 x 125)
# We sum across the layers axis (axis=2) to get interaction counts
bipartite_matrix = np.sum(original_tensor, axis=2)

# 3. Project to a square node-node adjacency matrix (1043 x 1043)
# Multiplying the matrix by its transpose finds co-occurrences between Type A nodes
adjacency_matrix = np.dot(bipartite_matrix, bipartite_matrix.T)

# Zero out the diagonal so nodes don't have self-loops to themselves
np.fill_diagonal(adjacency_matrix, 0)

# 4. Convert the dense NumPy array into a NetworkX Graph
G = nx.from_numpy_array(adjacency_matrix)

# 5. Run Louvain community detection
communities = community_louvain.best_partition(G)

# 'communities' is a dictionary: {node_id: community_id}
print(f"Detected {len(set(communities.values()))} unique communities.")

# ===build image===

# 1. Extract the community assignments to color the nodes
node_colors = [communities[node] for node in G.nodes()]

# 2. Filter weak edges to make the layout cleaner (optional but recommended)
# Only keep edges with a weight above a certain threshold (e.g., median weight)
weights = [d['weight'] for u, v, d in G.edges(data=True)]
threshold = np.percentile(weights, 75)  # Keep top 25% strongest connections

edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
G_filtered = nx.Graph()
G_filtered.add_nodes_from(G.nodes())
G_filtered.add_edges_from(edges_to_keep)

# 3. Compute layout positions (Spring layout separates clusters visually)
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G_filtered, k=0.15, seed=42)

# 4. Draw the nodes and edges
nx.draw_networkx_nodes(G_filtered, pos, node_size=20, 
                       node_color=node_colors, cmap=plt.cm.jet)
nx.draw_networkx_edges(G_filtered, pos, alpha=0.1, edge_color="gray")

plt.title("Community Network Graph (1043 Nodes Bipartite Projection)")
plt.axis("off")
plt.show()
