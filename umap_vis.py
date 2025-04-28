import umap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

path_to_embeddings = '/Users/anthony/Documents/2024-2025 Brown Stuff/Spring 2025/DL_in_Genomics/sc-DMAE/data/embedding_59.npy'
path_to_labels = '/Users/anthony/Documents/2024-2025 Brown Stuff/Spring 2025/DL_in_Genomics/sc-DMAE/data/types_59.txt'

embeddings = np.load(path_to_embeddings)
labels = np.genfromtxt(path_to_labels, delimiter=',', skip_header=1)[:, 1]

# Create UMAP reducer
reducer = umap.UMAP(n_components=2, random_state=42)

# Fit and transform the embeddings
embedding_2d = reducer.fit_transform(embeddings)

# Get the unique labels
unique_labels = np.unique(labels)

# Choose a discrete color palette
colors_list = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # or use 'Set1', 'tab20', etc.

# Create a figure
plt.figure(figsize=(10,8))

# Plot each label individually
for i, label in enumerate(unique_labels):
    idx = labels == label  # Find indices where label matches
    plt.scatter(
        embedding_2d[idx, 0],   # x values
        embedding_2d[idx, 1],   # y values
        color=colors_list[i],   # assign color
        label=str(label),       # label for the legend
        s=2                    # size of the points
    )

# Add legend
plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('UMAP Visualization of Hrvatin 59')
plt.tight_layout()
plt.show()