import numpy as np
import hdbscan

INPUT_EMB = "data/jlpt_embeddings_umap_15d.npy"
OUTPUT_LABELS = "data/jlpt_cluster_labels.npy"

def main():
    X = np.load(INPUT_EMB)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=3,
        metric="euclidean"
    )

    labels = clusterer.fit_predict(X)
    np.save(OUTPUT_LABELS, labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Clusters found:", n_clusters)
    print("Noise points:", (labels == -1).sum())

if __name__ == "__main__":
    main()
