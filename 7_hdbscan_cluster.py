import numpy as np
import hdbscan

#input_emb = "data/jlpt_embeddings_umap_15d.npy"
#output_labels = "data/jlpt_cluster_labels.npy"

def main(input_emb, output_labels):
    X = np.load(input_emb)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=3,
        metric="euclidean"
    )

    labels = clusterer.fit_predict(X)
    np.save(output_labels, labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Clusters found:", n_clusters)
    print("Noise points:", (labels == -1).sum())

if __name__ == "__main__":
    main(input_emb=snakemake.input.input_emb, 
         output_labels=snakemake.output.output_labels)