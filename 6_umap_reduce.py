import numpy as np
import umap

INPUT_EMB = "data/jlpt_embeddings_normalized.npy"
OUTPUT_EMB = "data/jlpt_embeddings_umap_15d.npy"

def main():
    X = np.load(INPUT_EMB)

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    X_reduced = reducer.fit_transform(X)
    np.save(OUTPUT_EMB, X_reduced)

    print("UMAP-reduced shape:", X_reduced.shape)

if __name__ == "__main__":
    main()
