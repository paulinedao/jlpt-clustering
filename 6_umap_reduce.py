import numpy as np
import umap

#input_emb = "data/jlpt_embeddings_normalized.npy"
#output_emb = "data/jlpt_embeddings_umap_15d.npy"

def main(input_emb=None, output_emb=None):
    X = np.load(input_emb)

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    X_reduced = reducer.fit_transform(X)
    np.save(output_emb, X_reduced)

    print("UMAP-reduced shape:", X_reduced.shape)

if __name__ == "__main__":
    main(input_emb=snakemake.input.input_emb, 
        output_emb=snakemake.output.output_emb)