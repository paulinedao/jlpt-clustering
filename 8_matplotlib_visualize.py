import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

#INPUT_EMB = "data/jlpt_embeddings_umap_15d.npy"  
#INPUT_LABELS = "data/jlpt_cluster_labels.npy"   
#WORDS_CSV = "data/jlpt_words_used_bert.csv"         
#OUTPUT_CSV = "data/jlpt_viz_2d.csv"                 
#OUTPUT_PNG = "data/jlpt_umap_clusters.png"         

def main(input_emb, input_labels, words_csv, output_csv, output_png):
    X = np.load(input_emb)             
    labels = np.load(input_labels)     
    words_df = pd.read_csv(words_csv)
    words = words_df["Word"].tolist()

    # Sanity check
    assert X.shape[0] == len(labels) == len(words), "Mismatch in number of embeddings / labels / words"

    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer_2d.fit_transform(X)

    df_viz = pd.DataFrame({
        "word": words,
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
        "cluster": labels
    })
    df_viz.to_csv(output_csv, index=False)
    print("Saved CSV for visualization:", output_csv)


    plt.figure(figsize=(10, 8))

    clusters = np.unique(labels)
    for c in clusters:
        subset = df_viz[df_viz["cluster"] == c]
        if c == -1:
            plt.scatter(
                subset["x"], subset["y"],
                c="lightgray", s=15, label="Noise"
            )
        else:
            plt.scatter(
                subset["x"], subset["y"],
                s=25, label=f"Cluster {c}"
            )

    plt.title("UMAP + HDBSCAN Clustering (Japanese N5 Words)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=1.2, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()

    print("Saved matplotlib plot:", output_png)

if __name__ == "__main__":
    main(
        input_emb=snakemake.input.input_emb,
        input_labels=snakemake.input.input_labels,
        words_csv=snakemake.input.words_csv,
        output_csv=snakemake.output.output_csv,
        output_png=snakemake.output.output_png
    )