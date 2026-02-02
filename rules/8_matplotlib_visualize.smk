rule matplotlib_visualize:
    input:
        input_emb="data/jlpt_embeddings_umap_15d.npy",
        input_labels="data/jlpt_cluster_labels.npy",
        words_csv="data/jlpt_words_used_bert.csv"  
    output:
        output_csv="data/jlpt_viz_2d.csv",
        output_png="data/jlpt_umap_clusters.png"
    script:
        "../8_matplotlib_visualize.py"