rule hdbscan_cluster:
    input:
        input_emb="data/jlpt_embeddings_umap_15d.npy"
    output:
        output_labels="data/jlpt_cluster_labels.npy"
    script:
        "../7_hdbscan_cluster.py"