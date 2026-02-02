rule umap_reduce:
    input:
        input_emb="data/jlpt_embeddings_normalized.npy"
    output:
        output_emb="data/jlpt_embeddings_umap_15d.npy"
    script:
        "../6_umap_reduce.py"