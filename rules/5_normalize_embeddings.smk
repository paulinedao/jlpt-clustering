rule normalize_embeddings:
    input:
        input_emb="data/jlpt_word_vectors_bert.npy"
    output:
        output_emb="data/jlpt_embeddings_normalized.npy"
    script:
        "../5_normalize_embeddings.py"