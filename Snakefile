include: "rules/1_process_vocab.smk"
include: "rules/2_apply_pos_filter.smk"
include: "rules/3_tokenize.smk"
include: "rules/4_embed.smk"
include: "rules/5_normalize_embeddings.smk"
include: "rules/6_umap_reduce.smk"
include: "rules/7_hdbscan_cluster.smk"
include: "rules/8_matplotlib_visualize.smk"

rule all:
    input:
        "data/jlpt_umap_clusters.png"
