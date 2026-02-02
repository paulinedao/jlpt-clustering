rule embed:
    input:
        token_json="data/jlpt_tokens.json"
    params:
        model_name="sonoisa/sentence-bert-base-ja-mean-tokens"
    output:
        output_vectors="data/jlpt_word_vectors_bert.npy",
        output_words="data/jlpt_words_used_bert.csv"
    script:
        "../4_embed.py"