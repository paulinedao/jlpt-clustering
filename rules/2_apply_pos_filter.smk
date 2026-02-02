rule apply_pos_filter:
    input:
        input_csv="data/jlpt_vocab_n5.csv"
    output:
        output_csv="data/jlpt_words_pos_filtered.csv"
    script:
        "../2_apply_pos_filter.py"