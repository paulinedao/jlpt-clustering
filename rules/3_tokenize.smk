rule tokenize:
    input:
        input_csv="data/jlpt_words_pos_filtered.csv"
    output:
        output_json="data/jlpt_tokens.json"
    script:
        "../3_tokenize.py"