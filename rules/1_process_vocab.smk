rule process_vocab:
    input:
        input_csv='data/jlpt_vocab.csv'
    output:
        output_csv='data/jlpt_vocab_n5.csv'
    script:
        "../1_process_vocab.py"
