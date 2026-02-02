import pandas as pd

def main(input_csv=None, output_csv=None):
    #input_csv = 'data/jlpt_vocab.csv'
    #output_csv = 'data/jlpt_vocab_n5.csv'
    process_vocab(input_csv, output_csv)
    
def process_vocab(input_csv, output_csv):
    (
        pd.read_csv(input_csv)
        .drop(columns=['Furigana', 'English'], errors='ignore')
        .query("`JLPT Level` == 'N5'")
        .rename(columns={'Original': 'Word'})
        .to_csv(output_csv, index=False)
    )
    
if __name__ == "__main__":
    main(input_csv=snakemake.input.input_csv, 
         output_csv=snakemake.output.output_csv)