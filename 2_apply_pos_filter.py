import re
import pandas as pd
from sudachipy import tokenizer, dictionary

ALLOWED_POS = {"名詞", "動詞", "形容詞"}  # nouns, verbs, adjectives
BAD_CHAR_PATTERN = re.compile(r"[;、,]")

def main(input_csv=None, output_csv=None):
    #input_csv = "data/jlpt_vocab_n5.csv"
    #output_csv = "data/jlpt_words_pos_filtered.csv"
    
    df = pd.read_csv(input_csv)

    kept_words = []

    for word in df["Word"]:
        if not is_clean_word(word):
            continue

        pos, normalized = analyze_word(word)

        if pos in ALLOWED_POS:
            kept_words.append(normalized)

    output_df = (
        pd.DataFrame({"Word": kept_words})
        .drop_duplicates()
        .sort_values("Word")
        .reset_index(drop=True)
    )

    output_df.to_csv(output_csv, index=False)

    print("=== POS Filtering Summary ===")
    print(f"Original words: {len(df)}")
    print(f"Kept words:     {len(output_df)}")
    print(f"Saved to:      {output_csv}")

def is_clean_word(word):
    """
    Remove entries with obvious noise such as punctuation.
    """
    return not bool(BAD_CHAR_PATTERN.search(word))

# load dictionnary from sudachidict_core
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

def analyze_word(word):
    """
    Analyze a Japanese word using Sudachi.

    Returns:
        (pos, normalized_word) or (None, None)
    """
    tokens = tokenizer_obj.tokenize(word, mode)

    if len(tokens) == 0:
        return None, None

    # Use POS of first token as representative
    pos = tokens[0].part_of_speech()[0]

    # Normalize only if single-token
    if len(tokens) == 1:
        normalized = tokens[0].dictionary_form()
    else:
        normalized = word  # keep surface form for compounds

    return pos, normalized


if __name__ == "__main__":
    main(input_csv=snakemake.input.input_csv, 
         output_csv=snakemake.output.output_csv)
