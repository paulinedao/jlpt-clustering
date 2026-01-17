# tokenize words by treating each word (even those composed of multiple characters) as a single token
import pandas as pd
import json

INPUT_CSV = "data/jlpt_words_pos_filtered.csv"
OUTPUT_JSON = "data/jlpt_tokens.json"

def main():
    tokenize_words(INPUT_CSV, OUTPUT_JSON)

def tokenize_words(input_csv, output_json):
    # Load words
    df = pd.read_csv(input_csv)
    words = df["Word"].astype(str).tolist()

    # Each word is treated as ONE token
    tokens = [[word] for word in words]

    # Save tokenized output
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(tokens)} tokenized entries to {output_json}")

if __name__ == "__main__":
    main()
