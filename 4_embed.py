import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

#token_json = "data/jlpt_tokens.json"  # your tokenized words JSON
#model_name = "sonoisa/sentence-bert-base-ja-mean-tokens"  # Japanese sentence-BERT
#output_vectors = "data/jlpt_word_vectors_bert.npy"
#output_words = "data/jlpt_words_used_bert.csv"

def main(token_json, model_name, output_vectors, output_words):
    # Load words
    with open(token_json, "r", encoding="utf-8") as f:
        tokenized = json.load(f)

    # Flatten to list of words (each sublist has one word)
    words = [t[0] for t in tokenized]

    # Load BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # evaluation mode

    embeddings = []

    # Encode each word
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors="pt")
            emb = mean_pooling(model(**inputs), inputs["attention_mask"]).squeeze().numpy()
            embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Save embeddings and words
    np.save(output_vectors, embeddings)
    import pandas as pd
    pd.DataFrame({"Word": words}).to_csv(output_words, index=False)

    print(f"Embedded {len(words)} words")
    print(f"Embedding shape: {embeddings.shape}")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

if __name__ == "__main__":
    main(token_json=snakemake.input.token_json, 
         model_name=snakemake.params.model_name, 
         output_vectors=snakemake.output.output_vectors, 
         output_words=snakemake.output.output_words)