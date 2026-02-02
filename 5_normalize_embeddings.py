import numpy as np
from sklearn.preprocessing import StandardScaler

#input_emb = "data/jlpt_word_vectors_bert.npy"
#output_emb = "data/jlpt_embeddings_normalized.npy"

def main(input_emb=None, output_emb=None):
    X = np.load(input_emb)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    np.save(output_emb, X_norm)
    print("Saved normalized embeddings:", X_norm.shape)

if __name__ == "__main__":
    main(input_emb=snakemake.input.input_emb, 
         output_emb=snakemake.output.output_emb)