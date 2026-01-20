import numpy as np
from sklearn.preprocessing import StandardScaler

INPUT_EMB = "data/jlpt_word_vectors_bert.npy"
OUTPUT_EMB = "data/jlpt_embeddings_normalized.npy"

def main():
    X = np.load(INPUT_EMB)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    np.save(OUTPUT_EMB, X_norm)
    print("Saved normalized embeddings:", X_norm.shape)

if __name__ == "__main__":
    main()
