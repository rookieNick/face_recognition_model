import os
import json
import numpy as np
from numpy.linalg import norm

def check_embedding(embedding):
    """Check embedding properties"""
    print(f"Shape: {embedding.shape}")
    print(f"Norm (should be 1.0 for L2 normalized): {norm(embedding):.6f}")
    print(f"Min value: {np.min(embedding):.6f}")
    print(f"Max value: {np.max(embedding):.6f}")
    print(f"Mean value: {np.mean(embedding):.6f}")
    print(f"Std deviation: {np.std(embedding):.6f}")
    print("-" * 50)

def main():
    embeddings_dir = 'embeddings'
    if not os.path.exists(embeddings_dir):
        print("❌ No embeddings directory found!")
        return

    print("Checking embeddings in directory:", embeddings_dir)
    print("=" * 50)

    for filename in os.listdir(embeddings_dir):
        if filename.endswith('_embedding.json'):
            print(f"\nChecking {filename}:")
            with open(os.path.join(embeddings_dir, filename), 'r') as f:
                embedding = np.array(json.load(f))
                check_embedding(embedding)

if __name__ == "__main__":
    main() 