
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def build_nlp_index():
    print("Loading cleaned dataset...")
    df = pd.read_csv("data/cleaned_contracts.csv")
    
    # Fill any remaining empty strings
    df['Cleaned_Description'] = df['Cleaned_Description'].fillna("")
    
    print(f"Loaded {len(df)} contracts.")
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 1. Generate dense vector embeddings for all contract descriptions
    print("Generating semantic vector embeddings (this takes ~30-60 seconds)...")
    descriptions = df['Cleaned_Description'].tolist()
    embeddings = model.encode(descriptions, batch_size=64, show_progress_bar=True)
    
    # 2. Save embeddings to a numpy array for quick reloading later
    np.save("data/contract_embeddings.npy", embeddings)
    print("Vector embeddings saved to data/contract_embeddings.npy!")
    
    return model, df, embeddings

def semantic_search(query, model, df, embeddings, top_k=5):
    """
    Performs cosine similarity search between a natural language query and all contracts.
    """
    print(f"\n--- Searching for: '{query}' ---")
    query_vector = model.encode([query])
    
    # Compute cosine similarity between query and all contract vectors
    scores = cosine_similarity(query_vector, embeddings)[0]
    
    # Get top K indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        row = df.iloc[idx]
        score = scores[idx]
        print(f"\n[Match #{rank}] Similarity Score: {score:.4f}")
        print(f"Recipient:       {row['Recipient Name']}")
        print(f"Awarding Agency: {row['Awarding Agency']}")
        print(f"Award Amount:    ${row['Award Amount']:,.2f}")
        print(f"Description:     {row['Cleaned_Description'][:250]}...")
        results.append(row)
        
    return results

if __name__ == "__main__":
    # Build or load embeddings
    model, df, embeddings = build_nlp_index()
    
    # Test semantic queries
    print("\n================ TEST QUERIES ================")
    
    # Test 1: High-tech / AI / Cloud
    semantic_search("Cloud computing and cyber defense modernization", model, df, embeddings, top_k=3)
    
    # Test 2: Medical / Healthcare
    semantic_search("Medical supplies and healthcare clinical support", model, df, embeddings, top_k=3)
    
    # Test 3: Infrastructure / Construction
    semantic_search("Nuclear facility maintenance and laboratory operations", model, df, embeddings, top_k=3)