import faiss
import numpy as np
import pickle
import os

class FAISSIndex:
    """Manages vector database for fast similarity search"""

    def __init__(self, dimension = 512):
        self.dimension=dimension
        self.index= faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add_embedding(self, embedding, metadata):
        """
        Add a single embedding to index
        
        embedding: 512-dim numpy array
        metadata: dict with keys like {'path': ..., 'modality': ..., 'description': ...}
        """
        # Ensure embedding is normalized for cosine similarity
        embedding = embedding/np.linalg.norm(embedding)
        embedding=embedding.reshape(1, -1).astype('float32')

        self.index.add(embedding)
        self.metadata.append(metadata)

    def add_batch(self, embeddings, metadata_list):
        """Add multiple embeddings at once"""
        embeddings = np.array(embeddings).astype('float32')
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def search(self, query_embedding, k=10):
        """
        search for top k similar items

        returns: list of (score, metadata) tuples
        """

        query_embedding = query_embedding/np.linalg.norm(query_embedding)
        query_embedding=query_embedding.reshape(1,-1).astype('float32')

        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx!= -1: #valid result
                results.append({
                    'score':float(dist),
                    'metadata' : self.metadata[idx]
                })
        return results
    
    def save(self, index_path='embeddings/faiss.index', metadata_path='embeddings/metadata.pkl'):
        """Save index and metadata to disk"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Index saved: {len(self.metadata)} items")

    def load(self, index_path='embeddings/faiss.index', metadata_path='embeddings/metadata.pkl'):
        """Load index and metadata from disk"""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Index loaded: {len(self.metadata)} items")