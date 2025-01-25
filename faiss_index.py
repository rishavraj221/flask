import faiss
import numpy as np
import os


class FaissIndex:
    def __init__(self, dimension, index_type="Flat", index_file="index.faiss"):
        self.dimension = dimension
        self.index_type = index_type
        self.index_file = index_file
        self.index = self._create_index()

    def _create_index(self):
        if self.index_type == "Flat":
            return faiss.IndexFlatL2(self.dimension)  # L2 distance
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError("Unsupported index type")

    def add_vectors(self, embeddings):
        self.index.add(embeddings)

    def search(self, query_embedding, k=10):
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def save_index(self):
        faiss.write_index(self.index, self.index_file)

    def load_index(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        else:
            raise FileNotFoundError("Index file not found")

    def is_trained(self):
        return self.index.is_trained
