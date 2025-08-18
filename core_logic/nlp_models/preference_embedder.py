import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class PreferenceEmbedder:
    """
    Manages the generation of contextual embeddings for predefined attributes
    and calculates the similarity of review texts to these attributes.
    """
    def __init__(self, model_name: str = 'jinaai/jina-embeddings-v3'):
        """Initializes the SentenceTransformer model."""
        print(f"Loading SentenceTransformer model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to 'paraphrase-multilingual-MiniLM-L12-v2' for stability.")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("Fallback model 'paraphrase-multilingual-MiniLM-L12-v2' loaded successfully.")
        
        self.attribute_embeddings = None
        self.attribute_names = None

    def generate_attribute_embeddings(self, attributes: list[str]) -> np.ndarray:
        """Generates and stores embeddings for a list of predefined attributes."""
        if not attributes:
            raise ValueError("Attribute list cannot be empty.")

        print(f"Generating embeddings for {len(attributes)} attributes...")
        self.attribute_embeddings = self.model.encode(attributes, convert_to_numpy=True)
        self.attribute_names = attributes
        print("Attribute embeddings generated.")
        return self.attribute_embeddings

    def get_review_attribute_scores(self, review_text: str) -> dict[str, float]:
        """Calculates the similarity of a given review text to the predefined attributes."""
        if self.attribute_embeddings is None or self.attribute_names is None:
            print("Warning: Attributes not set. Call generate_attribute_embeddings first.")
            return {}
        if not review_text or not review_text.strip():
            return {attr: 0.0 for attr in self.attribute_names} 

        review_embedding = self.model.encode([review_text], convert_to_numpy=True)
        similarities = cosine_similarity(review_embedding, self.attribute_embeddings)[0]
        attribute_scores = {
            self.attribute_names[i]: float(similarities[i])
            for i in range(len(self.attribute_names))
        }
        return attribute_scores