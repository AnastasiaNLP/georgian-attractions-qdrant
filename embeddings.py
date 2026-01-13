# embeddings generator 
"""
Creates embeddings for text data using SentenceTransformers.
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    """
    Generates embeddings for records using SentenceTransformers.

    Attributes:
    model_name : str
        SentenceTransformer model name
    device : str
        'cuda' or 'cpu'
    """

    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'

        logger.info(f"Loading embedding model: {model_name}")
        print(f"Loading embedding model")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

        # test
        test_vector = self.model.encode("Test")
        print(f" Model loaded. Vector size: {len(test_vector)}")

    def create_combined_text(self, record: Dict[str, Any]) -> str:
        """Create combined text for embedding."""
        parts = []

        if record.get('name'):
            parts.append(f"Name: {record['name']}")

        if record.get('description'):
            parts.append(f"Description: {record['description']}")

        if record.get('category'):
            parts.append(f"Category: {record['category']}")

        if record.get('location'):
            parts.append(f"Location: {record['location']}")

        if record.get('tags'):
            tags = record['tags']
            if isinstance(tags, list):
                parts.append(f"Tags: {', '.join(tags)}")
            else:
                parts.append(f"Tags: {tags}")

        return " | ".join(parts)

    def generate(self, records: List[Dict[str, Any]], batch_size: int = 32) -> pd.DataFrame:
        """
        Generate embeddings for all records.

        Parameters:
        records : List[Dict]
            Records to process
        batch_size : int
            Batch size for encoding

        Returns:
        pd.DataFrame
            DataFrame with embeddings
        """
        logger.info(f"Generating embeddings for {len(records)} records")
        print(f"Generating embeddings")

        # create combined text
        print("Creating combined text...")
        for rec in records:
            rec['combined_text'] = self.create_combined_text(rec)

        # generate embeddings in batches
        print(f"Encoding text (batch_size={batch_size})...")
        embeddings = []

        for i in tqdm(range(0, len(records), batch_size), desc="Encoding batches"):
            batch_texts = [rec['combined_text'] for rec in records[i:i+batch_size]]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings)

        # Add embeddings to records
        for rec, emb in zip(records, embeddings):
            rec['embedding'] = emb

        # Create DataFrame
        df = pd.DataFrame(records)

        print(f" Generated {len(df)} embeddings")
        print(f"   Vector size: {len(df.iloc[0]['embedding'])}")

        return df
