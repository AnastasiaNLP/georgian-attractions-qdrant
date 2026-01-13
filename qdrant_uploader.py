# Qdrant uploader 

"""
Uploads processed data to Qdrant Cloud.
"""

import logging
from typing import List
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class QdrantUploader:
    """
    Uploads data to Qdrant Cloud.

    Attributes:
    client : QdrantClient
        Qdrant client instance
    collection_name : str
        Name of the collection
    vector_size : int
        Size of embedding vectors
    """

    def __init__(self, url: str, api_key: str, collection_name: str, vector_size: int):
        self.collection_name = collection_name
        self.vector_size = vector_size

        logger.info(f"Connecting to Qdrant Cloud...")
        print(f"Connecting to Qdrant Cloud")

        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)

        print(f" Connected to Qdrant!")
        print(f"   URL: {url[:50]}...")

    def create_collection(self, recreate: bool = False):
        """Create or recreate collection."""
        print(f"Create collection")

        # check if collection exists
        collections = self.client.get_collections()
        existing_names = [col.name for col in collections.collections]

        if self.collection_name in existing_names:
            if recreate:
                print(f" Collection '{self.collection_name}' exists. Deleting...")
                self.client.delete_collection(self.collection_name)
                print(f"    Deleted")
            else:
                print(f" Collection '{self.collection_name}' already exists!")
                return

        # create collection
        print(f" Creating collection '{self.collection_name}'...")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )

        print(f" Collection created")

        # verify
        collection_info = self.client.get_collection(self.collection_name)
        print(f"\n Collection info:")
        print(f"   Name: {self.collection_name}")
        print(f"   Vector size: {collection_info.config.params.vectors.size}")
        print(f"   Distance: {collection_info.config.params.vectors.distance}")
        print(f"   Points: {collection_info.points_count}")

    def upload_data(self, df: pd.DataFrame, batch_size: int = 100):
        """Upload data in batches."""
        print(f" Uploading data to Qdrant")

        print(f"   Total records: {len(df)}")
        print(f"   Batch size: {batch_size}")

        # prepare points
        points = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing points"):
            # prepare payload (metadata)
            payload = {
                'id': str(row['id']),
                'name': str(row['name']),
                'description': str(row['description']),
                'location': str(row['location']),
                'category': str(row['category']),
                'language': str(row['language']),
                'tags': row['tags'] if isinstance(row['tags'], list) else [],
                'photo_name': str(row['photo_name']),
                'photo_author': str(row['photo_author']),
                'license': str(row['license']),
                'has_processed_image': bool(row['has_processed_image']),
                'image_url': str(row['image_url']) if row['image_url'] else None,
                'combined_text': str(row['combined_text'])
            }

            # create point
            point = PointStruct(
                id=idx,
                vector=row['embedding'].tolist(),
                payload=payload
            )

            points.append(point)

        # upload in batches
        print(f"\n Uploading in batches...")

        for i in tqdm(range(0, len(points), batch_size), desc="Uploading batches"):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        print(f"\n Upload complete")

        # verify
        collection_info = self.client.get_collection(self.collection_name)
        print(f"\n Final collection stats:")
        print(f"   Points uploaded: {collection_info.points_count}")
        print(f"   Expected: {len(df)}")

        if collection_info.points_count == len(df):
            print(f"   All points uploaded successfully")
        else:
            print(f"   Mismatch in point count")
