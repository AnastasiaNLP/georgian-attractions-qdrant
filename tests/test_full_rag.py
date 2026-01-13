# TEST:  a complete formula for thermal conveying that works with images test 

"""
Test complete RAG pipeline: Query -> Search -> Text + Image
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import Config
import torch
from PIL import Image
import requests
from io import BytesIO

print(" Test: full rag pipeline")

# 1. setup
print("\n Setting up...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer(Config.EMBEDDING_MODEL, device=device)
client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
print(f" Model loaded on {device}")
print(f" Connected to Qdrant")

# 2. test queries
print("Testing queries")

test_queries = [
    "Borjomi National Park",  # English
    "пляжи Батуми",           # Russian
    "churches",        # Category search
    "lakes",    # General
]

for query in test_queries:
    print(f" QUERY: '{query}'")

    # Step 1: create embedding
    print("\n Step 1: Creating query embedding...")
    query_vector = embedder.encode(query).tolist()
    print(f" Vector created (size: {len(query_vector)})")

    # Step 2: search in Qdrant
    print("\n Step 2: Searching in Qdrant...")
    results = client.search(
        collection_name=Config.COLLECTION_NAME,
        query_vector=query_vector,
        limit=3,
        with_payload=True
    )
    print(f" Found {len(results)} results")

    # Step 3: display results
    print("\n Step 3: Results with images:")

    for i, result in enumerate(results, 1):
        print(f"Result #{i} (Score: {result.score:.4f})")

        # text data
        print(f" Name: {result.payload['name']}")
        print(f" Location: {result.payload['location']}")
        print(f" Category: {result.payload['category']}")
        print(f" Language: {result.payload['language']}")
        print(f" Description: {result.payload['description'][:200]}...")

        # image
        image_url = result.payload.get('image_url')
        if image_url:
            print(f"\n  Image URL: {image_url}")

            # Try to load and display image info
            try:
                response = requests.get(image_url, timeout=5)
                img = Image.open(BytesIO(response.content))
                print(f" Image loaded successfully!")
                print(f"   Size: {img.size}")
                print(f"   Format: {img.format}")
                print(f"   Mode: {img.mode}")
            except Exception as e:
                print(f"  Could not load image: {e}")
        else:
            print(f"\n No image available")

print(" Test completed")
print("   1. Query -> Embedding ")
print("   2. Search in Qdrant ")
print("   3. Retrieve text data ")
print("   4. Load images from Cloudinary ")
print("\n Ready for RAG bot!")
