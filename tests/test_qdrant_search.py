# TEST: QDRANT SEARCH

"""
Тестирование поиска в Qdrant базе.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import Config
import torch

print(" TEST: QDRANT SEARCH")


print("\n1Connecting to Qdrant...")
client = QdrantClient(
    url=Config.QDRANT_URL,
    api_key=Config.QDRANT_API_KEY
)


collection_info = client.get_collection(Config.COLLECTION_NAME)
print(f" Connected!")
print(f"   Collection: {Config.COLLECTION_NAME}")
print(f"   Points: {collection_info.points_count}")


print("\n Loading embedding model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer(Config.EMBEDDING_MODEL, device=device)
print(f" Model loaded on {device}")


test_queries = [
    "пляжи в Батуми",
    "ancient churches in Georgia",
    "горы и природа",
    "museums in Tbilisi",
    "wine tasting"
]

print("\n Testing search...")

for query in test_queries:
    print(f"\n Query: '{query}'")
    # Создаём embedding запроса
    query_vector = embedder.encode(query).tolist()

    
    results = client.search(
        collection_name=Config.COLLECTION_NAME,
        query_vector=query_vector,
        limit=3,  # Топ-3 результата
        with_payload=True
    )


    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.payload['name']}")
        print(f"   Score: {result.score:.4f}")
        print(f"   Category: {result.payload['category']}")
        print(f"   Location: {result.payload['location']}")
        print(f"   Language: {result.payload['language']}")
        print(f"   Has image: {result.payload['has_processed_image']}")
        print(f"   Description: {result.payload['description'][:150]}...")

print(" SEARCH TEST COMPLETED!")


print("\n Scrolling through all records for statistics...")

from collections import Counter

categories = []
languages = []
with_images = 0


scroll_result = client.scroll(
    collection_name=Config.COLLECTION_NAME,
    limit=10000,
    with_payload=True
)

for point in scroll_result[0]:
    categories.append(point.payload['category'])
    languages.append(point.payload['language'])
    if point.payload['has_processed_image']:
        with_images += 1

print("\n Database Statistics:")
print(f"   Total records: {len(scroll_result[0])}")
print(f"\n Categories:")
for cat, count in Counter(categories).most_common():
    print(f"   - {cat}: {count}")

print(f"\n Languages:")
for lang, count in Counter(languages).most_common():
    print(f"   - {lang}: {count}")

print(f"\n Images:")
print(f"   - With images: {with_images}")
print(f"   - Without images: {len(scroll_result[0]) - with_images}")


print(" ALL TESTS PASSED!")
