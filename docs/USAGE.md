# 📖 Usage Guide

How to query and use the Georgian Attractions Vector Database.

## Basic Search
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Setup
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Search
query = "beautiful beaches"
query_vector = embedder.encode(query).tolist()

results = client.search(
    collection_name="georgian_attractions",
    query_vector=query_vector,
    limit=5
)

# Display
for result in results:
    print(f"Name: {result.payload['name']}")
    print(f"Score: {result.score}")
    print(f"Description: {result.payload['description'][:100]}...")
    print()
```

## Search Examples

### 1. By Category
```python
query = "ancient churches"
# Returns: Svetitskhoveli, Jvari, etc.
```

### 2. By Location
```python
query = "attractions in Batumi"
# Returns: Batumi beaches, botanical garden, etc.
```

### 3. By Activity
```python
query = "hiking and nature"
# Returns: National parks, mountains, trails
```

### 4. Multilingual
```python
query = "музеи в Тбилиси"  # Russian
# Returns: Museums in Tbilisi
```

## Advanced Filtering

### Filter by Category
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="georgian_attractions",
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="Church")
            )
        ]
    ),
    limit=5
)
```

### Filter by Language
```python
results = client.search(
    collection_name="georgian_attractions",
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="language",
                match=MatchValue(value="EN")
            )
        ]
    ),
    limit=5
)
```

### Filter by Location
```python
results = client.search(
    collection_name="georgian_attractions",
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="location",
                match=MatchValue(value="Tbilisi")
            )
        ]
    ),
    limit=5
)
```

## Get All Records
```python
# Scroll through all records
records, next_page = client.scroll(
    collection_name="georgian_attractions",
    limit=100,
    with_payload=True,
    with_vectors=False
)

for record in records:
    print(record.payload['name'])
```

## Get Specific Record
```python
# Get by ID
result = client.retrieve(
    collection_name="georgian_attractions",
    ids=[1, 2, 3],
    with_payload=True
)

for point in result:
    print(point.payload)
```

## Statistics
```python
# Collection info
info = client.get_collection("georgian_attractions")
print(f"Total points: {info.points_count}")
print(f"Vector size: {info.config.params.vectors.size}")
```

## Display Images
```python
from PIL import Image
import requests
from io import BytesIO

# Get result with image
result = results[0]
image_url = result.payload['image_url']

if image_url:
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img.show()
```

## Performance Tips

### 1. Batch Queries

For multiple queries, batch them:
```python
queries = ["beaches", "museums", "mountains"]
query_vectors = embedder.encode(queries)

for i, vector in enumerate(query_vectors):
    results = client.search(
        collection_name="georgian_attractions",
        query_vector=vector.tolist(),
        limit=3
    )
    print(f"\nQuery: {queries[i]}")
    print(results)
```

### 2. Limit Results

Only get what you need:
```python
results = client.search(..., limit=5)  # Not 100!
```

### 3. Disable Vectors

If you don't need vectors in response:
```python
results = client.search(..., with_vectors=False)
```

## Common Use Cases

### Tourism Chatbot
```python
user_question = "What are the best places to visit in Mtskheta?"
results = search(user_question)

# Feed to LLM
context = "\n".join([r.payload['description'] for r in results])
answer = llm.generate(question=user_question, context=context)
```

### Recommendation System
```python
# User liked this attraction
attraction_id = 42
attraction = client.retrieve("georgian_attractions", ids=[attraction_id])[0]

# Find similar
similar = client.search(
    collection_name="georgian_attractions",
    query_vector=attraction.vector,
    limit=5
)
```

### Analytics
```python
# All churches
churches = client.scroll(
    collection_name="georgian_attractions",
    scroll_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="Church"))]
    ),
    limit=1000
)

print(f"Total churches: {len(churches[0])}")
```