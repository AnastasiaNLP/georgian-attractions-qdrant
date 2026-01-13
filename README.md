# 🇬🇪 Georgian Attractions Vector Database

A production-ready vector database for semantic search of 1,715 Georgian tourist attractions with multilingual support and cloud-hosted images.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red.svg)](https://qdrant.tech/)
[![Cloudinary](https://img.shields.io/badge/Images-Cloudinary-blue.svg)](https://cloudinary.com/)

## What is this?

This project provides a **complete vector database** for Georgian tourist attractions, ready to be used in RAG (Retrieval-Augmented Generation) systems, chatbots, or search applications.

### What's Included

- **Vector Database**: 1,715 attractions with semantic embeddings
- **Image CDN**: 1,520 photos hosted on Cloudinary
- **Metadata**: Names, descriptions, locations, categories, tags
- **Multilingual**: English and Russian
- **Production Ready**: Fully tested and documented

## Use Cases

This database is ready to use for:

1. **RAG Systems**: Use with any LLM to build tourist chatbots
2. **Search Engines**: Semantic search for attractions
3. **Recommendation Systems**: Find similar attractions
4. **Mobile Apps**: Backend for travel apps
5. **Research**: Georgian tourism analytics

## Architecture
```
HuggingFace Dataset (1,715 attractions)
    ↓
Data Processing (text normalization)
    ↓
Embeddings (384D vectors)
    ↓
Qdrant Vector Database
    ↓
Cloudinary Image CDN (1,520 photos)
    ↓
Ready for Your Application!
```

## Database Contents

### Statistics
- **Total Records**: 1,715 attractions
- **With Images**: 1,520 (88.6%)
- **Vector Dimensions**: 384D
- **Languages**: EN (857), RU (858)

### Data Fields
Each record contains:
- `id`: Unique identifier
- `name`: Attraction name
- `description`: Full description
- `location`: City/region
- `category`: Type (church, museum, beach, etc.)
- `tags`: Keywords
- `language`: EN/RU
- `image_url`: Cloudinary URL
- `embedding`: 384D vector

### Categories
- Churches & Cathedrals
- Museums & Galleries
- Beaches & Resorts
- Mountains & Nature
- Castles & Fortresses
- Historical Sites
- And more...

## Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/georgian-attractions-vector-db.git
cd georgian-attractions-vector-db
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup Credentials

Create `.env`:
```bash
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
DATASET_NAME=AIAnastasia/georgian-attractions
```

### 3. Build Database (Optional - if starting from scratch)
```bash
# Load and process dataset
python3 tests/test_loader.py

# Generate embeddings
python3 tests/test_embeddings.py

# Upload to Qdrant
python3 tests/test_upload.py
```

### 4. Use the Database
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Connect
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Search
query = "ancient churches near Tbilisi"
query_vector = embedder.encode(query).tolist()

results = client.search(
    collection_name="georgian_attractions",
    query_vector=query_vector,
    limit=5
)

# Results include:
for result in results:
    print(result.payload['name'])          # Name
    print(result.payload['description'])   # Description
    print(result.payload['image_url'])     # Image URL
    print(result.score)                    # Relevance score
```

## Project Structure
```
georgian-attractions-vector-db/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── .env.example                # Environment template
│
├── config.py                    # Configuration
├── data_loader.py              # Dataset loader
├── embeddings.py               # Embedding generator
├── qdrant_uploader.py          # Qdrant uploader
├── cloudinary_uploader.py      # Image uploader
│
├── tests/                      # Setup & test scripts
│   ├── test_loader.py
│   ├── test_embeddings.py
│   ├── test_upload.py
│   ├── test_cloudinary_upload.py
│   ├── test_qdrant_search.py
│   └── test_full_rag.py
│
└── docs/                       # Documentation
    ├── SETUP.md
    ├── CLOUDINARY.md
    └── USAGE.md
```

## Technical Details

- **Vector DB**: Qdrant Cloud (free 1GB tier)
- **Embeddings**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Image CDN**: Cloudinary (free 25GB tier)
- **Dataset**: HuggingFace Datasets
- **Vector Size**: 384 dimensions
- **Distance Metric**: Cosine similarity

## Documentation

- [Setup Guide](docs/SETUP.md) - Detailed setup instructions
- [Cloudinary Guide](docs/CLOUDINARY.md) - Image upload process
- [Usage Examples](docs/USAGE.md) - How to query the database

## Roadmap

Future additions:
- [ ] Hybrid search (BM25 + vectors)
- [ ] Category filters
- [ ] Geolocation search
- [ ] Additional language support
- [ ] Data enrichment

## Contributing

PRs welcome! Especially for:
- Additional data enrichment
- Performance optimizations
- Documentation improvements
- Bug fixes

## License

MIT License

## Acknowledgments

- **Dataset**: [AIAnastasia/georgian-attractions](https://huggingface.co/datasets/AIAnastasia/georgian-attractions)
- **Vector DB**: [Qdrant](https://qdrant.tech/)
- **Embeddings**: [sentence-transformers](https://www.sbert.net/)
- **Images**: [Cloudinary](https://cloudinary.com/)

## Contact

- GitHub: [@AIAnastasia](https://github.com/AnastasiaNLP)
- HuggingFace: [@AIAnastasia](https://huggingface.co/AIAnastasia)

---

**Production-ready vector database for Georgian attractions** 🇬🇪
