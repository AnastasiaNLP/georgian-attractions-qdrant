# 🛠️ Setup Guide

Complete setup instructions for Georgian Attractions Vector Database.

## Prerequisites

- Python 3.8+
- pip
- 2GB RAM (for embeddings generation)
- Internet connection

## Step-by-Step Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/georgian-attractions-vector-db.git
cd georgian-attractions-vector-db
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Qdrant Cloud

1. Go to https://cloud.qdrant.io/
2. Create a free account
3. Create a new cluster (Free tier - 1GB)
4. Copy **Cluster URL** and **API Key**

### 5. (Optional) Setup Cloudinary

Only needed if you want to upload images:

1. Go to https://cloudinary.com/users/register/free
2. Create account (25GB free)
3. Copy **Cloud Name**, **API Key**, **API Secret**

### 6. Configure Environment

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
QDRANT_URL=https://your-cluster.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_actual_api_key
DATASET_NAME=AIAnastasia/georgian-attractions
```

### 7. Build Database

#### Option A: From Scratch (Full Setup)
```bash
# Step 1: Load dataset (test with 10 records first)
python3 tests/test_loader.py
# When prompted, type 'yes' to load full dataset

# Step 2: Generate embeddings
python3 tests/test_embeddings.py
# When prompted, type 'yes' for full dataset

# Step 3: Upload to Qdrant
python3 tests/test_upload.py
# When prompted: 'no' (don't recreate if exists)

# Step 4: (Optional) Upload images to Cloudinary
python3 tests/test_cloudinary_upload.py
# Takes 30-60 minutes

# Step 5: (Optional) Update Qdrant with image URLs
python3 tests/update_qdrant_images.py
```

#### Option B: Use Existing Database

If the database is already created in Qdrant Cloud:
```bash
# Just connect and use it!
python3 tests/test_qdrant_search.py
```

### 8. Verify Setup
```bash
python3 tests/test_full_rag.py
```

Should output search results with images!

## Troubleshooting

### Error: "QDRANT_URL not set"

Make sure `.env` file exists and has correct format (no spaces around `=`).

### Error: "403 Forbidden"

Check your Qdrant API key is correct and cluster is running.

### Error: "Out of Memory"

Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # Instead of 32
```

### Images fail to upload

Check Cloudinary credentials in `.env`.

### Slow embeddings

- Disable GPU if not available: set `DEVICE = 'cpu'` in `config.py`
- Or enable GPU for faster processing

## Next Steps

- See [USAGE.md](USAGE.md) for query examples
- See [CLOUDINARY.md](CLOUDINARY.md) for image details