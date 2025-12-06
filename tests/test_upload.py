# ════════════════════════════════════════════════════════════════
# TEST: QDRANT UPLOAD
# ════════════════════════════════════════════════════════════════

"""
Upload processed data to Qdrant Cloud.
"""

import logging
import pickle
from config import Config
from qdrant_uploader import QdrantUploader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_upload():
    """Upload data to Qdrant."""

    print(" TEST: QDRANT UPLOAD")

    # Load processed data
    print("\n Loading processed data...")
    with open('../data/processed_data.pkl', 'rb') as f:
        df = pickle.load(f)

    print(f" Loaded {len(df)} records")

    # Create uploader
    uploader = QdrantUploader(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
        collection_name=Config.COLLECTION_NAME,
        vector_size=Config.VECTOR_SIZE
    )

    # Create collection
    recreate = input("Recreate collection if exists? (yes/no): ")
    uploader.create_collection(recreate=(recreate.lower() == 'yes'))

    # Upload data
    uploader.upload_data(df, batch_size=100)

    print(" QDRANT DATABASE CREATED SUCCESSFULLY!")
    print(f"\n Summary:")
    print(f"   Collection: {Config.COLLECTION_NAME}")
    print(f"   Records: {len(df)}")
    print(f"   Vector size: {Config.VECTOR_SIZE}")
    print(f"   With images: {df['has_processed_image'].sum()}")
    print(f"\n Database ready for RAG!")


if __name__ == "__main__":
    try:
        test_upload()
    except Exception as e:
        print(f"\n❌ UPLOAD FAILED: {e}")
        import traceback
        traceback.print_exc()