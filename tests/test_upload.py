
# TEST: Qdarnt upload


"""
Upload processed data to Qdrant Cloud.
"""

import logging
import pickle
from config import Config
from qdrant_uploader import QdrantUploader

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_upload():
    """Upload data to Qdrant."""

    print(" Test: Qdrant")

    # load processed data
    print("\n Loading processed data...")
    with open('../data/processed_data.pkl', 'rb') as f:
        df = pickle.load(f)

    print(f" Loaded {len(df)} records")

    # create uploader
    uploader = QdrantUploader(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
        collection_name=Config.COLLECTION_NAME,
        vector_size=Config.VECTOR_SIZE
    )

    # create collection
    recreate = input("Recreate collection if exists? (yes/no): ")
    uploader.create_collection(recreate=(recreate.lower() == 'yes'))

    # upload data
    uploader.upload_data(df, batch_size=100)

    print(" Qdrant base created successfully!")
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
        print(f"\nUPLOAD FAILED: {e}")
        import traceback
        traceback.print_exc()
