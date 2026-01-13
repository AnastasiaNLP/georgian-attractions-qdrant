# Update Qdrant with image urls
"""
Updates Qdrant records with Cloudinary image URLs.
"""

import json
import logging
from qdrant_client import QdrantClient
from tqdm.auto import tqdm
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def update_qdrant_with_images():
    """Update Qdrant records with image URLs from Cloudinary."""

    print("Update Qdrant records with image URLs")

    # load image URLs
    print("\n Loading image URLs...")
    try:
        with open('../data/image_urls.json', 'r') as f:
            image_urls = json.load(f)
        print(f" Loaded {len(image_urls)} image URLs")
    except FileNotFoundError:
        print(" image_urls.json not found!")
        print("   Run: python3 test_cloudinary_upload.py first")
        return

    # connect to Qdrant
    print("\n Connecting to Qdrant...")
    client = QdrantClient(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY
    )
    print(" Connected to Qdrant")

    # get current collection info
    collection_info = client.get_collection(Config.COLLECTION_NAME)
    print(f"   Collection: {Config.COLLECTION_NAME}")
    print(f"   Points: {collection_info.points_count}")

    # update records
    print("\n Updating records with image URLs...")

    updated = 0
    failed = 0

    for record_id, image_url in tqdm(image_urls.items(), desc="Updating"):
        try:
            # update payload with image_url
            client.set_payload(
                collection_name=Config.COLLECTION_NAME,
                payload={"image_url": image_url},
                points=[int(record_id)]
            )
            updated += 1

        except Exception as e:
            logger.error(f"Failed to update {record_id}: {e}")
            failed += 1

    print(f"\n Update cpmleted!")
    print(f"   Updated: {updated}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {updated/(updated+failed)*100:.1f}%")

    # verify with sample
    print("\n Verifying update...")
    sample_id = list(image_urls.keys())[0]

    result = client.retrieve(
        collection_name=Config.COLLECTION_NAME,
        ids=[int(sample_id)],
        with_payload=True
    )

    if result and result[0].payload.get('image_url'):
        print(f" Verification successful!")
        print(f"   Sample ID: {sample_id}")
        print(f"   Name: {result[0].payload['name']}")
        print(f"   Image URL: {result[0].payload['image_url'][:80]}...")
    else:
        print(f"  Verification failed - no image_url in payload")

    print(" Qdrant base updated with images urls")
    print("\n Your database now has:")
    print("   - Text data")
    print("   - Embeddings (vectors)")
    print("   - Image URLs from Cloudinary")
    print("\n Ready for RAG with images!")


if __name__ == "__main__":
    try:
        update_qdrant_with_images()
    except Exception as e:
        print(f"\n UPDATE FAILED: {e}")
        import traceback
        traceback.print_exc()
