"""
Upload images to Cloudinary and save URL mapping.
"""

import logging
import sys
from config import Config
from cloudinary_uploader import CloudinaryUploader

# Fix encoding for terminal
if sys.version_info[0] >= 3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_cloudinary_upload():
    """Upload images to Cloudinary."""

    print("TEST: CLOUDINARY IMAGE UPLOAD")

    # Check config
    if not Config.CLOUDINARY_CLOUD_NAME:
        print("\nERROR: Cloudinary not configured!")
        print("   Add to .env:")
        print("   CLOUDINARY_CLOUD_NAME=your_cloud_name")
        print("   CLOUDINARY_API_KEY=your_api_key")
        print("   CLOUDINARY_API_SECRET=your_api_secret")
        return

    print(f"\nCloudinary configured")
    print(f"   Cloud: {Config.CLOUDINARY_CLOUD_NAME}")

    # Confirm
    print("WARNING: This will upload ~1522 images to Cloudinary")
    print("   - Takes 30-60 minutes")
    print("   - Uses ~500MB-1GB of bandwidth")
    print("   - Free tier: 25GB storage, 25GB bandwidth/month")
    print("\nType 'yes' to continue or 'no' to cancel")

    try:
        response = input("\nContinue? (yes/no): ").strip().lower()
    except:
        # Fallback if input fails
        print("\nInput error. Please run again and type 'yes' or 'no'")
        return

    if response != 'yes':
        print("\nUpload cancelled")
        return

    # Create uploader
    uploader = CloudinaryUploader(
        cloud_name=Config.CLOUDINARY_CLOUD_NAME,
        api_key=Config.CLOUDINARY_API_KEY,
        api_secret=Config.CLOUDINARY_API_SECRET
    )

    # Upload images
    image_urls = uploader.upload_images(
        dataset_name=Config.DATASET_NAME,
        output_file='../data/image_urls.json'
    )

    print("\nImages uploaded to Cloudinary!")
    print(f"URLs saved to: image_urls.json")
    print(f"\nNext step: Update Qdrant with image URLs")
    print(f"   Run: python3 update_qdrant_images.py")


if __name__ == "__main__":
    try:
        test_cloudinary_upload()
    except Exception as e:
        print(f"\nUPLOAD FAILED: {e}")
        import traceback
        traceback.print_exc()