"""
Uploads images from HuggingFace dataset to Cloudinary.
"""

import logging
import cloudinary
import cloudinary.uploader
from datasets import load_dataset
from tqdm.auto import tqdm
import json
from pathlib import Path
from io import BytesIO

logger = logging.getLogger(__name__)


class CloudinaryUploader:
    """
    Uploads images to Cloudinary and returns URL mapping.
    """

    def __init__(self, cloud_name: str, api_key: str, api_secret: str):
        self.cloud_name = cloud_name

        # configure Cloudinary
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret
        )

        logger.info("Cloudinary configured")
        print(f"  Cloudinary uploader")
        print(f"   Cloud: {cloud_name}")

    def upload_images(self, dataset_name: str, output_file: str = 'image_urls.json'):
        """
        Upload all images from dataset to Cloudinary.
        """
        print(f"\n Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split='train')

        print(f" Dataset loaded: {len(dataset)} records")
        print(f"\n Uploading images to Cloudinary...")

        image_urls = {}
        failed = []

        for i, rec in enumerate(tqdm(dataset, desc="Uploading")):
            rec_id = rec.get('id', str(i))
            image = rec.get('image')

            if not image:
                continue

            try:
                # converting a PIL Image to Bytes
                if hasattr(image, 'save'):  # Это PIL Image
                    buffer = BytesIO()
                    image.save(buffer, format='JPEG')
                    buffer.seek(0)
                    upload_data = buffer
                else:
                    # If it is a string (Base64 or URL)
                    upload_data = image

                # upload to Cloudinary
                result = cloudinary.uploader.upload(
                    upload_data,
                    public_id=f"georgian_attractions/{rec_id}",
                    folder="georgian_attractions",
                    resource_type="auto"
                )

                # save URL
                image_urls[str(rec_id)] = result['secure_url']

                # progress
                if (i + 1) % 50 == 0:
                    uploaded = len(image_urls)
                    print(f"\n   Progress: {uploaded} images uploaded")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to upload {rec_id}: {error_msg}")
                failed.append((rec_id, error_msg))

                # show the first 5 errors
                if len(failed) <= 5:
                    print(f"\n    Error for {rec_id}: {error_msg}")

        # save mapping
        print(f"\n Saving URL mapping to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(image_urls, f, indent=2)

        print(f" Upload complete!")
        print(f"   Total images: {len(dataset)}")
        print(f"   Uploaded: {len(image_urls)}")
        print(f"   Failed: {len(failed)}")
        print(f"   Success rate: {len(image_urls)/(len(dataset) if len(dataset) > 0 else 1)*100:.1f}%")

        if failed:
            print(f"\n  First 5 failed uploads:")
            for rec_id, error in failed[:5]:
                print(f"   - {rec_id}: {error[:100]}")

        return image_urls
