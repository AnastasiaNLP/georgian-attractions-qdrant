# ════════════════════════════════════════════════════════════════
# DATA LOADER (UPDATED - TEXT ONLY)
# ════════════════════════════════════════════════════════════════

"""
Georgian Attractions Data Loader
Loads the dataset from HuggingFace and normalizes fields.
Images are NOT loaded to save memory - will be uploaded to Cloudinary separately.
"""

import logging
from typing import Any, Dict, List
from datasets import load_dataset
from tqdm.auto import tqdm
import gc

logger = logging.getLogger(__name__)


def safe_str(value: Any) -> str:
    """Safely convert any type to string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


class GeorgianAttractionsDataLoader:
    """
    Loads and normalizes the georgian-attractions dataset.
    Images are NOT loaded into memory - only metadata.

    Attributes
    ----------
    dataset_name : str
        HuggingFace dataset name
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        logger.info(f"Initialized DataLoader for: {dataset_name}")

    def load(self, sample_size: int = None) -> List[Dict[str, Any]]:
        """
        Load dataset and return normalized records WITHOUT images.

        Parameters
        ----------
        sample_size : int, optional
            Limit number of records for testing

        Returns
        -------
        List[Dict]
            Normalized records (text only, no images in memory)
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        print(f" LOADING DATASET (TEXT ONLY - NO IMAGES)")

        # Load dataset
        dataset = load_dataset(self.dataset_name, split='train')

        if sample_size:
            dataset = dataset.select(range(min(sample_size, len(dataset))))
            logger.info(f"Limited to sample: {len(dataset)} records")

        print(f" Dataset loaded: {len(dataset)} records")
        print(f" Normalizing records (skipping images to save RAM)...")

        # Normalize records
        records = []
        for i, rec in enumerate(tqdm(dataset, desc="Processing")):
            record = {
                # Core fields
                'id': safe_str(rec.get('id', str(i))),
                'name': safe_str(rec.get('name')),
                'description': safe_str(rec.get('description')),
                'location': safe_str(rec.get('location')),
                'category': safe_str(rec.get('category')),

                # Additional fields
                'tags': rec.get('tags', []),
                'language': safe_str(rec.get('language', 'en')).upper(),

                # Image metadata (but NOT the image data itself!)
                'photo_name': safe_str(rec.get('photo_name', '')),
                'photo_author': safe_str(rec.get('photo_author', '')),
                'license': safe_str(rec.get('license', '')),

                # Image flags (will be updated with Cloudinary URLs later)
                'has_processed_image': bool(rec.get('image')),
                'image_url': None,

                # For compatibility - explicitly NOT loading image
                'image': None
            }
            records.append(record)

        # Clean memory
        del dataset
        gc.collect()

        logger.info(f" Normalized {len(records)} records (text only)")
        print(f" Normalized {len(records)} records")
        print(f" Memory saved by NOT loading {sum(1 for r in records if r['has_processed_image'])} images")

        return records