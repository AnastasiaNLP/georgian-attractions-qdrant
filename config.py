# CONFIGURATION (UPDATED WITH CLOUDINARY)
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path, override=True)

class Config:
    """Configuration for Georgian Attractions Qdrant project."""
    # QDRANT CLOUD
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    COLLECTION_NAME = 'georgian_attractions'
    # DATASET
    DATASET_NAME = os.getenv('DATASET_NAME', 'AIAnastasia/georgian-attractions')
    # MODEL
    EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    VECTOR_SIZE = 384
    DEVICE = 'cuda'  # or 'cpu'
    # PROCESSING
    BATCH_SIZE = 32
    # CLOUDINARY (NEW!)
    CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
    CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')

    @classmethod
    def validate(cls):
        """Validate that all required config is present."""
        # Qdrant required
        if not cls.QDRANT_URL:
            raise ValueError("QDRANT_URL not set in .env file")
        if not cls.QDRANT_API_KEY:
            raise ValueError("QDRANT_API_KEY not set in .env file")

        # Cloudinary optional (only if uploading images)
        if cls.CLOUDINARY_CLOUD_NAME:
            if not cls.CLOUDINARY_API_KEY:
                raise ValueError("CLOUDINARY_API_KEY not set in .env file")
            if not cls.CLOUDINARY_API_SECRET:
                raise ValueError("CLOUDINARY_API_SECRET not set in .env file")
            print(" Cloudinary credentials loaded")

        print(" Configuration validated")


# Validate on import
Config.validate()