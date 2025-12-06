# ════════════════════════════════════════════════════════════════
# TEST: EMBEDDINGS GENERATOR
# ════════════════════════════════════════════════════════════════

"""
Тестирование создания embeddings для текстовых данных.
"""

import logging
from config import Config
from data_loader import GeorgianAttractionsDataLoader
from embeddings import EmbeddingsGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_embeddings():
    """Test creating embeddings for the dataset."""

    print("🧪 TEST: EMBEDDINGS GENERATOR")

    # Load data
    loader = GeorgianAttractionsDataLoader(Config.DATASET_NAME)

    # First test with small sample
    print("\n Testing with 10 records...")
    records = loader.load(sample_size=10)

    # Create embeddings generator
    embedder = EmbeddingsGenerator(
        model_name=Config.EMBEDDING_MODEL,
        device=Config.DEVICE
    )

    # Generate embeddings
    df = embedder.generate(records, batch_size=Config.BATCH_SIZE)

    print(f"\n Sample test passed!")
    print(f"   Records: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Vector size: {len(df.iloc[0]['embedding'])}")

    # Ask to continue with full dataset
    response = input("Continue with full dataset (1715 records)? (yes/no): ")

    if response.lower() == 'yes':
        print("\n Loading full dataset...")
        all_records = loader.load()

        print("\n Generating embeddings for all records...")
        df_full = embedder.generate(all_records, batch_size=Config.BATCH_SIZE)

        print(f"\n Full dataset processed!")
        print(f"   Total records: {len(df_full)}")

        # Save for next step
        import pickle
        with open('../data/processed_data.pkl', 'wb') as f:
            pickle.dump(df_full, f)

        print(f" Saved to 'processed_data.pkl'")
        print(f"\n Ready for Qdrant upload!")

        return df_full


if __name__ == "__main__":
    try:
        df = test_embeddings()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()