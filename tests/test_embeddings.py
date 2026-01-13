# TEST: embeddings generator 

"""
Testing the creation of embeddings for text data.
"""

import logging
from config import Config
from data_loader import GeorgianAttractionsDataLoader
from embeddings import EmbeddingsGenerator

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_embeddings():
    """Test creating embeddings for the dataset."""

    print(" TEST: embedding generator")

    # load data
    loader = GeorgianAttractionsDataLoader(Config.DATASET_NAME)

    # first test with small sample
    print("\n Testing with 10 records...")
    records = loader.load(sample_size=10)

    # create embeddings generator
    embedder = EmbeddingsGenerator(
        model_name=Config.EMBEDDING_MODEL,
        device=Config.DEVICE
    )

    # generate embeddings
    df = embedder.generate(records, batch_size=Config.BATCH_SIZE)

    print(f"\n Sample test passed")
    print(f"   Records: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Vector size: {len(df.iloc[0]['embedding'])}")

    # ask to continue with full dataset
    response = input("Continue with full dataset (1715 records)? (yes/no): ")

    if response.lower() == 'yes':
        print("\n Loading full dataset")
        all_records = loader.load()

        print("\n Generating embeddings for all records")
        df_full = embedder.generate(all_records, batch_size=Config.BATCH_SIZE)

        print(f"\n Full dataset processed")
        print(f"   Total records: {len(df_full)}")

        # save for next step
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
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
