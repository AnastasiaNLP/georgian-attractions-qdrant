# TEST: DATA LOADER
"""
Тестирование загрузки датасета.
Загружает 10 записей для проверки.
"""

import logging
from config import Config
from data_loader import GeorgianAttractionsDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_data_loader():
    """Test loading a small sample of the dataset."""
    print(" TEST: DATA LOADER")
    # Create loader
    loader = GeorgianAttractionsDataLoader(Config.DATASET_NAME)

    # Load small sample
    print("\n Loading sample (10 records)...")
    records = loader.load(sample_size=10)

    # Display results
    print(f"\n Sample loaded: {len(records)} records")

    print("\ First record:")
    first = records[0]
    for key, value in first.items():
        if key == 'description':
            print(f"   {key}: {str(value)[:100]}...")
        elif key == 'image':
            if value:
                img_type = "URL" if isinstance(value, str) and value.startswith('http') else "Base64"
                print(f"   {key}: [{img_type} data present]")
            else:
                print(f"   {key}: None")
        else:
            print(f"   {key}: {value}")

    # Statistics
    print(f"\n Sample statistics:")
    with_images = sum(1 for r in records if r['has_processed_image'])
    print(f"   Records with images: {with_images}/{len(records)}")

    languages = {}
    for r in records:
        lang = r['language']
        languages[lang] = languages.get(lang, 0) + 1
    print(f"   Languages: {languages}")

    print("\n TEST PASSED!")
    return records


if __name__ == "__main__":
    try:
        records = test_data_loader()

        # Ask to continue
        print("\n" + "="*70)
        response = input("Continue with full dataset? (yes/no): ")

        if response.lower() == 'yes':
            print("\n Loading FULL dataset...")
            loader = GeorgianAttractionsDataLoader(Config.DATASET_NAME)
            all_records = loader.load()
            print(f" Full dataset loaded: {len(all_records)} records")

    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()