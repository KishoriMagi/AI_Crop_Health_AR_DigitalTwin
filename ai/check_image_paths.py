import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

FILES = [
    BASE_DIR / "datasets" / "disease_train_clean.csv",
    BASE_DIR / "datasets" / "disease_val_clean.csv",
    BASE_DIR / "datasets" / "disease_test_clean.csv"
]


print("=" * 65)
print("CHECKING IMAGE PATHS")
print("=" * 65)


total_all = 0
found_all = 0
missing_all = 0


for csv_file in FILES:

    print(f"\nChecking: {csv_file.name}")

    df = pd.read_csv(csv_file)

    total = len(df)
    found = 0
    missing = []

    for image_path in df["image_path"]:

        path = Path(str(image_path))

        if not path.is_absolute():
            path = BASE_DIR / path

        if path.exists():
            found += 1
        else:
            missing.append(str(path))

    missing_count = len(missing)

    total_all += total
    found_all += found
    missing_all += missing_count

    print(f"Total images:   {total:,}")
    print(f"Images found:   {found:,}")
    print(f"Images missing: {missing_count:,}")

    if missing:

        print("\nFirst 10 missing paths:")

        for path in missing[:10]:
            print(path)


print("\n" + "=" * 65)
print("OVERALL RESULT")
print("=" * 65)

print(f"\nTotal images checked: {total_all:,}")
print(f"Images found:         {found_all:,}")
print(f"Images missing:       {missing_all:,}")

if missing_all == 0:
    print("\n✓ ALL IMAGE PATHS ARE VALID.")
    print("✓ Ready for model training.")
else:
    print("\n⚠ Some image paths are missing.")
    print("Do NOT start training yet.")

print("=" * 65)