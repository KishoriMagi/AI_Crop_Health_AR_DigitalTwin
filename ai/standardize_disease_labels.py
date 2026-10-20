import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"


FILES = {
    "train": DATASET_DIR / "disease_train.csv",
    "val": DATASET_DIR / "disease_val.csv",
    "test": DATASET_DIR / "disease_test.csv"
}


OUTPUT_FILES = {
    "train": DATASET_DIR / "disease_train_clean.csv",
    "val": DATASET_DIR / "disease_val_clean.csv",
    "test": DATASET_DIR / "disease_test_clean.csv"
}


# =========================================================
# LABEL MAPPING
# =========================================================

LABEL_MAP = {

    # Apple
    "Apple Scab Leaf": "Apple Scab",
    "Apple scab": "Apple Scab",

    "Apple rust leaf": "Apple Rust",
    "Cedar apple rust": "Apple Rust",

    # Healthy
    "Healthy": "Healthy",
    "healthy": "Healthy",

    # Pepper
    "Bacterial spot": "Bacterial Spot",
    "Bell_pepper leaf spot": "Bacterial Spot",

    # Rice
    "Bacterial Leaf Blight": "Bacterial Leaf Blight",
    "Bacterial Leaf Streak": "Bacterial Leaf Streak",
    "Bacterial Panicle Blight": "Bacterial Panicle Blight",
    "Blast": "Rice Blast",
    "Brown Spot": "Rice Brown Spot",
    "Tungro": "Rice Tungro",

    # Maize
    "Cercospora leaf spot Gray leaf spot": "Maize Gray Leaf Spot",
    "Common rust": "Maize Common Rust",
    "Corn Gray leaf spot": "Maize Gray Leaf Spot",
    "Corn leaf blight": "Maize Leaf Blight",
    "Corn rust leaf": "Maize Common Rust",
    "Northern Leaf Blight": "Maize Northern Leaf Blight",

    # Potato
    "Potato leaf early blight": "Potato Early Blight",
    "Potato leaf late blight": "Potato Late Blight",

    # Tomato
    "Tomato Early blight leaf": "Tomato Early Blight",
    "Tomato Septoria leaf spot": "Tomato Septoria Leaf Spot",
    "Tomato Yellow Leaf Curl Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato leaf bacterial spot": "Tomato Bacterial Spot",
    "Tomato leaf late blight": "Tomato Late Blight",
    "Tomato leaf mosaic virus": "Tomato Mosaic Virus",
    "Tomato mosaic virus": "Tomato Mosaic Virus",
    "Tomato leaf yellow virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato mold leaf": "Tomato Leaf Mold",

    # General disease names
    "Black rot": "Black Rot",
    "Downy Mildew": "Downy Mildew",
    "Early blight": "Early Blight",
    "Esca (Black Measles)": "Esca",
    "Haunglongbing (Citrus greening)": "Citrus Greening",
    "Late blight": "Late Blight",
    "Leaf Mold": "Leaf Mold",
    "Leaf blight (Isariopsis Leaf Spot)": "Leaf Blight",
    "Leaf scorch": "Leaf Scorch",
    "Powdery mildew": "Powdery Mildew",
    "Septoria leaf spot": "Septoria Leaf Spot",
    "Squash Powdery mildew leaf": "Squash Powdery Mildew",
    "Target Spot": "Target Spot"
}


print("=" * 65)
print("STANDARDIZING DISEASE LABELS")
print("=" * 65)


for split, input_file in FILES.items():

    print(f"\nProcessing {split}...")

    df = pd.read_csv(input_file)

    before = df["condition"].nunique()

    unknown_labels = set(df["condition"]) - set(LABEL_MAP)

    if unknown_labels:
        print("\nWARNING: Labels without mapping:")

        for label in sorted(unknown_labels):
            print(f"  {label}")

    df["condition"] = df["condition"].map(
        LABEL_MAP
    ).fillna(df["condition"])

    after = df["condition"].nunique()

    df.to_csv(
        OUTPUT_FILES[split],
        index=False
    )

    print(f"Classes before: {before}")
    print(f"Classes after:  {after}")
    print(f"Records:        {len(df):,}")


print("\n" + "=" * 65)
print("FINAL CONDITION LIST")
print("=" * 65)

combined = pd.concat(
    [
        pd.read_csv(OUTPUT_FILES["train"]),
        pd.read_csv(OUTPUT_FILES["val"]),
        pd.read_csv(OUTPUT_FILES["test"])
    ],
    ignore_index=True
)

conditions = sorted(
    combined["condition"].unique()
)

for i, condition in enumerate(conditions, 1):
    count = (
        combined["condition"] == condition
    ).sum()

    print(
        f"{i:3}. {condition:<40} {count:,}"
    )


print("\nTotal unique conditions:")
print(len(conditions))

print("\nFiles created:")

for file in OUTPUT_FILES.values():
    print(file)

print("\n" + "=" * 65)
print("LABEL STANDARDIZATION COMPLETED")
print("=" * 65)