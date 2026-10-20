from pathlib import Path
import csv
import hashlib
from collections import Counter


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"

INPUT_FILE = DATASET_DIR / "master_dataset.csv"

CLEAN_FILE = DATASET_DIR / "clean_master_dataset.csv"
DUPLICATE_FILE = DATASET_DIR / "duplicate_images.csv"
SUMMARY_FILE = DATASET_DIR / "clean_class_summary.csv"


# ============================================================
# CROP NORMALIZATION
# ============================================================

CROP_MAP = {
    "Cherry": "Cherry",
    "Cherry Including Sour": "Cherry",

    "Corn": "Maize",
    "Corn Maize": "Maize",
    "Maize": "Maize",

    "Pepper, Bell": "Bell Pepper",
    "Bell Pepper": "Bell Pepper",

    "Soybean": "Soybean",

    "Rice": "Rice",
    "Paddy": "Rice",

    "Ash Gourd": "Ash Gourd",
    "Bitter Gourd": "Bitter Gourd",
    "Bottle Gourd": "Bottle Gourd",
    "Cucumber": "Cucumber",
    "Eggplant": "Eggplant",
    "Ridge Gourd": "Ridge Gourd",
    "Snake Gourd": "Snake Gourd",

    "Apple": "Apple",
    "Blueberry": "Blueberry",
    "Banana": "Banana",
    "Grape": "Grape",
    "Orange": "Orange",
    "Peach": "Peach",
    "Potato": "Potato",
    "Raspberry": "Raspberry",
    "Squash": "Squash",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato"
}


# ============================================================
# CONDITION NORMALIZATION
# ============================================================

def normalize_condition(condition):

    text = condition.strip()

    replacements = {
        "Cercospora leaf spot Gray leaf spot":
            "Cercospora Leaf Spot / Gray Leaf Spot",

        "Common rust":
            "Common Rust",

        "Northern Leaf Blight":
            "Northern Leaf Blight",

        "Esca (Black Measles)":
            "Esca / Black Measles",

        "Leaf blight (Isariopsis Leaf Spot)":
            "Leaf Blight / Isariopsis Leaf Spot",

        "Haunglongbing (Citrus greening)":
            "Huanglongbing / Citrus Greening",

        "Bacterial spot":
            "Bacterial Spot",

        "Early blight":
            "Early Blight",

        "Late blight":
            "Late Blight",

        "Leaf Mold":
            "Leaf Mold",

        "Septoria leaf spot":
            "Septoria Leaf Spot",

        "Spider mites Two-spotted spider mite":
            "Two-Spotted Spider Mite",

        "Target Spot":
            "Target Spot",

        "Tomato Yellow Leaf Curl Virus":
            "Tomato Yellow Leaf Curl Virus",

        "Tomato mosaic virus":
            "Tomato Mosaic Virus",

        "Powdery mildew":
            "Powdery Mildew",

        "Leaf scorch":
            "Leaf Scorch",

        "Bacterial Leaf Blight":
            "Bacterial Leaf Blight",

        "Bacterial Leaf Streak":
            "Bacterial Leaf Streak",

        "Bacterial Panicle Blight":
            "Bacterial Panicle Blight",

        "Brown Spot":
            "Brown Spot",

        "Downy Mildew":
            "Downy Mildew",

        "Nitrogen Deficiency":
            "Nitrogen Deficiency",

        "Potassium Deficiency":
            "Potassium Deficiency",

        "Magnesium Deficiency":
            "Magnesium Deficiency",

        "Phosphorus Deficiency":
            "Phosphorus Deficiency",

        "Healthy":
            "Healthy"
    }

    return replacements.get(text, text)


# ============================================================
# IMAGE HASH
# ============================================================

def calculate_hash(file_path):

    sha256 = hashlib.sha256()

    try:

        with open(file_path, "rb") as file:

            while True:

                data = file.read(1024 * 1024)

                if not data:
                    break

                sha256.update(data)

        return sha256.hexdigest()

    except Exception:

        return None


# ============================================================
# LOAD MASTER DATASET
# ============================================================

def load_master_dataset():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    return rows


# ============================================================
# CLEAN LABELS
# ============================================================

def clean_labels(rows):

    for row in rows:

        old_crop = row["crop"]

        row["crop"] = CROP_MAP.get(
            old_crop,
            old_crop
        )

        row["condition"] = normalize_condition(
            row["condition"]
        )

        # Standardize health category
        if row["health_type"] == "Pest/Symptom":

            row["health_type"] = "Pest"

        # Unlabeled data should never be treated as trainable
        if row["label_status"] == "unlabeled":

            row["training_eligible"] = "No"

        else:

            row["training_eligible"] = "Yes"

        # Augmented images will not be used in the initial
        # validation/test split.
        if row["augmentation_status"] == "augmented":

            row["split_eligible"] = "Training Only"

        else:

            row["split_eligible"] = "Train/Validation/Test"

    return rows


# ============================================================
# FIND EXACT DUPLICATES
# ============================================================

def find_duplicates(rows):

    print("\nCalculating image hashes...")
    print("This may take some time for 86,000+ images.\n")

    hash_map = {}

    duplicate_rows = []

    for index, row in enumerate(rows, start=1):

        image_path = PROJECT_ROOT / row["image_path"]

        image_hash = calculate_hash(image_path)

        row["image_hash"] = image_hash

        if image_hash is None:

            row["file_status"] = "Missing/Unreadable"

            continue

        row["file_status"] = "OK"

        if image_hash in hash_map:

            row["duplicate_of"] = hash_map[image_hash]

            duplicate_rows.append({
                "image_path": row["image_path"],
                "duplicate_of": hash_map[image_hash],
                "image_hash": image_hash
            })

        else:

            hash_map[image_hash] = row["image_path"]

            row["duplicate_of"] = ""

        if index % 5000 == 0:

            print(
                f"Processed {index:,} / {len(rows):,}"
            )

    return rows, duplicate_rows


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

def save_clean_dataset(rows):

    fieldnames = [
        "image_path",
        "dataset",
        "crop",
        "health_type",
        "condition",
        "original_class",
        "source_split",
        "label_status",
        "augmentation_status",
        "multi_label",
        "representation",
        "training_eligible",
        "split_eligible",
        "image_hash",
        "file_status",
        "duplicate_of"
    ]

    with open(
        CLEAN_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# SAVE DUPLICATES
# ============================================================

def save_duplicates(duplicates):

    with open(
        DUPLICATE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "duplicate_of",
                "image_hash"
            ]
        )

        writer.writeheader()

        writer.writerows(duplicates)


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(rows):

    counter = Counter()

    for row in rows:

        if row["file_status"] != "OK":
            continue

        key = (
            row["dataset"],
            row["crop"],
            row["health_type"],
            row["condition"]
        )

        counter[key] += 1

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "dataset",
            "crop",
            "health_type",
            "condition",
            "image_count"
        ])

        for key, count in sorted(counter.items()):

            writer.writerow([
                key[0],
                key[1],
                key[2],
                key[3],
                count
            ])


# ============================================================
# PRINT FINAL REPORT
# ============================================================

def print_report(rows, duplicates):

    print("\n")
    print("=" * 55)
    print("CLEAN DATASET REPORT")
    print("=" * 55)

    total = len(rows)

    valid = sum(
        1
        for row in rows
        if row["file_status"] == "OK"
    )

    unreadable = sum(
        1
        for row in rows
        if row["file_status"] != "OK"
    )

    training = sum(
        1
        for row in rows
        if row["training_eligible"] == "Yes"
        and row["file_status"] == "OK"
    )

    unlabeled = sum(
        1
        for row in rows
        if row["training_eligible"] == "No"
    )

    augmented = sum(
        1
        for row in rows
        if row["augmentation_status"] == "augmented"
    )

    print(f"Total records       : {total:,}")
    print(f"Valid images        : {valid:,}")
    print(f"Unreadable images   : {unreadable:,}")
    print(f"Training eligible   : {training:,}")
    print(f"Unlabeled images    : {unlabeled:,}")
    print(f"Augmented images    : {augmented:,}")
    print(f"Exact duplicates    : {len(duplicates):,}")

    print("\n")
    print("NORMALIZED CROPS")
    print("-" * 55)

    crop_counter = Counter(
        row["crop"]
        for row in rows
        if row["file_status"] == "OK"
    )

    for crop, count in sorted(crop_counter.items()):

        print(
            f"{crop:<30} {count:>8,}"
        )

    print("\n")
    print("HEALTH CATEGORIES")
    print("-" * 55)

    health_counter = Counter(
        row["health_type"]
        for row in rows
        if row["file_status"] == "OK"
    )

    for category, count in sorted(
        health_counter.items()
    ):

        print(
            f"{category:<30} {count:>8,}"
        )

    print("\n")
    print("=" * 55)
    print("FILES CREATED")
    print("=" * 55)

    print(f"Clean dataset : {CLEAN_FILE}")
    print(f"Duplicates    : {DUPLICATE_FILE}")
    print(f"Summary       : {SUMMARY_FILE}")

    print("\nCleaning completed successfully.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 55)
    print("AI CROP HEALTH DATASET CLEANER")
    print("=" * 55)

    if not INPUT_FILE.exists():

        print(
            "\nERROR: master_dataset.csv was not found."
        )

        return

    rows = load_master_dataset()

    print(
        f"\nLoaded {len(rows):,} records."
    )

    rows = clean_labels(rows)

    rows, duplicates = find_duplicates(rows)

    save_clean_dataset(rows)

    save_duplicates(duplicates)

    save_summary(rows)

    print_report(rows, duplicates)


if __name__ == "__main__":
    main()