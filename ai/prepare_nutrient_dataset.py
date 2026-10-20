"""
prepare_nutrient_dataset.py

Purpose:
1. Inspect Banana Nutrient dataset
2. Inspect Maize Nutrient dataset
3. Inspect OLID-I dataset
4. Standardize Banana and Maize nutrient labels
5. Create nutrient CSV files
6. Check missing images and duplicate paths

This script DOES NOT train a model.
"""

import os
import pandas as pd
from collections import Counter


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(PROJECT_ROOT, "datasets", "raw")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets")


BANANA_DIR = os.path.join(RAW_DIR, "Banana_Nutrient")
MAIZE_DIR = os.path.join(RAW_DIR, "Maize_Nutrient")
OLID_DIR = os.path.join(RAW_DIR, "OLID")


# ============================================================
# 2. SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def is_image_file(filename):
    """
    Check whether a file is an image.
    """
    extension = os.path.splitext(filename)[1].lower()
    return extension in IMAGE_EXTENSIONS


def collect_images(root_directory):
    """
    Recursively collect all image paths from a directory.
    """

    image_paths = []

    if not os.path.exists(root_directory):
        print(f"WARNING: Directory does not exist: {root_directory}")
        return image_paths

    for root, dirs, files in os.walk(root_directory):

        for filename in files:

            if is_image_file(filename):

                full_path = os.path.join(root, filename)

                image_paths.append(
                    os.path.abspath(full_path)
                )

    return image_paths


# ============================================================
# 4. BANANA LABEL STANDARDIZATION
# ============================================================

def standardize_banana_label(folder_name):
    """
    Convert Banana Nutrient folder names into clean labels.
    """

    label = folder_name.strip().lower()

    mapping = {

        "boron": "Boron Deficiency",
        "boron deficiency": "Boron Deficiency",

        "calcium": "Calcium Deficiency",
        "calcium deficiency": "Calcium Deficiency",

        "iron": "Iron Deficiency",
        "iron deficiency": "Iron Deficiency",

        "magnesium": "Magnesium Deficiency",
        "magnesium deficiency": "Magnesium Deficiency",

        "manganese": "Manganese Deficiency",
        "manganese deficiency": "Manganese Deficiency",

        "potassium": "Potassium Deficiency",
        "potassium deficiency": "Potassium Deficiency",

        "sulphur": "Sulphur Deficiency",
        "sulfur": "Sulphur Deficiency",
        "sulphur deficiency": "Sulphur Deficiency",
        "sulfur deficiency": "Sulphur Deficiency",

        "zinc": "Zinc Deficiency",
        "zinc deficiency": "Zinc Deficiency",

        "healthy": "Healthy",

    }

    return mapping.get(label, folder_name.strip())


# ============================================================
# 5. MAIZE LABEL STANDARDIZATION
# ============================================================

def standardize_maize_label(folder_name):
    """
    Convert Maize Nutrient folder names into clean labels.
    """

    label = folder_name.strip().lower()

    mapping = {

        "helathy": "Healthy",
        "healthy": "Healthy",

        "magnessium": "Magnesium Deficiency",
        "magnesium": "Magnesium Deficiency",
        "magnesium deficiency": "Magnesium Deficiency",

        "nitrogen": "Nitrogen Deficiency",
        "nitrogen deficiency": "Nitrogen Deficiency",

        "phosphorous": "Phosphorus Deficiency",
        "phosphorus": "Phosphorus Deficiency",
        "phosphorous deficiency": "Phosphorus Deficiency",

        "pottasium": "Potassium Deficiency",
        "potassium": "Potassium Deficiency",
        "potassium deficiency": "Potassium Deficiency",

    }

    return mapping.get(label, folder_name.strip())


# ============================================================
# 6. INSPECT BANANA DATASET
# ============================================================

def process_banana_dataset():

    print("\n" + "=" * 70)
    print("BANANA NUTRIENT DATASET")
    print("=" * 70)

    print(f"Location: {BANANA_DIR}")

    if not os.path.exists(BANANA_DIR):

        print("ERROR: Banana Nutrient directory not found.")

        return pd.DataFrame()

    records = []

    for root, dirs, files in os.walk(BANANA_DIR):

        image_files = [
            f for f in files
            if is_image_file(f)
        ]

        if len(image_files) == 0:
            continue

        relative_path = os.path.relpath(
            root,
            BANANA_DIR
        )

        parts = relative_path.split(os.sep)

        if len(parts) == 0:
            continue

        folder_name = parts[-1]

        label = standardize_banana_label(
            folder_name
        )

        # Detect whether this looks like augmented data
        path_lower = root.lower()

        if "augment" in path_lower:
            image_type = "augmented"
        else:
            image_type = "original"

        for filename in image_files:

            full_path = os.path.join(
                root,
                filename
            )

            records.append({

                "image_path": os.path.abspath(
                    full_path
                ),

                "dataset": "Banana_Nutrient",

                "crop": "Banana",

                "label": label,

                "health_type": (
                    "Healthy"
                    if label == "Healthy"
                    else "Nutrient Deficiency"
                ),

                "image_type": image_type

            })

    df = pd.DataFrame(records)

    print(f"\nTotal Banana images found: {len(df)}")

    if len(df) > 0:

        print("\nBanana classes:")

        print(
            df["label"]
            .value_counts()
            .to_string()
        )

        print("\nOriginal vs Augmented:")

        print(
            df["image_type"]
            .value_counts()
            .to_string()
        )

    return df


# ============================================================
# 7. INSPECT MAIZE DATASET
# ============================================================

def process_maize_dataset():

    print("\n" + "=" * 70)
    print("MAIZE NUTRIENT DATASET")
    print("=" * 70)

    print(f"Location: {MAIZE_DIR}")

    if not os.path.exists(MAIZE_DIR):

        print("ERROR: Maize Nutrient directory not found.")

        return pd.DataFrame()

    records = []

    for root, dirs, files in os.walk(MAIZE_DIR):

        image_files = [
            f for f in files
            if is_image_file(f)
        ]

        if len(image_files) == 0:
            continue

        relative_path = os.path.relpath(
            root,
            MAIZE_DIR
        )

        parts = relative_path.split(os.sep)

        if len(parts) == 0:
            continue

        folder_name = parts[-1]

        label = standardize_maize_label(
            folder_name
        )

        for filename in image_files:

            full_path = os.path.join(
                root,
                filename
            )

            records.append({

                "image_path": os.path.abspath(
                    full_path
                ),

                "dataset": "Maize_Nutrient",

                "crop": "Maize",

                "label": label,

                "health_type": (
                    "Healthy"
                    if label == "Healthy"
                    else "Nutrient Deficiency"
                ),

                "image_type": "original"

            })

    df = pd.DataFrame(records)

    print(f"\nTotal Maize images found: {len(df)}")

    if len(df) > 0:

        print("\nMaize classes:")

        print(
            df["label"]
            .value_counts()
            .to_string()
        )

    return df


# ============================================================
# 8. INSPECT OLID-I
# ============================================================

def process_olid_dataset():

    print("\n" + "=" * 70)
    print("OLID-I DATASET")
    print("=" * 70)

    print(f"Location: {OLID_DIR}")

    if not os.path.exists(OLID_DIR):

        print("ERROR: OLID directory not found.")

        return pd.DataFrame()

    images = collect_images(OLID_DIR)

    print(
        f"\nTotal OLID-I images found: {len(images)}"
    )

    if len(images) == 0:

        return pd.DataFrame()

    records = []

    for image_path in images:

        records.append({

            "image_path": image_path,

            "dataset": "OLID-I",

            "crop": "Unknown",

            "label": "Unknown",

            "health_type": "Unknown",

            "image_type": "original"

        })

    df = pd.DataFrame(records)

    print(
        "\nIMPORTANT:"
    )

    print(
        "OLID-I labels need separate inspection because "
        "the dataset can contain multiple labels."
    )

    return df


# ============================================================
# 9. CHECK DUPLICATE IMAGE PATHS
# ============================================================

def check_duplicates(df, dataset_name):

    if len(df) == 0:
        return

    duplicates = df[
        df["image_path"].duplicated(
            keep=False
        )
    ]

    print(
        f"\n{dataset_name} duplicate image paths: "
        f"{len(duplicates)}"
    )


# ============================================================
# 10. CHECK MISSING IMAGES
# ============================================================

def check_missing_images(df, dataset_name):

    if len(df) == 0:
        return

    missing = []

    for path in df["image_path"]:

        if not os.path.exists(path):

            missing.append(path)

    print(
        f"{dataset_name} missing images: "
        f"{len(missing)}"
    )


# ============================================================
# 11. SAVE DATASET CSV
# ============================================================

def save_dataset(df, filename):

    if len(df) == 0:

        print(
            f"Nothing to save for {filename}"
        )

        return

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved: {output_path}"
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("NUTRIENT DATASET PREPARATION")
    print("=" * 70)

    print(
        f"\nProject root:\n{PROJECT_ROOT}"
    )

    print(
        f"\nRaw dataset directory:\n{RAW_DIR}"
    )

    # --------------------------------------------------------
    # Banana
    # --------------------------------------------------------

    banana_df = process_banana_dataset()

    check_duplicates(
        banana_df,
        "Banana"
    )

    check_missing_images(
        banana_df,
        "Banana"
    )

    save_dataset(
        banana_df,
        "nutrient_banana.csv"
    )

    # --------------------------------------------------------
    # Maize
    # --------------------------------------------------------

    maize_df = process_maize_dataset()

    check_duplicates(
        maize_df,
        "Maize"
    )

    check_missing_images(
        maize_df,
        "Maize"
    )

    save_dataset(
        maize_df,
        "nutrient_maize.csv"
    )

    # --------------------------------------------------------
    # OLID-I
    # --------------------------------------------------------

    olid_df = process_olid_dataset()

    check_duplicates(
        olid_df,
        "OLID-I"
    )

    check_missing_images(
        olid_df,
        "OLID-I"
    )

    save_dataset(
        olid_df,
        "nutrient_olid.csv"
    )

    # --------------------------------------------------------
    # Combined Banana + Maize
    # --------------------------------------------------------

    nutrient_df = pd.concat(
        [
            banana_df,
            maize_df
        ],
        ignore_index=True
    )

    if len(nutrient_df) > 0:

        print("\n" + "=" * 70)
        print("BANANA + MAIZE SUMMARY")
        print("=" * 70)

        print(
            f"\nTotal images: {len(nutrient_df)}"
        )

        print("\nImages by dataset:")

        print(
            nutrient_df[
                "dataset"
            ]
            .value_counts()
            .to_string()
        )

        print("\nImages by crop:")

        print(
            nutrient_df[
                "crop"
            ]
            .value_counts()
            .to_string()
        )

        print("\nImages by health type:")

        print(
            nutrient_df[
                "health_type"
            ]
            .value_counts()
            .to_string()
        )

        print("\nAll nutrient classes:")

        print(
            nutrient_df[
                "label"
            ]
            .value_counts()
            .to_string()
        )

        save_dataset(
            nutrient_df,
            "nutrient_banana_maize.csv"
        )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NUTRIENT DATASET PREPARATION COMPLETED")
    print("=" * 70)

    print("\nCreated files:")

    print(
        "1. datasets/nutrient_banana.csv"
    )

    print(
        "2. datasets/nutrient_maize.csv"
    )

    print(
        "3. datasets/nutrient_olid.csv"
    )

    print(
        "4. datasets/nutrient_banana_maize.csv"
    )

    print(
        "\nNext step: inspect the output before creating "
        "train/validation/test splits."
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()