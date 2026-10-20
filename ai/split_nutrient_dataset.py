"""
split_nutrient_dataset.py

Creates train / validation / test CSV files for nutrient detection.

Important:
- Banana original images are split first.
- Banana augmented images are used ONLY for training.
- Maize images are stratified into train/validation/test.
- OLID-I is NOT included yet because its multi-label annotations
  need separate processing.
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets"
)

BANANA_CSV = os.path.join(
    DATASET_DIR,
    "nutrient_banana.csv"
)

MAIZE_CSV = os.path.join(
    DATASET_DIR,
    "nutrient_maize.csv"
)


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_STATE = 42


# ============================================================
# SPLIT FUNCTION
# ============================================================

def stratified_split(
    df,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15
):

    # First split:
    # 70% train
    # 30% temporary

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    # Split temporary 50/50:
    # 15% validation
    # 15% test

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


# ============================================================
# BANANA DATASET
# ============================================================

def split_banana():

    print("\n" + "=" * 70)
    print("BANANA NUTRIENT SPLIT")
    print("=" * 70)

    df = pd.read_csv(BANANA_CSV)

    print(
        f"\nTotal Banana records: {len(df)}"
    )

    # --------------------------------------------------------
    # Original images
    # --------------------------------------------------------

    original_df = df[
        df["image_type"] == "original"
    ].copy()

    augmented_df = df[
        df["image_type"] == "augmented"
    ].copy()

    print(
        f"Original images: {len(original_df)}"
    )

    print(
        f"Augmented images: {len(augmented_df)}"
    )

    # --------------------------------------------------------
    # Split ORIGINAL images only
    # --------------------------------------------------------

    (
        train_original,
        val_original,
        test_original
    ) = stratified_split(
        original_df
    )

    # --------------------------------------------------------
    # Augmented images:
    # Training ONLY
    # --------------------------------------------------------

    train_banana = pd.concat(
        [
            train_original,
            augmented_df
        ],
        ignore_index=True
    )

    val_banana = val_original

    test_banana = test_original

    print("\nBanana split:")

    print(
        f"Training:   {len(train_banana)}"
    )

    print(
        f"Validation: {len(val_banana)}"
    )

    print(
        f"Testing:    {len(test_banana)}"
    )

    return (
        train_banana,
        val_banana,
        test_banana
    )


# ============================================================
# MAIZE DATASET
# ============================================================

def split_maize():

    print("\n" + "=" * 70)
    print("MAIZE NUTRIENT SPLIT")
    print("=" * 70)

    df = pd.read_csv(MAIZE_CSV)

    print(
        f"\nTotal Maize records: {len(df)}"
    )

    (
        train_maize,
        val_maize,
        test_maize
    ) = stratified_split(
        df
    )

    print("\nMaize split:")

    print(
        f"Training:   {len(train_maize)}"
    )

    print(
        f"Validation: {len(val_maize)}"
    )

    print(
        f"Testing:    {len(test_maize)}"
    )

    return (
        train_maize,
        val_maize,
        test_maize
    )


# ============================================================
# PRINT CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    df,
    name
):

    print(
        f"\n{name} class distribution:"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("NUTRIENT DATASET SPLITTING")
    print("=" * 70)

    # --------------------------------------------------------
    # Banana
    # --------------------------------------------------------

    (
        train_banana,
        val_banana,
        test_banana
    ) = split_banana()

    # --------------------------------------------------------
    # Maize
    # --------------------------------------------------------

    (
        train_maize,
        val_maize,
        test_maize
    ) = split_maize()

    # --------------------------------------------------------
    # Combine Banana + Maize
    # --------------------------------------------------------

    train_df = pd.concat(
        [
            train_banana,
            train_maize
        ],
        ignore_index=True
    )

    val_df = pd.concat(
        [
            val_banana,
            val_maize
        ],
        ignore_index=True
    )

    test_df = pd.concat(
        [
            test_banana,
            test_maize
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    train_df = train_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    val_df = val_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_path = os.path.join(
        DATASET_DIR,
        "nutrient_train.csv"
    )

    val_path = os.path.join(
        DATASET_DIR,
        "nutrient_val.csv"
    )

    test_path = os.path.join(
        DATASET_DIR,
        "nutrient_test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print(
        f"\nTraining records:   {len(train_df)}"
    )

    print(
        f"Validation records: {len(val_df)}"
    )

    print(
        f"Testing records:    {len(test_df)}"
    )

    print(
        f"Total records:      "
        f"{len(train_df) + len(val_df) + len(test_df)}"
    )

    # --------------------------------------------------------
    # Class distributions
    # --------------------------------------------------------

    print_distribution(
        train_df,
        "TRAINING"
    )

    print_distribution(
        val_df,
        "VALIDATION"
    )

    print_distribution(
        test_df,
        "TESTING"
    )

    # --------------------------------------------------------
    # Dataset distributions
    # --------------------------------------------------------

    print("\nTraining by dataset:")

    print(
        train_df[
            "dataset"
        ]
        .value_counts()
        .to_string()
    )

    print("\nValidation by dataset:")

    print(
        val_df[
            "dataset"
        ]
        .value_counts()
        .to_string()
    )

    print("\nTesting by dataset:")

    print(
        test_df[
            "dataset"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    train_paths = set(
        train_df["image_path"]
    )

    val_paths = set(
        val_df["image_path"]
    )

    test_paths = set(
        test_df["image_path"]
    )

    train_val_overlap = (
        train_paths & val_paths
    )

    train_test_overlap = (
        train_paths & test_paths
    )

    val_test_overlap = (
        val_paths & test_paths
    )

    print("\n" + "=" * 70)
    print("LEAKAGE CHECK")
    print("=" * 70)

    print(
        f"\nTrain-Val overlap: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train-Test overlap: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Val-Test overlap: "
        f"{len(val_test_overlap)}"
    )

    # --------------------------------------------------------
    # Final paths
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        f"\n{train_path}"
    )

    print(
        f"{val_path}"
    )

    print(
        f"{test_path}"
    )

    print("\nOLID-I was intentionally excluded.")

    print(
        "It will be processed separately after inspecting "
        "its multi-label annotations."
    )

    print("\nDONE.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()