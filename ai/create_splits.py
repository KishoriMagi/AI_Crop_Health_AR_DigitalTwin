import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"

INPUT_FILE = DATASET_DIR / "master_dataset_v2.csv"

TRAIN_FILE = DATASET_DIR / "train.csv"
VAL_FILE = DATASET_DIR / "val.csv"
TEST_FILE = DATASET_DIR / "test.csv"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


print("=" * 65)
print("CREATING LEAKAGE-FREE DATASET SPLITS")
print("=" * 65)


# =========================================================
# 1. LOAD MASTER DATASET
# =========================================================

print("\nLoading master dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total records loaded: {len(df):,}")

print("\nColumns detected:")
print(df.columns.tolist())


# =========================================================
# 2. REMOVE UNLABELED RECORDS
# =========================================================

print("\n" + "-" * 65)
print("REMOVING UNLABELED RECORDS")
print("-" * 65)

unlabeled_mask = (
    df["label_status"]
    .astype(str)
    .str.lower()
    .isin(["unlabeled", "unknown"])
)

unlabeled_df = df[unlabeled_mask].copy()
labeled_df = df[~unlabeled_mask].copy()

print(f"Unlabeled records excluded: {len(unlabeled_df):,}")
print(f"Labeled records available: {len(labeled_df):,}")


# =========================================================
# 3. REMOVE EXACT DUPLICATES
# =========================================================

print("\n" + "-" * 65)
print("CHECKING EXACT DUPLICATES")
print("-" * 65)

# V2 does not contain SHA256.
# Therefore use image_path as the first duplicate check.

before = len(labeled_df)

labeled_df = labeled_df.drop_duplicates(
    subset=["image_path"],
    keep="first"
)

removed = before - len(labeled_df)

print(f"Duplicate image paths removed: {removed:,}")


# =========================================================
# 4. SEPARATE BANANA DATASET
# =========================================================

print("\n" + "-" * 65)
print("HANDLING BANANA AUGMENTATION")
print("-" * 65)

banana_mask = (
    labeled_df["dataset"]
    .astype(str)
    .str.lower()
    == "banana_nutrient"
)

banana_df = labeled_df[banana_mask].copy()
other_df = labeled_df[~banana_mask].copy()

banana_original = banana_df[
    banana_df["augmentation_status"]
    .astype(str)
    .str.lower()
    == "original"
].copy()

banana_augmented = banana_df[
    banana_df["augmentation_status"]
    .astype(str)
    .str.lower()
    == "augmented"
].copy()

print(f"Banana original images:  {len(banana_original):,}")
print(f"Banana augmented images: {len(banana_augmented):,}")


# =========================================================
# 5. SPLIT BANANA ORIGINAL IMAGES
# =========================================================

print("\nSplitting Banana original images...")

banana_train, banana_temp = train_test_split(
    banana_original,
    test_size=VAL_RATIO + TEST_RATIO,
    random_state=RANDOM_STATE,
    stratify=banana_original["condition"]
)

banana_val, banana_test = train_test_split(
    banana_temp,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_STATE,
    stratify=banana_temp["condition"]
)

banana_train["split"] = "train"
banana_val["split"] = "val"
banana_test["split"] = "test"

# ALL augmented Banana images go ONLY into training.
banana_augmented["split"] = "train"

print(f"Banana train originals:       {len(banana_train):,}")
print(f"Banana validation originals:  {len(banana_val):,}")
print(f"Banana test originals:        {len(banana_test):,}")
print(f"Banana augmented → train:     {len(banana_augmented):,}")


# =========================================================
# 6. SPLIT REMAINING DATASETS
# =========================================================

print("\n" + "-" * 65)
print("SPLITTING REMAINING DATASETS")
print("-" * 65)

# Use condition as the class for stratification.
#
# Classes with only one image cannot be stratified.
# We therefore identify rare classes first.

condition_counts = other_df["condition"].value_counts()

rare_conditions = condition_counts[
    condition_counts < 3
].index

normal_df = other_df[
    ~other_df["condition"].isin(rare_conditions)
].copy()

rare_df = other_df[
    other_df["condition"].isin(rare_conditions)
].copy()

print(f"Normal records: {len(normal_df):,}")
print(f"Rare-condition records: {len(rare_df):,}")


# =========================================================
# 7. STRATIFIED SPLIT
# =========================================================

print("\nCreating stratified train/validation/test split...")

other_train, other_temp = train_test_split(
    normal_df,
    test_size=VAL_RATIO + TEST_RATIO,
    random_state=RANDOM_STATE,
    stratify=normal_df["condition"]
)

other_val, other_test = train_test_split(
    other_temp,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_STATE,
    stratify=other_temp["condition"]
)


# =========================================================
# 8. HANDLE RARE CONDITIONS
# =========================================================

print("\nHandling rare conditions...")

# Rare classes cannot reliably be divided into
# train/validation/test using stratification.
#
# Keep them in training so they are available to the model.

rare_df["split"] = "train"

other_train["split"] = "train"
other_val["split"] = "val"
other_test["split"] = "test"

other_train = pd.concat(
    [other_train, rare_df],
    ignore_index=True
)


# =========================================================
# 9. COMBINE EVERYTHING
# =========================================================

train_df = pd.concat(
    [
        other_train,
        banana_train,
        banana_augmented
    ],
    ignore_index=True
)

val_df = pd.concat(
    [
        other_val,
        banana_val
    ],
    ignore_index=True
)

test_df = pd.concat(
    [
        other_test,
        banana_test
    ],
    ignore_index=True
)


# =========================================================
# 10. SHUFFLE
# =========================================================

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


# =========================================================
# 11. SAVE FILES
# =========================================================

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

val_df.to_csv(
    VAL_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# =========================================================
# 12. FINAL REPORT
# =========================================================

print("\n")
print("=" * 65)
print("FINAL SPLIT REPORT")
print("=" * 65)

print(f"\nTraining records:    {len(train_df):,}")
print(f"Validation records:  {len(val_df):,}")
print(f"Testing records:     {len(test_df):,}")

total_split = (
    len(train_df)
    + len(val_df)
    + len(test_df)
)

print(f"Total split records: {total_split:,}")


print("\nSplit percentages:")

print(
    f"Train:       {len(train_df) / total_split * 100:.2f}%"
)

print(
    f"Validation:  {len(val_df) / total_split * 100:.2f}%"
)

print(
    f"Test:        {len(test_df) / total_split * 100:.2f}%"
)


# =========================================================
# 13. DATASET DISTRIBUTION
# =========================================================

print("\n" + "-" * 65)
print("DATASET DISTRIBUTION")
print("-" * 65)

distribution = pd.DataFrame({
    "Train": train_df["dataset"].value_counts(),
    "Validation": val_df["dataset"].value_counts(),
    "Test": test_df["dataset"].value_counts()
}).fillna(0).astype(int)

print(distribution)


# =========================================================
# 14. HEALTH TYPE DISTRIBUTION
# =========================================================

print("\n" + "-" * 65)
print("HEALTH TYPE DISTRIBUTION")
print("-" * 65)

health_distribution = pd.DataFrame({
    "Train": train_df["health_type"].value_counts(),
    "Validation": val_df["health_type"].value_counts(),
    "Test": test_df["health_type"].value_counts()
}).fillna(0).astype(int)

print(health_distribution)


# =========================================================
# 15. CONDITION DISTRIBUTION
# =========================================================

print("\n" + "-" * 65)
print("CONDITION DISTRIBUTION")
print("-" * 65)

print("\nNumber of unique conditions:")

print(
    f"Train:       {train_df['condition'].nunique()}"
)

print(
    f"Validation:  {val_df['condition'].nunique()}"
)

print(
    f"Test:        {test_df['condition'].nunique()}"
)


# =========================================================
# 16. CHECK IMAGE PATH LEAKAGE
# =========================================================

print("\n" + "-" * 65)
print("CHECKING IMAGE PATH LEAKAGE")
print("-" * 65)

train_paths = set(train_df["image_path"])
val_paths = set(val_df["image_path"])
test_paths = set(test_df["image_path"])

train_val_overlap = train_paths.intersection(val_paths)
train_test_overlap = train_paths.intersection(test_paths)
val_test_overlap = val_paths.intersection(test_paths)

print(
    f"Train ↔ Validation overlap: {len(train_val_overlap)}"
)

print(
    f"Train ↔ Test overlap:       {len(train_test_overlap)}"
)

print(
    f"Validation ↔ Test overlap:  {len(val_test_overlap)}"
)


if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):

    print("\n✓ No image-path leakage detected.")

else:

    print(
        "\nWARNING: Image-path leakage detected!"
    )


# =========================================================
# 17. BANANA VERIFICATION
# =========================================================

print("\n" + "-" * 65)
print("BANANA AUGMENTATION VERIFICATION")
print("-" * 65)

banana_train_count = len(
    train_df[
        train_df["dataset"]
        .astype(str)
        .str.lower()
        == "banana_nutrient"
    ]
)

banana_val_count = len(
    val_df[
        val_df["dataset"]
        .astype(str)
        .str.lower()
        == "banana_nutrient"
    ]
)

banana_test_count = len(
    test_df[
        test_df["dataset"]
        .astype(str)
        .str.lower()
        == "banana_nutrient"
    ]
)

print(f"Banana train: {banana_train_count:,}")
print(f"Banana val:   {banana_val_count:,}")
print(f"Banana test:  {banana_test_count:,}")


banana_val_augmented = len(
    val_df[
        (val_df["dataset"].astype(str).str.lower() == "banana_nutrient")
        &
        (val_df["augmentation_status"].astype(str).str.lower() == "augmented")
    ]
)

banana_test_augmented = len(
    test_df[
        (test_df["dataset"].astype(str).str.lower() == "banana_nutrient")
        &
        (test_df["augmentation_status"].astype(str).str.lower() == "augmented")
    ]
)

print(
    f"Banana augmented in validation: {banana_val_augmented}"
)

print(
    f"Banana augmented in test:       {banana_test_augmented}"
)


if banana_val_augmented == 0 and banana_test_augmented == 0:
    print("\n✓ Banana augmentation leakage prevented.")
else:
    print("\nWARNING: Banana augmentation leakage detected!")


# =========================================================
# 18. FINAL FILES
# =========================================================

print("\n")
print("=" * 65)
print("FILES CREATED")
print("=" * 65)

print(f"\n{TRAIN_FILE}")
print(f"{VAL_FILE}")
print(f"{TEST_FILE}")

print("\n✓ Dataset splitting completed.")
print("=" * 65)