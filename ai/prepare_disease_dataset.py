import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"

TRAIN_INPUT = DATASET_DIR / "train.csv"
VAL_INPUT = DATASET_DIR / "val.csv"
TEST_INPUT = DATASET_DIR / "test.csv"

TRAIN_OUTPUT = DATASET_DIR / "disease_train.csv"
VAL_OUTPUT = DATASET_DIR / "disease_val.csv"
TEST_OUTPUT = DATASET_DIR / "disease_test.csv"

ALLOWED_DATASETS = [
    "PlantVillage",
    "PlantDoc",
    "PaddyDoctor"
]

print("=" * 65)
print("PREPARING DISEASE CLASSIFICATION DATASET")
print("=" * 65)

print("\nLoading split datasets...")

train = pd.read_csv(TRAIN_INPUT)
val = pd.read_csv(VAL_INPUT)
test = pd.read_csv(TEST_INPUT)

print(f"Train records loaded: {len(train):,}")
print(f"Validation records loaded: {len(val):,}")
print(f"Test records loaded: {len(test):,}")


def prepare(df):

    df = df[
        df["dataset"].isin(ALLOWED_DATASETS)
    ].copy()

    df = df[
        df["health_type"].isin(
            ["Disease", "Healthy"]
        )
    ].copy()

    df = df[
        df["multi_label"].astype(str).str.lower()
        != "true"
    ].copy()

    return df


print("\nFiltering disease-oriented datasets...")

train = prepare(train)
val = prepare(val)
test = prepare(test)


train.to_csv(TRAIN_OUTPUT, index=False)
val.to_csv(VAL_OUTPUT, index=False)
test.to_csv(TEST_OUTPUT, index=False)


print("\n" + "=" * 65)
print("DISEASE DATASET REPORT")
print("=" * 65)

print(f"\nTraining records:    {len(train):,}")
print(f"Validation records:  {len(val):,}")
print(f"Testing records:     {len(test):,}")

print("\nUnique conditions:")

print(f"Train:      {train['condition'].nunique()}")
print(f"Validation: {val['condition'].nunique()}")
print(f"Test:       {test['condition'].nunique()}")


print("\nDataset distribution:")

distribution = pd.DataFrame({
    "Train": train["dataset"].value_counts(),
    "Validation": val["dataset"].value_counts(),
    "Test": test["dataset"].value_counts()
}).fillna(0).astype(int)

print(distribution)


print("\nHealth type distribution:")

health_distribution = pd.DataFrame({
    "Train": train["health_type"].value_counts(),
    "Validation": val["health_type"].value_counts(),
    "Test": test["health_type"].value_counts()
}).fillna(0).astype(int)

print(health_distribution)


print("\nCondition list:")

conditions = sorted(
    train["condition"].dropna().unique()
)

for i, condition in enumerate(conditions, start=1):
    print(f"{i:3}. {condition}")


print("\n" + "=" * 65)
print("FILES CREATED")
print("=" * 65)

print(f"\n{TRAIN_OUTPUT}")
print(VAL_OUTPUT)
print(TEST_OUTPUT)

print("\nDisease dataset preparation completed.")
print("=" * 65)