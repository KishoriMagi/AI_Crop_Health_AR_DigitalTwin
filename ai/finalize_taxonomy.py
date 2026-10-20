from pathlib import Path
import csv
import re
from collections import Counter


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"

INPUT_FILE = DATASET_DIR / "clean_master_dataset.csv"

OUTPUT_FILE = DATASET_DIR / "final_master_dataset.csv"
TAXONOMY_FILE = DATASET_DIR / "final_taxonomy.csv"


# ============================================================
# CONDITION NORMALIZATION
# ============================================================

def normalize_condition(crop, condition):

    c = condition.strip().lower()

    # --------------------------------------------------------
    # HEALTHY
    # --------------------------------------------------------

    if c == "healthy":
        return "Healthy"

    # --------------------------------------------------------
    # APPLE
    # --------------------------------------------------------

    if crop == "Apple":

        if "scab" in c:
            return "Apple Scab"

        if "black rot" in c:
            return "Black Rot"

        if "rust" in c:
            return "Cedar Apple Rust"

    # --------------------------------------------------------
    # CHERRY
    # --------------------------------------------------------

    if crop == "Cherry":

        if "powdery mildew" in c:
            return "Powdery Mildew"

    # --------------------------------------------------------
    # MAIZE
    # --------------------------------------------------------

    if crop == "Maize":

        if "gray leaf spot" in c or "cercospora" in c:
            return "Gray Leaf Spot"

        if "common rust" in c or "rust leaf" in c:
            return "Common Rust"

        if "northern leaf blight" in c:
            return "Northern Leaf Blight"

        if "leaf blight" in c:
            return "Leaf Blight"

        if "magnesium deficiency" in c:
            return "Magnesium Deficiency"

        if "magnesium" in c:
            return "Magnesium Deficiency"

        if "nitrogen" in c:
            return "Nitrogen Deficiency"

        if "phosphor" in c:
            return "Phosphorus Deficiency"

        if "potassium" in c:
            return "Potassium Deficiency"

    # --------------------------------------------------------
    # PEACH
    # --------------------------------------------------------

    if crop == "Peach":

        if "bacterial spot" in c:
            return "Bacterial Spot"

    # --------------------------------------------------------
    # PEPPER
    # --------------------------------------------------------

    if crop == "Bell Pepper":

        if "bacterial spot" in c:
            return "Bacterial Spot"

        if "leaf spot" in c:
            return "Leaf Spot"

    # --------------------------------------------------------
    # POTATO
    # --------------------------------------------------------

    if crop == "Potato":

        if "early blight" in c:
            return "Early Blight"

        if "late blight" in c:
            return "Late Blight"

    # --------------------------------------------------------
    # GRAPE
    # --------------------------------------------------------

    if crop == "Grape":

        if "black rot" in c:
            return "Black Rot"

        if "esca" in c or "black measles" in c:
            return "Esca / Black Measles"

        if "isariopsis" in c or "leaf blight" in c:
            return "Leaf Blight / Isariopsis Leaf Spot"

    # --------------------------------------------------------
    # ORANGE
    # --------------------------------------------------------

    if crop == "Orange":

        if "huanglongbing" in c or "citrus greening" in c:
            return "Huanglongbing / Citrus Greening"

    # --------------------------------------------------------
    # SQUASH
    # --------------------------------------------------------

    if crop == "Squash":

        if "powdery mildew" in c:
            return "Powdery Mildew"

    # --------------------------------------------------------
    # STRAWBERRY
    # --------------------------------------------------------

    if crop == "Strawberry":

        if "leaf scorch" in c:
            return "Leaf Scorch"

    # --------------------------------------------------------
    # TOMATO
    # --------------------------------------------------------

    if crop == "Tomato":

        if "bacterial spot" in c:
            return "Bacterial Spot"

        if "early blight" in c:
            return "Early Blight"

        if "late blight" in c:
            return "Late Blight"

        if "leaf mold" in c or "mold leaf" in c:
            return "Leaf Mold"

        if "septoria" in c:
            return "Septoria Leaf Spot"

        if "target spot" in c:
            return "Target Spot"

        if "yellow leaf curl" in c or "yellow virus" in c:
            return "Tomato Yellow Leaf Curl Virus"

        if "mosaic" in c:
            return "Tomato Mosaic Virus"

        if "spider mite" in c:
            return "Two-Spotted Spider Mite"

        if "mite" in c and "jassid" in c:
            return "Jassid + Mite"

        if "leaf miner" in c:
            return "Leaf Miner"

        if c == "mite":
            return "Mite"

        if c == "jassid + mite":
            return "Jassid + Mite"

        if "nitrogen + potassium" in c:
            return "Nitrogen + Potassium Deficiency"

        if "nitrogen deficiency" in c:
            return "Nitrogen Deficiency"

        if "potassium deficiency" in c:
            return "Potassium Deficiency"

    # --------------------------------------------------------
    # RICE
    # --------------------------------------------------------

    if crop == "Rice":

        if "bacterial leaf blight" in c:
            return "Bacterial Leaf Blight"

        if "bacterial leaf streak" in c:
            return "Bacterial Leaf Streak"

        if "bacterial panicle blight" in c:
            return "Bacterial Panicle Blight"

        if c == "blast":
            return "Blast"

        if "brown spot" in c:
            return "Brown Spot"

        if "downy mildew" in c:
            return "Downy Mildew"

        if "tungro" in c:
            return "Tungro"

        if "hispa" in c:
            return "Hispa"

        if "dead heart" in c:
            return "Dead Heart"

    # --------------------------------------------------------
    # OLID NUTRIENTS
    # --------------------------------------------------------

    if "nitrogen + potassium" in c:
        return "Nitrogen + Potassium Deficiency"

    if "nitrogen + magnesium" in c:
        return "Nitrogen + Magnesium Deficiency"

    if "potassium + magnesium" in c:
        return "Potassium + Magnesium Deficiency"

    if "nitrogen deficiency" in c:
        return "Nitrogen Deficiency"

    if "potassium deficiency" in c:
        return "Potassium Deficiency"

    if "magnesium deficiency" in c:
        return "Magnesium Deficiency"

    if "phosphorus deficiency" in c:
        return "Phosphorus Deficiency"

    # --------------------------------------------------------
    # BANANA NUTRIENTS
    # --------------------------------------------------------

    if crop == "Banana":

        nutrient_map = {
            "boron": "Boron Deficiency",
            "calcium": "Calcium Deficiency",
            "iron": "Iron Deficiency",
            "magnesium": "Magnesium Deficiency",
            "manganese": "Manganese Deficiency",
            "potassium": "Potassium Deficiency",
            "sulphur": "Sulphur Deficiency",
            "sulfur": "Sulphur Deficiency",
            "zinc": "Zinc Deficiency"
        }

        for key, value in nutrient_map.items():

            if key in c:
                return value

    # --------------------------------------------------------
    # GENERIC NORMALIZATION
    # --------------------------------------------------------

    replacements = {
        "apple scab leaf": "Apple Scab",
        "apple rust leaf": "Cedar Apple Rust",

        "corn gray leaf spot": "Gray Leaf Spot",
        "corn leaf blight": "Leaf Blight",
        "corn rust leaf": "Common Rust",

        "potato leaf early blight": "Early Blight",
        "potato leaf late blight": "Late Blight",

        "tomato early blight leaf": "Early Blight",
        "tomato septoria leaf spot": "Septoria Leaf Spot",
        "tomato leaf bacterial spot": "Bacterial Spot",
        "tomato leaf late blight": "Late Blight",
        "tomato leaf mosaic virus": "Tomato Mosaic Virus",
        "tomato mold leaf": "Leaf Mold",

        "bell_pepper leaf spot": "Leaf Spot",
        "squash powdery mildew leaf": "Powdery Mildew",

        "powdery mildew": "Powdery Mildew",
        "downy mildew": "Downy Mildew",
        "leaf spot": "Leaf Spot"
    }

    if c in replacements:
        return replacements[c]

    return condition.strip()


# ============================================================
# HEALTH TYPE NORMALIZATION
# ============================================================

def normalize_health_type(row, final_condition):

    original = row["health_type"]

    if final_condition == "Healthy":
        return "Healthy"

    nutrient_words = [
        "nitrogen",
        "potassium",
        "magnesium",
        "phosphorus",
        "boron",
        "calcium",
        "iron",
        "manganese",
        "sulphur",
        "sulfur",
        "zinc"
    ]

    if any(
        word in final_condition.lower()
        for word in nutrient_words
    ):
        return "Nutrient Deficiency"

    if original == "Pest":
        return "Pest"

    return original


# ============================================================
# LOAD DATA
# ============================================================

def load_rows():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return list(csv.DictReader(file))


# ============================================================
# PROCESS
# ============================================================

def process(rows):

    final_rows = []

    for row in rows:

        final_condition = normalize_condition(
            row["crop"],
            row["condition"]
        )

        final_health = normalize_health_type(
            row,
            final_condition
        )

        row["final_condition"] = final_condition
        row["final_health_type"] = final_health

        # ----------------------------------------------------
        # Training eligibility
        # ----------------------------------------------------

        if row["label_status"] == "unlabeled":
            row["training_eligible"] = "No"

        elif row["file_status"] != "OK":
            row["training_eligible"] = "No"

        else:
            row["training_eligible"] = "Yes"

        # ----------------------------------------------------
        # Duplicate status
        # ----------------------------------------------------

        if row["duplicate_of"]:
            row["duplicate_status"] = "Duplicate"

        else:
            row["duplicate_status"] = "Unique"

        final_rows.append(row)

    return final_rows


# ============================================================
# SAVE FINAL MASTER DATASET
# ============================================================

def save_final(rows):

    fieldnames = [
        "image_path",
        "dataset",
        "crop",
        "health_type",
        "condition",
        "final_health_type",
        "final_condition",
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
        "duplicate_of",
        "duplicate_status"
    ]

    with open(
        OUTPUT_FILE,
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
# SAVE TAXONOMY
# ============================================================

def save_taxonomy(rows):

    counter = Counter()

    for row in rows:

        if row["training_eligible"] != "Yes":
            continue

        key = (
            row["crop"],
            row["final_health_type"],
            row["final_condition"]
        )

        counter[key] += 1

    taxonomy = []

    for index, (key, count) in enumerate(
        sorted(counter.items()),
        start=1
    ):

        crop, health_type, condition = key

        taxonomy.append({
            "label_id": f"LABEL_{index:04d}",
            "crop": crop,
            "health_type": health_type,
            "condition": condition,
            "image_count": count
        })

    with open(
        TAXONOMY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "label_id",
                "crop",
                "health_type",
                "condition",
                "image_count"
            ]
        )

        writer.writeheader()
        writer.writerows(taxonomy)

    return taxonomy


# ============================================================
# REPORT
# ============================================================

def report(rows, taxonomy):

    print("\n")
    print("=" * 65)
    print("FINAL TAXONOMY REPORT")
    print("=" * 65)

    print(
        f"\nTotal records: {len(rows):,}"
    )

    training = [
        row for row in rows
        if row["training_eligible"] == "Yes"
    ]

    print(
        f"Training eligible: {len(training):,}"
    )

    duplicates = [
        row for row in rows
        if row["duplicate_status"] == "Duplicate"
    ]

    print(
        f"Duplicate records: {len(duplicates):,}"
    )

    print(
        f"Unique final labels: {len(taxonomy)}"
    )

    print("\nHealth categories:")
    print("-" * 65)

    health_counter = Counter(
        row["final_health_type"]
        for row in training
    )

    for health, count in sorted(
        health_counter.items()
    ):

        print(
            f"{health:<30} {count:>8,}"
        )

    print("\nFinal labels:")
    print("-" * 65)

    for row in taxonomy:

     print(
        f"{row['label_id']:<12} "
        f"{row['crop']:<18} "
        f"{row['health_type']:<22} "
        f"{row['condition']:<45} "
        f"{row['image_count']:>5}"
    )

    print("\n")
    print("=" * 65)
    print("FILES CREATED")
    print("=" * 65)

    print(f"Final master dataset:")
    print(OUTPUT_FILE)

    print(f"\nFinal taxonomy:")
    print(TAXONOMY_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("FINALIZING CROP HEALTH TAXONOMY")
    print("=" * 65)

    rows = load_rows()

    print(
        f"\nLoaded {len(rows):,} records."
    )

    rows = process(rows)

    save_final(rows)

    taxonomy = save_taxonomy(rows)

    report(rows, taxonomy)

    print("\nTaxonomy finalization completed.")


if __name__ == "__main__":
    main()