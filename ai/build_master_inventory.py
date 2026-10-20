from pathlib import Path
import csv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "datasets"

OUTPUT_FILE = OUTPUT_DIR / "master_dataset_v2.csv"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}


# ============================================================
# HELPERS
# ============================================================

def is_image(path):

    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def rel_path(path):

    return path.relative_to(
        PROJECT_ROOT
    ).as_posix()


def normalize_crop(name):

    name = name.lower().strip()

    mapping = {

        "apple": "Apple",
        "blueberry": "Blueberry",
        "cherry": "Cherry",

        "corn": "Maize",
        "maize": "Maize",

        "grape": "Grape",
        "orange": "Orange",
        "peach": "Peach",

        "pepper,bell": "Bell Pepper",
        "bell_pepper": "Bell Pepper",

        "potato": "Potato",
        "raspberry": "Raspberry",

        "soybean": "Soybean",
        "soyabean": "Soybean",

        "squash": "Squash",
        "strawberry": "Strawberry",
        "tomato": "Tomato",

        "ash_gourd": "Ash Gourd",
        "bitter_gourd": "Bitter Gourd",
        "bottle_gourd": "Bottle Gourd",
        "cucumber": "Cucumber",
        "eggplant": "Eggplant",
        "ridge_gourd": "Ridge Gourd",
        "snake_gourd": "Snake Gourd",

        "banana": "Banana",
        "rice": "Rice",
        "paddy": "Rice"
    }

    return mapping.get(
        name,
        name.title()
    )


# ============================================================
# PLANTVILLAGE
# ============================================================

def process_plantvillage(rows):

    root = RAW_DIR / "PlantVillage" / "color"

    print("\nProcessing PlantVillage...")

    if not root.exists():

        print("ERROR: PlantVillage color folder not found.")
        return

    for class_dir in root.iterdir():

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        if "___" not in class_name:
            continue

        crop_raw, condition = class_name.split(
            "___",
            1
        )

        crop = normalize_crop(
            crop_raw.replace("_", " ")
        )

        condition = condition.replace(
            "_",
            " "
        ).strip()

        if condition.lower() == "healthy":

            health_type = "Healthy"

        elif "spider" in condition.lower():

            health_type = "Pest"

        else:

            health_type = "Disease"

        for image in class_dir.rglob("*"):

            if not is_image(image):
                continue

            rows.append({

                "image_path": rel_path(image),

                "dataset": "PlantVillage",

                "crop": crop,

                "health_type": health_type,

                "condition": condition,

                "original_class": class_name,

                "source_split": "unspecified",

                "label_status": "labeled",

                "augmentation_status": "original",

                "multi_label": "No",

                "representation": "color"
            })


# ============================================================
# PLANTDOC
# ============================================================

def process_plantdoc(rows):

    root = RAW_DIR / "PlantDoc"

    print("\nProcessing PlantDoc...")

    if not root.exists():

        print("ERROR: PlantDoc folder not found.")
        return

    disease_words = [
        "scab",
        "rust",
        "spot",
        "blight",
        "mold",
        "mosaic",
        "virus",
        "powdery mildew"
    ]

    crop_words = [
        "apple",
        "bell_pepper",
        "blueberry",
        "cherry",
        "corn",
        "peach",
        "potato",
        "raspberry",
        "soyabean",
        "squash",
        "strawberry",
        "tomato",
        "grape"
    ]

    for split_dir in root.iterdir():

        if not split_dir.is_dir():
            continue

        split = split_dir.name.lower()

        if split not in {
            "train",
            "test"
        }:
            continue

        for class_dir in split_dir.iterdir():

            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            class_lower = class_name.lower()

            crop = "Unknown"

            for crop_word in crop_words:

                if crop_word in class_lower:

                    crop = normalize_crop(
                        crop_word
                    )

                    break

            disease = any(
                word in class_lower
                for word in disease_words
            )

            if disease:

                health_type = "Disease"
                condition = class_name

            else:

                health_type = "Healthy"
                condition = "Healthy"

            for image in class_dir.rglob("*"):

                if not is_image(image):
                    continue

                rows.append({

                    "image_path": rel_path(image),

                    "dataset": "PlantDoc",

                    "crop": crop,

                    "health_type": health_type,

                    "condition": condition,

                    "original_class": class_name,

                    "source_split": split,

                    "label_status": "labeled",

                    "augmentation_status": "original",

                    "multi_label": "No",

                    "representation": "rgb"
                })


# ============================================================
# PADDY DOCTOR
# ============================================================

def process_paddy(rows):

    root = RAW_DIR / "PaddyDoctor"

    print("\nProcessing PaddyDoctor...")

    if not root.exists():

        print("ERROR: PaddyDoctor folder not found.")
        return

    disease_map = {

        "bacterial_leaf_blight":
            "Bacterial Leaf Blight",

        "bacterial_leaf_streak":
            "Bacterial Leaf Streak",

        "bacterial_panicle_blight":
            "Bacterial Panicle Blight",

        "blast":
            "Blast",

        "brown_spot":
            "Brown Spot",

        "downy_mildew":
            "Downy Mildew",

        "tungro":
            "Tungro"
    }

    pest_map = {

        "hispa":
            "Hispa",

        "dead_heart":
            "Dead Heart"
    }

    healthy_map = {

        "normal":
            "Healthy"
    }

    train_dir = root / "train_images"

    if train_dir.exists():

        for class_dir in train_dir.iterdir():

            if not class_dir.is_dir():
                continue

            original = class_dir.name.lower()

            if original in disease_map:

                health_type = "Disease"
                condition = disease_map[original]

            elif original in pest_map:

                health_type = "Pest"
                condition = pest_map[original]

            elif original in healthy_map:

                health_type = "Healthy"
                condition = "Healthy"

            else:

                health_type = "Unknown"
                condition = class_dir.name

            for image in class_dir.rglob("*"):

                if not is_image(image):
                    continue

                rows.append({

                    "image_path": rel_path(image),

                    "dataset": "PaddyDoctor",

                    "crop": "Rice",

                    "health_type": health_type,

                    "condition": condition,

                    "original_class": class_dir.name,

                    "source_split": "train",

                    "label_status": "labeled",

                    "augmentation_status": "original",

                    "multi_label": "No",

                    "representation": "rgb"
                })

    # --------------------------------------------------------
    # UNLABELED TEST IMAGES
    # --------------------------------------------------------

    test_dir = root / "test_images"

    if test_dir.exists():

        for image in test_dir.rglob("*"):

            if not is_image(image):
                continue

            rows.append({

                "image_path": rel_path(image),

                "dataset": "PaddyDoctor",

                "crop": "Rice",

                "health_type": "Unknown",

                "condition": "Unknown",

                "original_class": "",

                "source_split": "test",

                "label_status": "unlabeled",

                "augmentation_status": "original",

                "multi_label": "No",

                "representation": "rgb"
            })


# ============================================================
# OLID-I
# ============================================================

def process_olid(rows):

    root = RAW_DIR / "OLID"

    print("\nProcessing OLID-I...")

    if not root.exists():

        print("ERROR: OLID folder not found.")
        return

    mapping = {

        "N": (
            "Nutrient Deficiency",
            "Nitrogen Deficiency"
        ),

        "K": (
            "Nutrient Deficiency",
            "Potassium Deficiency"
        ),

        "N_K": (
            "Nutrient Deficiency",
            "Nitrogen + Potassium Deficiency"
        ),

        "N_Mg": (
            "Nutrient Deficiency",
            "Nitrogen + Magnesium Deficiency"
        ),

        "K_Mg": (
            "Nutrient Deficiency",
            "Potassium + Magnesium Deficiency"
        ),

        "PM": (
            "Disease",
            "Powdery Mildew"
        ),

        "DM": (
            "Disease",
            "Downy Mildew"
        ),

        "LS": (
            "Disease",
            "Leaf Spot"
        ),

        "JAS": (
            "Pest",
            "Jassid"
        ),

        "MIT": (
            "Pest",
            "Mite"
        ),

        "EB": (
            "Pest",
            "Epilachna Beetle"
        ),

        "FB": (
            "Pest",
            "Flea Beetle"
        ),

        "LM": (
            "Pest",
            "Leaf Miner"
        ),

        "PC": (
            "Pest",
            "Pumpkin Caterpillar"
        ),

        "PLEI": (
            "Pest",
            "Pumpkin Leaf Eating Insect"
        ),

        "JAS_MIT": (
            "Pest",
            "Jassid + Mite"
        ),

        "MIT_EB": (
            "Pest",
            "Mite + Epilachna Beetle"
        ),

        "PLEI_IEM": (
            "Pest",
            "Pumpkin Leaf Eating Insect + Insect Egg Mass"
        ),

        "PLEI_MIT": (
            "Pest",
            "Pumpkin Leaf Eating Insect + Mite"
        ),

        "healthy": (
            "Healthy",
            "Healthy"
        )
    }

    for class_dir in root.iterdir():

        if not class_dir.is_dir():
            continue

        folder = class_dir.name

        if "__" not in folder:
            continue

        crop_raw, code = folder.split(
            "__",
            1
        )

        crop = normalize_crop(crop_raw)

        code = code.strip()

        health_type, condition = mapping.get(
            code,
            ("Unknown", code)
        )

        multi_label = (
            "Yes"
            if "_" in code
            else "No"
        )

        for image in class_dir.rglob("*"):

            if not is_image(image):
                continue

            rows.append({

                "image_path": rel_path(image),

                "dataset": "OLID-I",

                "crop": crop,

                "health_type": health_type,

                "condition": condition,

                "original_class": code,

                "source_split": "unspecified",

                "label_status": "labeled",

                "augmentation_status": "original",

                "multi_label": multi_label,

                "representation": "rgb"
            })


# ============================================================
# BANANA NUTRIENT
# ============================================================

def process_banana(rows):

    root = RAW_DIR / "Banana_Nutrient"

    print("\nProcessing Banana Nutrient...")

    if not root.exists():

        print("ERROR: Banana folder not found.")
        return

    for image in root.rglob("*"):

        if not is_image(image):
            continue

        parts = [
            part.lower()
            for part in image.parts
        ]

        # ----------------------------------------------------
        # Determine RAW / AUGMENTED
        # ----------------------------------------------------

        if any(
            "augmented" in part
            for part in parts
        ):

            augmentation_status = "augmented"

        elif any(
            "raw" in part
            for part in parts
        ):

            augmentation_status = "original"

        else:

            augmentation_status = "unknown"

        # ----------------------------------------------------
        # Class name
        # ----------------------------------------------------

        class_name = image.parent.name

        if class_name.lower() == "healthy":

            health_type = "Healthy"
            condition = "Healthy"

        else:

            health_type = "Nutrient Deficiency"

            nutrient_map = {

                "boron": "Boron Deficiency",
                "calcium": "Calcium Deficiency",
                "iron": "Iron Deficiency",
                "magnesium": "Magnesium Deficiency",
                "manganese": "Manganese Deficiency",
                "potassium": "Potassium Deficiency",
                "sulphur": "Sulphur Deficiency",
                "zinc": "Zinc Deficiency"
            }

            condition = nutrient_map.get(
                class_name.lower(),
                class_name.title() + " Deficiency"
            )

        rows.append({

            "image_path": rel_path(image),

            "dataset": "Banana_Nutrient",

            "crop": "Banana",

            "health_type": health_type,

            "condition": condition,

            "original_class": class_name,

            "source_split": "unspecified",

            "label_status": "labeled",

            "augmentation_status":
                augmentation_status,

            "multi_label": "No",

            "representation": "rgb"
        })


# ============================================================
# MAIZE NUTRIENT
# ============================================================

def process_maize(rows):

    root = RAW_DIR / "Maize_Nutrient"

    print("\nProcessing Maize Nutrient...")

    if not root.exists():

        print("ERROR: Maize folder not found.")
        return

    # Actual structure:
    #
    # Maize_Nutrient
    # └── Maize Nutrient Deficiency Dataset
    #     ├── Helathy
    #     ├── Magnessium
    #     ├── Nitrogen
    #     ├── Phosphorous
    #     └── Pottasium

    class_map = {

        "helathy": (
            "Healthy",
            "Healthy"
        ),

        "magnessium": (
            "Nutrient Deficiency",
            "Magnesium Deficiency"
        ),

        "nitrogen": (
            "Nutrient Deficiency",
            "Nitrogen Deficiency"
        ),

        "phosphorous": (
            "Nutrient Deficiency",
            "Phosphorus Deficiency"
        ),

        "pottasium": (
            "Nutrient Deficiency",
            "Potassium Deficiency"
        )
    }

    for image in root.rglob("*"):

        if not is_image(image):
            continue

        # Find the class folder
        class_name = image.parent.name

        key = class_name.lower().strip()

        if key not in class_map:

            continue

        health_type, condition = class_map[key]

        rows.append({

            "image_path": rel_path(image),

            "dataset": "Maize_Nutrient",

            "crop": "Maize",

            "health_type": health_type,

            "condition": condition,

            "original_class": class_name,

            "source_split": "unspecified",

            "label_status": "labeled",

            "augmentation_status": "original",

            "multi_label": "No",

            "representation": "rgb"
        })


# ============================================================
# SAVE
# ============================================================

def save(rows):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fields = [
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
        "representation"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# REPORT
# ============================================================

def report(rows):

    from collections import Counter

    print("\n")
    print("=" * 65)
    print("MASTER DATASET V2 REPORT")
    print("=" * 65)

    print(
        f"\nTotal records: {len(rows):,}"
    )

    print("\nDataset counts:")
    print("-" * 65)

    dataset_counts = Counter(
        row["dataset"]
        for row in rows
    )

    for name, count in sorted(
        dataset_counts.items()
    ):

        print(
            f"{name:<25} {count:>8,}"
        )

    print("\nHealth categories:")
    print("-" * 65)

    health_counts = Counter(
        row["health_type"]
        for row in rows
    )

    for name, count in sorted(
        health_counts.items()
    ):

        print(
            f"{name:<25} {count:>8,}"
        )

    print("\nBanana image types:")
    print("-" * 65)

    banana_counts = Counter(
        row["augmentation_status"]
        for row in rows
        if row["dataset"] == "Banana_Nutrient"
    )

    for name, count in sorted(
        banana_counts.items()
    ):

        print(
            f"{name:<25} {count:>8,}"
        )

    print("\nMaize classes:")
    print("-" * 65)

    maize_counts = Counter(
        row["condition"]
        for row in rows
        if row["dataset"] == "Maize_Nutrient"
    )

    for name, count in sorted(
        maize_counts.items()
    ):

        print(
            f"{name:<45} {count:>8,}"
        )

    print("\nUnlabeled records:")

    unlabeled = sum(
        1
        for row in rows
        if row["label_status"] == "unlabeled"
    )

    print(
        f"  {unlabeled:,}"
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nInventory V2 completed.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("BUILDING MASTER DATASET V2")
    print("=" * 65)

    rows = []

    process_plantvillage(rows)
    process_plantdoc(rows)
    process_paddy(rows)
    process_olid(rows)
    process_banana(rows)
    process_maize(rows)

    save(rows)

    report(rows)


if __name__ == "__main__":
    main()