import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw"
)

DATASETS = [
    "PlantVillage",
    "PlantDoc",
    "PaddyDoctor",
    "OLID",
    "Banana_Nutrient",
    "Maize_Nutrient"
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


def inspect_dataset(dataset_name):

    dataset_path = os.path.join(
        DATASET_ROOT,
        dataset_name
    )

    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    if not os.path.exists(dataset_path):

        print("STATUS: NOT FOUND")
        print("Expected location:")
        print(dataset_path)

        return

    total_images = 0
    folders = {}

    for root, dirs, files in os.walk(dataset_path):

        image_files = [
            file
            for file in files
            if file.lower().endswith(IMAGE_EXTENSIONS)
        ]

        if not image_files:
            continue

        relative_path = os.path.relpath(
            root,
            dataset_path
        )

        if relative_path == ".":
            folder_name = "[ROOT]"
        else:
            folder_name = relative_path

        count = len(image_files)

        folders[folder_name] = count

        total_images += count

    print("\nFOLDER / CLASS DISTRIBUTION")
    print("-" * 80)

    if folders:

        for folder, count in sorted(folders.items()):

            print(
                f"{folder:<65} {count:>7}"
            )

    else:

        print("No images found.")

    print("\nSUMMARY")
    print("-" * 80)

    print(
        f"Total images              : {total_images}"
    )

    print(
        f"Folders containing images : {len(folders)}"
    )


def main():

    print("\n")
    print("=" * 80)
    print("AI CROP HEALTH PROJECT - FAST DATASET INSPECTOR")
    print("=" * 80)

    print("\nProject directory:")
    print(PROJECT_ROOT)

    print("\nDataset directory:")
    print(DATASET_ROOT)

    for dataset in DATASETS:

        inspect_dataset(dataset)

    print("\n")
    print("=" * 80)
    print("INSPECTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()