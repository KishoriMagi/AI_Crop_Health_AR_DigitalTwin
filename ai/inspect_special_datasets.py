from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}


def show_structure(root, max_depth=3):

    print("\n")
    print("=" * 70)
    print(f"DATASET: {root.name}")
    print("=" * 70)

    if not root.exists():

        print("ERROR: Dataset folder not found.")
        return

    for path in sorted(root.rglob("*")):

        relative = path.relative_to(root)

        depth = len(relative.parts)

        if depth > max_depth:
            continue

        if path.is_dir():

            print(
                "  " * (depth - 1)
                + f"[DIR] {path.name}"
            )

        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:

            print(
                "  " * (depth - 1)
                + f"[IMG] {path.name}"
            )


def count_by_folder(root):

    print("\nFolder image counts:")
    print("-" * 70)

    if not root.exists():
        return

    for folder in sorted(root.rglob("*")):

        if not folder.is_dir():
            continue

        count = sum(
            1
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        if count > 0:

            relative = folder.relative_to(root)

            print(
                f"{str(relative):<55} {count:>6}"
            )


def inspect_dataset(name):

    root = RAW_DIR / name

    show_structure(
        root,
        max_depth=4
    )

    count_by_folder(root)


def main():

    print("=" * 70)
    print("SPECIAL DATASET STRUCTURE INSPECTOR")
    print("=" * 70)

    inspect_dataset("Banana_Nutrient")

    inspect_dataset("Maize_Nutrient")

    print("\n")
    print("=" * 70)
    print("INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()