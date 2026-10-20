from pathlib import Path
import csv
from collections import Counter, defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"

CLEAN_FILE = DATASET_DIR / "clean_master_dataset.csv"
DUPLICATE_FILE = DATASET_DIR / "duplicate_images.csv"


def load_csv(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return list(csv.DictReader(file))


def main():

    print("=" * 60)
    print("DUPLICATE IMAGE ANALYSIS")
    print("=" * 60)

    rows = load_csv(CLEAN_FILE)
    duplicates = load_csv(DUPLICATE_FILE)

    print(f"\nTotal records: {len(rows):,}")
    print(f"Duplicate records: {len(duplicates):,}")

    # --------------------------------------------------------
    # Dataset duplicate analysis
    # --------------------------------------------------------

    path_to_dataset = {
        row["image_path"]: row["dataset"]
        for row in rows
    }

    same_dataset = 0
    cross_dataset = 0

    cross_pairs = Counter()

    for duplicate in duplicates:

        image1 = duplicate["image_path"]
        image2 = duplicate["duplicate_of"]

        dataset1 = path_to_dataset.get(image1, "Unknown")
        dataset2 = path_to_dataset.get(image2, "Unknown")

        if dataset1 == dataset2:

            same_dataset += 1

        else:

            cross_dataset += 1

            pair = tuple(
                sorted([dataset1, dataset2])
            )

            cross_pairs[pair] += 1

    print("\nDuplicate distribution:")
    print("-" * 60)

    print(
        f"Same dataset   : {same_dataset:,}"
    )

    print(
        f"Cross dataset  : {cross_dataset:,}"
    )

    print("\nCross-dataset duplicate pairs:")
    print("-" * 60)

    for pair, count in cross_pairs.most_common():

        print(
            f"{pair[0]} <-> {pair[1]} : {count:,}"
        )

    # --------------------------------------------------------
    # Duplicate examples
    # --------------------------------------------------------

    print("\nExample duplicate records:")
    print("-" * 60)

    for duplicate in duplicates[:20]:

        image1 = duplicate["image_path"]
        image2 = duplicate["duplicate_of"]

        print("\nImage:")
        print(" ", image1)

        print("Duplicate of:")
        print(" ", image2)

    # --------------------------------------------------------
    # Condition conflicts
    # --------------------------------------------------------

    path_to_row = {
        row["image_path"]: row
        for row in rows
    }

    conflicts = []

    for duplicate in duplicates:

        row1 = path_to_row.get(
            duplicate["image_path"]
        )

        row2 = path_to_row.get(
            duplicate["duplicate_of"]
        )

        if row1 is None or row2 is None:
            continue

        condition1 = row1["condition"]
        condition2 = row2["condition"]

        if condition1 != condition2:

            conflicts.append({
                "image_path_1":
                    duplicate["image_path"],

                "dataset_1":
                    row1["dataset"],

                "condition_1":
                    condition1,

                "image_path_2":
                    duplicate["duplicate_of"],

                "dataset_2":
                    row2["dataset"],

                "condition_2":
                    condition2
            })

    print("\n")
    print("=" * 60)
    print("LABEL CONFLICT ANALYSIS")
    print("=" * 60)

    print(
        f"\nDuplicate pairs with different labels: "
        f"{len(conflicts):,}"
    )

    if conflicts:

        print("\nExamples:")

        for conflict in conflicts[:20]:

            print("\n----------------------------------------")

            print(
                f"{conflict['dataset_1']}: "
                f"{conflict['condition_1']}"
            )

            print(
                f"    {conflict['image_path_1']}"
            )

            print(
                f"{conflict['dataset_2']}: "
                f"{conflict['condition_2']}"
            )

            print(
                f"    {conflict['image_path_2']}"
            )

    # --------------------------------------------------------
    # Save conflict report
    # --------------------------------------------------------

    conflict_file = DATASET_DIR / "duplicate_label_conflicts.csv"

    with open(
        conflict_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "image_path_1",
            "dataset_1",
            "condition_1",
            "image_path_2",
            "dataset_2",
            "condition_2"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(conflicts)

    print("\nConflict report:")
    print(conflict_file)

    print("\nDuplicate analysis completed.")


if __name__ == "__main__":
    main()