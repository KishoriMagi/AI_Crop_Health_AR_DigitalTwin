from pathlib import Path
import csv
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "clean_master_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "taxonomy_registry.csv"
)


def main():

    print("=" * 60)
    print("CROP HEALTH TAXONOMY BUILDER")
    print("=" * 60)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        rows = list(csv.DictReader(file))

    counter = Counter()

    for row in rows:

        if row["label_status"] != "labeled":
            continue

        if row["file_status"] != "OK":
            continue

        key = (
            row["crop"],
            row["health_type"],
            row["condition"]
        )

        counter[key] += 1

    taxonomy_rows = []

    for index, (
        key,
        count
    ) in enumerate(
        sorted(counter.items()),
        start=1
    ):

        crop, health_type, condition = key

        taxonomy_rows.append({
            "label_id": f"CROP_{index:04d}",
            "crop": crop,
            "health_type": health_type,
            "condition": condition,
            "image_count": count
        })

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "label_id",
            "crop",
            "health_type",
            "condition",
            "image_count"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(taxonomy_rows)

    print(
        f"\nTotal unique labels: "
        f"{len(taxonomy_rows)}"
    )

    print("\nLabels:")
    print("-" * 60)

    for row in taxonomy_rows:

        print(
            f"{row['label_id']:<12} "
            f"{row['crop']:<18} "
            f"{row['health_type']:<20} "
            f"{row['condition']:<45} "
            f"{row['image_count']:>5}"
        )

    print("\n")
    print("=" * 60)
    print("TAXONOMY CREATED")
    print("=" * 60)

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()