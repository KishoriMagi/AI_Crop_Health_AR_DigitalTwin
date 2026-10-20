import os
import json
import requests
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:5000/api/predict"

DATASET_DIR = Path(
    r"D:\AI_Crop_Health_AR_DigitalTwin\datasets\raw\Maize_Nutrient"
)

NUMBER_OF_IMAGES = 10


# =========================================================
# FIND IMAGES
# =========================================================

image_paths = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:

    image_paths.extend(
        DATASET_DIR.rglob(extension)
    )


image_paths = sorted(image_paths)[:NUMBER_OF_IMAGES]


# =========================================================
# CHECK IMAGES
# =========================================================

print("=" * 70)
print("MAIZE BATCH API TEST")
print("=" * 70)

print()
print("API:", API_URL)
print("Images found:", len(image_paths))
print()


if len(image_paths) == 0:

    print("ERROR: No Maize images found.")
    raise SystemExit(1)


# =========================================================
# TEST RESULTS
# =========================================================

results = []

successful = 0
failed = 0


# =========================================================
# TEST EACH IMAGE
# =========================================================

for index, image_path in enumerate(image_paths, start=1):

    print("-" * 70)
    print(f"IMAGE {index}/{len(image_paths)}")
    print("-" * 70)

    print("Path:", image_path)

    try:

        with open(image_path, "rb") as image:

            files = {
                "image": (
                    image_path.name,
                    image,
                    "image/jpeg"
                )
            }

            data = {
                "crop": "maize"
            }

            response = requests.post(
                API_URL,
                files=files,
                data=data,
                timeout=120
            )


        print("HTTP Status:", response.status_code)

        if response.status_code != 200:

            print("API request failed.")

            failed += 1

            results.append({
                "image": str(image_path),
                "success": False,
                "http_status": response.status_code,
                "error": response.text
            })

            continue


        result = response.json()

        prediction = result.get(
            "prediction",
            {}
        )

        disease = prediction.get(
            "disease",
            {}
        )

        nutrient = prediction.get(
            "nutrient",
            {}
        )


        print()
        print("Disease:")
        print(
            "  Prediction :",
            disease.get("name")
        )

        print(
            "  Confidence :",
            disease.get("confidence_percent"),
            "%"
        )


        print()
        print("Nutrient:")
        print(
            "  Prediction :",
            nutrient.get("name")
        )

        print(
            "  Confidence :",
            nutrient.get("confidence_percent"),
            "%"
        )


        print()
        print(
            "Overall:",
            prediction.get("status")
        )


        successful += 1


        results.append({
            "image": str(image_path),
            "success": True,
            "http_status": response.status_code,
            "disease": disease.get("name"),
            "disease_confidence": disease.get(
                "confidence_percent"
            ),
            "nutrient": nutrient.get("name"),
            "nutrient_confidence": nutrient.get(
                "confidence_percent"
            ),
            "overall_status": prediction.get(
                "status"
            )
        })


    except Exception as e:

        print("ERROR:", str(e))

        failed += 1

        results.append({
            "image": str(image_path),
            "success": False,
            "error": str(e)
        })


# =========================================================
# SAVE RESULTS
# =========================================================

RESULTS_DIR = Path(
    r"D:\AI_Crop_Health_AR_DigitalTwin\ai\results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_FILE = RESULTS_DIR / "maize_batch_api_results.json"


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 70)
print("MAIZE BATCH TEST COMPLETED")
print("=" * 70)

print()
print("Total images :", len(image_paths))
print("Successful   :", successful)
print("Failed       :", failed)

print()
print("Results saved to:")
print(OUTPUT_FILE)

print()
print("=" * 70)