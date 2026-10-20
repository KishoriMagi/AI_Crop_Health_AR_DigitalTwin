import requests
import os
import json


# =========================================================
# API CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:5000/api/predict"


# =========================================================
# MAIZE IMAGE PATH
# =========================================================

IMAGE_PATH = (
    r"D:\AI_Crop_Health_AR_DigitalTwin"
    r"\datasets\raw\Maize_Nutrient"
    r"\Maize Nutrient Deficiency Dataset"
    r"\Helathy"
    r"\1.jpeg"
)


# =========================================================
# START TEST
# =========================================================

print("=" * 70)
print("TESTING AI CROP HEALTH PREDICTION API")
print("=" * 70)

print()
print("Image:")
print(IMAGE_PATH)


# =========================================================
# CHECK IMAGE
# =========================================================

if not os.path.exists(IMAGE_PATH):

    print()
    print("ERROR: Image does not exist.")
    print(IMAGE_PATH)

    raise SystemExit(1)


print()
print("Image found.")


# =========================================================
# SEND IMAGE TO API
# =========================================================

print()
print("Sending image to API...")
print("Crop: Maize")
print()


try:

    with open(IMAGE_PATH, "rb") as image:

        files = {
            "image": (
                os.path.basename(IMAGE_PATH),
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


    # =====================================================
    # HTTP RESPONSE
    # =====================================================

    print("HTTP Status:", response.status_code)

    print()

    print("=" * 70)
    print("API RESPONSE")
    print("=" * 70)

    try:

        result = response.json()

        print(
            json.dumps(
                result,
                indent=4
            )
        )

    except Exception:

        print(response.text)


    # =====================================================
    # TEST RESULT
    # =====================================================

    print()

    print("=" * 70)

    if response.status_code == 200:

        print("MAIZE API PREDICTION TEST PASSED")

    else:

        print("MAIZE API PREDICTION TEST FAILED")

    print("=" * 70)


# =========================================================
# CONNECTION ERROR
# =========================================================

except requests.exceptions.ConnectionError:

    print()
    print("ERROR: Could not connect to Flask API.")

    print()
    print("Make sure the Flask API is running in another terminal:")

    print()
    print("python ai\\api\\app.py")


# =========================================================
# TIMEOUT
# =========================================================

except requests.exceptions.Timeout:

    print()
    print("ERROR: API request timed out.")

    print()
    print("The AI model may still be processing the image.")


# =========================================================
# OTHER ERROR
# =========================================================

except Exception as e:

    print()
    print("ERROR:")
    print(str(e))