from crop_health_predictor import (
    CropHealthPredictor,
    print_prediction_result
)


# ===============================================================
# CREATE PREDICTOR
# ===============================================================

print("=" * 70)
print("TESTING UNIFIED CROP HEALTH PREDICTOR")
print("=" * 70)

predictor = CropHealthPredictor()


# ===============================================================
# CHANGE THIS PATH
# ===============================================================

IMAGE_PATH = r"D:\AI_Crop_Health_AR_DigitalTwin\datasets\raw\Banana_Nutrient"


# ===============================================================
# IMPORTANT
# ===============================================================
# IMAGE_PATH must point to ONE image.
#
# Example:
#
# IMAGE_PATH = r"D:\AI_Crop_Health_AR_DigitalTwin\datasets\raw\Banana_Nutrient\Healthy\image.jpg"
#
# Do NOT use the dataset folder itself.
# ===============================================================


# ===============================================================
# CROP
# ===============================================================
# Use:
#
# "banana"
#
# or
#
# "maize"
# ===============================================================

CROP = "banana"


# ===============================================================
# RUN PREDICTION
# ===============================================================

result = predictor.predict_crop_health(
    IMAGE_PATH,
    CROP
)


# ===============================================================
# DISPLAY RESULT
# ===============================================================

print_prediction_result(result)