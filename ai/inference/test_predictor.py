from crop_health_predictor import CropHealthPredictor, print_prediction_result


IMAGE_PATH = (
    r"D:\AI_Crop_Health_AR_DigitalTwin"
    r"\datasets\raw\Banana_Nutrient"
    r"\Augmented Images of Nutrient Deficient Banana leaves"
    r"\boron\b1.jpg"
)


print("=" * 70)
print("TESTING BANANA CROP HEALTH PREDICTION")
print("=" * 70)

predictor = CropHealthPredictor()

result = predictor.predict_crop_health(
    IMAGE_PATH,
    "banana"
)

print_prediction_result(result)