"""
===============================================================
AI CROP HEALTH AR DIGITAL TWIN
Crop Health Prediction Engine
===============================================================

IMPORTANT:
- Do NOT normalize images to [-1, 1] here.
- The trained MobileNetV3 models contain their own preprocessing.
- Images must be supplied to the models in [0, 255].
- Environmental context is decision-support information.
- Environmental context does NOT override visual AI predictions.
===============================================================
"""

import os
import json
import traceback

import numpy as np
import tensorflow as tf
from PIL import Image


# ===============================================================
# PROJECT PATHS
# ===============================================================

CURRENT_FILE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

AI_DIR = os.path.dirname(
    CURRENT_FILE_DIR
)

PROJECT_ROOT = os.path.dirname(
    AI_DIR
)

MODEL_DIR = os.path.join(
    AI_DIR,
    "models"
)


# ===============================================================
# MODEL PATHS
# ===============================================================

DISEASE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "disease_efficientnetv2s_v3.keras"
)

DISEASE_CLASSES_PATH = os.path.join(
    MODEL_DIR,
    "disease_classes.json"
)

BANANA_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "banana_nutrient_mobilenetv3.keras"
)

BANANA_CLASSES_PATH = os.path.join(
    MODEL_DIR,
    "banana_nutrient_classes.json"
)

MAIZE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "maize_nutrient_mobilenetv3.keras"
)

MAIZE_CLASSES_PATH = os.path.join(
    MODEL_DIR,
    "maize_nutrient_classes.json"
)


# ===============================================================
# IMAGE CONFIGURATION
# ===============================================================

IMAGE_SIZE = (224, 224)


# ===============================================================
# CONFIDENCE CONFIGURATION
# ===============================================================

HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.60

# Below this confidence, do not present the result
# as a confirmed specific condition.
PREDICTION_THRESHOLD = 0.60


# ===============================================================
# SUPPORTED CROPS
# ===============================================================

DISEASE_SUPPORTED_CROPS = [
    # These are the crops for which this disease model has
    # identifiable crop-specific disease classes in the current
    # 36-class mapping. Banana is intentionally excluded.
    "apple",
    "bell pepper",
    "grape",
    "maize",
    "orange",
    "potato",
    "rice",
    "squash",
    "tomato"
]

NUTRIENT_SUPPORTED_CROPS = [
    "banana",
    "maize"
]

ALL_SUPPORTED_CROPS = sorted(
    set(
        DISEASE_SUPPORTED_CROPS
        + NUTRIENT_SUPPORTED_CROPS
    )
)


# ===============================================================
# CROP -> VALID DISEASE CLASS MAPPING
# ===============================================================
#
# The disease network is a single multi-class model.  Its output
# contains diseases belonging to different crops.  A raw argmax
# therefore MUST NOT be shown as the selected disease for an
# arbitrary crop.
#
# Example of the problem this prevents:
#     selected crop = maize
#     raw model argmax = Late Blight
#     WRONG: report Late Blight for maize
#
# We only accept disease classes that are explicitly associated
# with the selected crop. Healthy is valid for every crop that
# uses the disease model.

CROP_DISEASE_CLASS_MAP = {

    "apple": {
        "Healthy",
        "Apple Rust",
        "Apple Scab"
    },

    "bell pepper": {
        "Healthy",
        "Bacterial Spot"
    },

    "grape": {
        "Healthy",
        "Black Rot",
        "Esca"
    },

    "maize": {
        "Healthy",
        "Maize Common Rust",
        "Maize Gray Leaf Spot",
        "Maize Leaf Blight",
        "Maize Northern Leaf Blight"
    },

    "orange": {
        "Healthy",
        "Citrus Greening"
    },

    "potato": {
        "Healthy",
        "Potato Early Blight",
        "Potato Late Blight"
    },

    "rice": {
        "Healthy",
        "Bacterial Leaf Blight",
        "Bacterial Leaf Streak",
        "Bacterial Panicle Blight",
        "Rice Blast",
        "Rice Brown Spot",
        "Rice Tungro"
    },

    "squash": {
        "Healthy",
        "Squash Powdery Mildew"
    },

    "tomato": {
        "Healthy",
        "Tomato Bacterial Spot",
        "Tomato Early Blight",
        "Tomato Late Blight",
        "Tomato Leaf Mold",
        "Tomato Mosaic Virus",
        "Tomato Septoria Leaf Spot",
        "Tomato Yellow Leaf Curl Virus",
        "Target Spot"
    }
}


# ===============================================================
# CROP ALIASES
# ===============================================================

CROP_ALIASES = {

    "apple": "apple",

    "banana": "banana",

    "bell pepper": "bell pepper",

    "pepper": "bell pepper",

    "pepper bell": "bell pepper",

    "pepper, bell": "pepper, bell",

    "blueberry": "blueberry",

    "cherry": "cherry",

    "cherry including sour": "cherry",

    "corn": "maize",

    "corn maize": "maize",

    "corn (maize)": "maize",

    "maize": "maize",

    "grape": "grape",

    "orange": "orange",

    "peach": "peach",

    "potato": "potato",

    "raspberry": "raspberry",

    "rice": "rice",

    "soybean": "soybean",

    "squash": "squash",

    "strawberry": "strawberry",

    "tomato": "tomato"
}


# ===============================================================
# BASIC UTILITIES
# ===============================================================

def safe_float(
    value,
    default=None
):
    """
    Safely convert a value to float.

    Prevents:
        float(None)

    from crashing the application.
    """

    if value is None:
        return default

    if isinstance(
        value,
        str
    ):
        value = value.strip()

        if value == "":
            return default

    try:

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (
        TypeError,
        ValueError
    ):

        return default


def normalize_crop_name(
    crop
):
    """
    Normalize crop name received
    from frontend/API.
    """

    if crop is None:
        return ""

    crop = str(
        crop
    ).strip().lower()

    crop = " ".join(
        crop.split()
    )

    return CROP_ALIASES.get(
        crop,
        crop
    )


def check_file(
    path,
    description
):

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def load_class_mapping(
    path
):

    check_file(
        path,
        "Class mapping"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


def get_class_name(
    classes,
    index
):

    index = int(
        index
    )

    if isinstance(
        classes,
        dict
    ):

        if str(index) in classes:

            return str(
                classes[
                    str(index)
                ]
            )

        if index in classes:

            return str(
                classes[
                    index
                ]
            )

        return f"Class {index}"

    if isinstance(
        classes,
        list
    ):

        if (
            0 <= index
            < len(classes)
        ):

            return str(
                classes[index]
            )

    return f"Class {index}"


# ===============================================================
# IMAGE LOADING
# ===============================================================

def load_image(
    image_path
):
    """
    Load image for MobileNetV3.

    VERY IMPORTANT:

    The trained MobileNetV3 models contain their own
    Rescaling layer:

        x / 127.5 - 1

    Therefore we MUST NOT perform that operation here.

    Correct input:
        [0, 255]

    Shape:
        (1, 224, 224, 3)
    """

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    try:

        image = (
            Image.open(
                image_path
            )
            .convert(
                "RGB"
            )
        )

    except Exception as error:

        raise ValueError(
            "Could not open image:\n"
            f"{image_path}\n"
            f"Error: {error}"
        )

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # -----------------------------------------------------------
    # DO NOT DO THIS:
    #
    # image_array = image_array / 127.5 - 1.0
    #
    # The model already performs this internally.
    # -----------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ===============================================================
# CONFIDENCE HELPERS
# ===============================================================

def confidence_level(
    confidence
):

    confidence = safe_float(
        confidence,
        0.0
    )

    if confidence >= HIGH_CONFIDENCE:

        return "High"

    if confidence >= MEDIUM_CONFIDENCE:

        return "Medium"

    return "Low"


def confidence_message(
    confidence
):

    confidence = safe_float(
        confidence,
        0.0
    )

    if confidence >= HIGH_CONFIDENCE:

        return (
            "High confidence prediction."
        )

    if confidence >= MEDIUM_CONFIDENCE:

        return (
            "Moderate confidence prediction. "
            "Verification is recommended."
        )

    return (
        "Low confidence prediction. "
        "Additional verification is recommended."
    )


# ===============================================================
# PREDICTION NORMALIZATION
# ===============================================================

def normalize_prediction(
    prediction
):
    """
    Convert model output to a probability vector.

    If the model already returns probabilities,
    preserve them.

    If logits are returned, apply softmax.
    """

    prediction = np.asarray(
        prediction,
        dtype=np.float32
    ).flatten()

    if prediction.size == 0:

        raise ValueError(
            "Model returned an empty prediction."
        )

    if not np.all(
        np.isfinite(
            prediction
        )
    ):

        raise ValueError(
            "Model returned invalid "
            "NaN/Inf predictions."
        )

    total = float(
        np.sum(
            prediction
        )
    )

    # Already probability-like.
    if (
        np.all(
            prediction >= 0
        )
        and abs(
            total - 1.0
        ) <= 0.05
    ):

        return prediction

    # Otherwise assume logits.
    probabilities = (
        tf.nn.softmax(
            prediction
        )
        .numpy()
    )

    return probabilities.astype(
        np.float32
    )


# ===============================================================
# TOP PREDICTIONS
# ===============================================================

def build_top_predictions(
    probabilities,
    class_mapping,
    top_k=5
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    ).flatten()

    count = min(
        top_k,
        len(probabilities)
    )

    indices = np.argsort(
        probabilities
    )[::-1][:count]

    results = []

    for index in indices:

        confidence = safe_float(
            probabilities[index],
            0.0
        )

        results.append({

            "name":
                get_class_name(
                    class_mapping,
                    int(index)
                ),

            "class_index":
                int(index),

            "confidence":
                round(
                    confidence,
                    4
                ),

            "confidence_percent":
                round(
                    confidence * 100,
                    2
                )
        })

    return results


# ===============================================================
# MAIN PREDICTOR
# ===============================================================

class CropHealthPredictor:

    def __init__(
        self,
        load_all_models=True
    ):

        print("=" * 70)
        print(
            "AI CROP HEALTH PREDICTOR"
        )
        print("=" * 70)

        print(
            "\nProject root:"
        )

        print(
            PROJECT_ROOT
        )

        print(
            "\nModel directory:"
        )

        print(
            MODEL_DIR
        )

        self.disease_model = None
        self.banana_model = None
        self.maize_model = None

        self.disease_classes = {}
        self.banana_classes = {}
        self.maize_classes = {}

        if load_all_models:

            self.load_models()


    # ===========================================================
    # MODEL LOADING
    # ===========================================================

    def load_models(
        self
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "LOADING MODELS"
        )

        print(
            "=" * 70
        )

        # -------------------------------------------------------
        # Disease
        # -------------------------------------------------------

        print(
            "\nLoading disease model..."
        )

        check_file(
            DISEASE_MODEL_PATH,
            "Disease model"
        )

        self.disease_model = (
            tf.keras.models.load_model(
                DISEASE_MODEL_PATH,
                compile=False
            )
        )

        self.disease_classes = (
            load_class_mapping(
                DISEASE_CLASSES_PATH
            )
        )

        print(
            "Disease model loaded."
        )

        print(
            "Disease classes:",
            len(
                self.disease_classes
            )
        )

        # -------------------------------------------------------
        # Banana nutrient
        # -------------------------------------------------------

        print(
            "\nLoading Banana nutrient model..."
        )

        check_file(
            BANANA_MODEL_PATH,
            "Banana nutrient model"
        )

        self.banana_model = (
            tf.keras.models.load_model(
                BANANA_MODEL_PATH
            )
        )

        self.banana_classes = (
            load_class_mapping(
                BANANA_CLASSES_PATH
            )
        )

        print(
            "Banana nutrient model loaded."
        )

        print(
            "Banana nutrient classes:",
            len(
                self.banana_classes
            )
        )

        # -------------------------------------------------------
        # Maize nutrient
        # -------------------------------------------------------

        print(
            "\nLoading Maize nutrient model..."
        )

        check_file(
            MAIZE_MODEL_PATH,
            "Maize nutrient model"
        )

        self.maize_model = (
            tf.keras.models.load_model(
                MAIZE_MODEL_PATH
            )
        )

        self.maize_classes = (
            load_class_mapping(
                MAIZE_CLASSES_PATH
            )
        )

        print(
            "Maize nutrient model loaded."
        )

        print(
            "Maize nutrient classes:",
            len(
                self.maize_classes
            )
        )

        print(
            "\nAll models loaded successfully."
        )


    # ===========================================================
    # CROP SUPPORT
    # ===========================================================

    def is_supported_crop(
        self,
        crop
    ):

        crop = normalize_crop_name(
            crop
        )

        return (
            crop in DISEASE_SUPPORTED_CROPS
            or
            crop in NUTRIENT_SUPPORTED_CROPS
        )


    def get_supported_crops(
        self
    ):

        return ALL_SUPPORTED_CROPS


    # ===========================================================
    # GENERIC MODEL PREDICTION
    # ===========================================================

    def _run_model(
        self,
        model,
        class_mapping,
        image_path,
        model_name,
        allowed_classes=None
    ):

        if model is None:

            raise RuntimeError(
                f"{model_name} is not loaded."
            )

        image = load_image(
            image_path
        )

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"RUNNING {model_name}"
        )

        print(
            "Image shape:",
            image.shape
        )

        print(
            "Image dtype:",
            image.dtype
        )

        print(
            "Image range:",
            float(
                np.min(image)
            ),
            "to",
            float(
                np.max(image)
            )
        )

        raw_prediction = (
            model.predict(
                image,
                verbose=0
            )
        )

        probabilities = (
            normalize_prediction(
                raw_prediction[0]
            )
        )

        # -------------------------------------------------------
        # CROP-SPECIFIC CLASS FILTERING
        # -------------------------------------------------------
        # The model can confidently predict a class belonging to
        # another crop. Never expose that class as the diagnosis.

        raw_model_index = int(
            np.argmax(
                probabilities
            )
        )

        raw_model_confidence = safe_float(
            probabilities[
                raw_model_index
            ],
            0.0
        )

        raw_model_class_name = (
            get_class_name(
                class_mapping,
                raw_model_index
            )
        )

        raw_top_predictions = (
            build_top_predictions(
                probabilities,
                class_mapping,
                top_k=5
            )
        )

        allowed_set = None

        if allowed_classes is not None:

            allowed_set = {
                str(name).strip().lower()
                for name in allowed_classes
            }

            valid_indices = []

            for index in range(
                len(probabilities)
            ):

                class_name = get_class_name(
                    class_mapping,
                    index
                )

                if (
                    str(class_name).strip().lower()
                    in allowed_set
                ):

                    valid_indices.append(
                        index
                    )

            if valid_indices:

                predicted_index = max(
                    valid_indices,
                    key=lambda index: probabilities[index]
                )

            else:

                predicted_index = None

        else:

            predicted_index = raw_model_index

        # -------------------------------------------------------
        # No valid crop-specific class
        # -------------------------------------------------------

        if predicted_index is None:

            print(
                "No valid crop-specific disease class was found."
            )

            return {

                "name":
                    "Uncertain",

                "raw_prediction":
                    raw_model_class_name,

                "raw_model_confidence":
                    round(
                        raw_model_confidence,
                        4
                    ),

                "class_index":
                    None,

                "confidence":
                    0.0,

                "confidence_percent":
                    0.0,

                "confidence_level":
                    "Low",

                "message":
                    (
                        f"The disease model predicted "
                        f"{raw_model_class_name} as its top class, "
                        f"but no valid class is available for the "
                        f"selected crop. No crop-specific disease "
                        f"diagnosis was made."
                    ),

                "top_predictions":
                    [],

                "raw_top_predictions":
                    raw_top_predictions,

                "crop_class_filter_applied":
                    allowed_classes is not None,

                "rejected_raw_prediction":
                    True,

                "model":
                    model_name
            }

        confidence = safe_float(
            probabilities[
                predicted_index
            ],
            0.0
        )

        raw_class_name = (
            get_class_name(
                class_mapping,
                predicted_index
            )
        )

        if allowed_classes is not None:

            filtered_probabilities = np.zeros_like(
                probabilities
            )

            for index in range(
                len(probabilities)
            ):

                class_name = get_class_name(
                    class_mapping,
                    index
                )

                if (
                    str(class_name).strip().lower()
                    in allowed_set
                ):

                    filtered_probabilities[index] = (
                        probabilities[index]
                    )

            top_predictions = (
                build_top_predictions(
                    filtered_probabilities,
                    class_mapping,
                    top_k=5
                )
            )

        else:

            top_predictions = raw_top_predictions

        print(
            "Top predictions:"
        )

        for item in top_predictions:

            print(
                f"  {item['name']}: "
                f"{item['confidence_percent']:.2f}%"
            )

        print(
            "Selected class:",
            raw_class_name
        )

        print(
            "Selected confidence:",
            f"{confidence * 100:.2f}%"
        )

        # -------------------------------------------------------
        # IMPORTANT:
        # Low confidence is not shown as a confirmed condition.
        # -------------------------------------------------------

        if confidence < PREDICTION_THRESHOLD:

            user_facing_name = (
                "Uncertain"
            )

            message = (
                f"The model's top prediction was "
                f"{raw_class_name}, but confidence was only "
                f"{confidence * 100:.2f}%. "
                f"The image does not provide sufficient "
                f"confidence for a specific identification."
            )

        else:

            user_facing_name = (
                raw_class_name
            )

            message = (
                confidence_message(
                    confidence
                )
            )

        return {

            "name":
                user_facing_name,

            "raw_prediction":
                raw_class_name,

            "class_index":
                predicted_index,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "confidence_percent":
                round(
                    confidence * 100,
                    2
                ),

            "confidence_level":
                confidence_level(
                    confidence
                ),

            "message":
                message,

            "top_predictions":
                top_predictions,

            "raw_top_predictions":
                raw_top_predictions,

            "crop_class_filter_applied":
                allowed_classes is not None,

            "rejected_raw_prediction":
                (
                    allowed_classes is not None
                    and raw_model_class_name != raw_class_name
                ),

            "model":
                model_name
        }


    # ===========================================================
    # DISEASE PREDICTION
    # ===========================================================

    def predict_disease(
        self,
        image_path,
        crop=None
    ):

        allowed_classes = None

        if crop is not None:

            normalized_crop = normalize_crop_name(
                crop
            )

            allowed_classes = CROP_DISEASE_CLASS_MAP.get(
                normalized_crop
            )

        return self._run_model(
            self.disease_model,
            self.disease_classes,
            image_path,
            "DISEASE MODEL",
            allowed_classes=allowed_classes
        )


    # ===========================================================
    # BANANA NUTRIENT
    # ===========================================================

    def predict_banana_nutrient(
        self,
        image_path
    ):

        return self._run_model(
            self.banana_model,
            self.banana_classes,
            image_path,
            "BANANA NUTRIENT MODEL"
        )


    # ===========================================================
    # MAIZE NUTRIENT
    # ===========================================================

    def predict_maize_nutrient(
        self,
        image_path
    ):

        return self._run_model(
            self.maize_model,
            self.maize_classes,
            image_path,
            "MAIZE NUTRIENT MODEL"
        )


    # ===========================================================
    # CONTEXT NORMALIZATION
    # ===========================================================

    def normalize_context(
        self,
        context
    ):

        if not isinstance(
            context,
            dict
        ):

            context = {}

        symptoms = context.get(
            "visible_symptoms",
            []
        )

        if symptoms is None:

            symptoms = []

        if isinstance(
            symptoms,
            str
        ):

            symptoms = [
                item.strip()
                for item in symptoms.split(",")
                if item.strip()
            ]

        elif not isinstance(
            symptoms,
            list
        ):

            symptoms = [
                str(symptoms)
            ]

        return {

            "growth_stage":
                context.get(
                    "growth_stage"
                ),

            "soil_condition":
                context.get(
                    "soil_condition"
                ),

            "soil_ph":
                safe_float(
                    context.get(
                        "soil_ph"
                    )
                ),

            "recent_rainfall":
                context.get(
                    "recent_rainfall"
                ),

            "irrigation":
                context.get(
                    "irrigation"
                ),

            "temperature":
                safe_float(
                    context.get(
                        "temperature"
                    )
                ),

            "humidity":
                safe_float(
                    context.get(
                        "humidity"
                    )
                ),

            "location":
                context.get(
                    "location"
                ),

            "visible_symptoms":
                symptoms,

            "weather_code":
                context.get(
                    "weather_code"
                ),

            "weather_condition":
                context.get(
                    "weather_condition"
                ),

            "wind_speed":
                safe_float(
                    context.get(
                        "wind_speed"
                    )
                ),

            "latitude":
                safe_float(
                    context.get(
                        "latitude"
                    )
                ),

            "longitude":
                safe_float(
                    context.get(
                        "longitude"
                    )
                )
        }


    # ===========================================================
    # CONTEXT ANALYSIS
    # ===========================================================

    def analyze_context(
        self,
        crop,
        context
    ):

        context = self.normalize_context(
            context
        )

        observations = []
        warnings = []

        temperature = context.get(
            "temperature"
        )

        humidity = context.get(
            "humidity"
        )

        rainfall = context.get(
            "recent_rainfall"
        )

        soil = context.get(
            "soil_condition"
        )

        soil_ph = context.get(
            "soil_ph"
        )

        irrigation = context.get(
            "irrigation"
        )

        # -------------------------------------------------------
        # Temperature
        # -------------------------------------------------------

        if temperature is not None:

            observations.append(
                f"Current temperature is "
                f"{temperature:.1f} °C."
            )

            if temperature >= 35:

                warnings.append(
                    "High temperature may increase "
                    "water demand and heat stress."
                )

            elif temperature <= 10:

                warnings.append(
                    "Low temperature may slow crop growth."
                )

        # -------------------------------------------------------
        # Humidity
        # -------------------------------------------------------

        if humidity is not None:

            observations.append(
                f"Current relative humidity is "
                f"approximately {humidity:.0f}%."
            )

            if humidity >= 80:

                warnings.append(
                    "High humidity can favor some fungal "
                    "or bacterial problems. Increase monitoring."
                )

        # -------------------------------------------------------
        # Rainfall
        # -------------------------------------------------------

        if rainfall:

            rainfall_text = str(
                rainfall
            ).lower()

            observations.append(
                f"Recent rainfall condition: "
                f"{rainfall}."
            )

            if rainfall_text in (
                "high",
                "heavy"
            ):

                warnings.append(
                    "Recent heavy rainfall increases the need "
                    "to monitor drainage, root-zone moisture "
                    "and leaf disease symptoms."
                )

        # -------------------------------------------------------
        # Soil
        # -------------------------------------------------------

        if soil:

            soil_text = str(
                soil
            ).lower()

            observations.append(
                f"Reported soil condition: "
                f"{soil}."
            )

            if soil_text in (
                "wet",
                "very wet",
                "waterlogged"
            ):

                warnings.append(
                    "Avoid unnecessary additional irrigation "
                    "while the root zone remains excessively wet."
                )

            elif soil_text in (
                "dry",
                "very dry"
            ):

                warnings.append(
                    "Check root-zone moisture before deciding "
                    "whether additional irrigation is required."
                )

        # -------------------------------------------------------
        # Soil pH
        # -------------------------------------------------------

        if soil_ph is not None:

            observations.append(
                f"Reported soil pH is "
                f"{soil_ph:.2f}."
            )

            if soil_ph < 5.0:

                warnings.append(
                    "The reported soil pH is strongly acidic. "
                    "Confirm with a soil test before amendments."
                )

            elif soil_ph > 8.0:

                warnings.append(
                    "The reported soil pH is alkaline. "
                    "Confirm with a soil test before amendments."
                )

        # -------------------------------------------------------
        # Irrigation
        # -------------------------------------------------------

        if irrigation:

            observations.append(
                f"Irrigation condition: "
                f"{irrigation}."
            )

        # -------------------------------------------------------
        # Visible symptoms
        # -------------------------------------------------------

        symptoms = context.get(
            "visible_symptoms",
            []
        )

        if symptoms:

            observations.append(
                "Reported visible symptoms: "
                + ", ".join(
                    str(x)
                    for x in symptoms
                )
                + "."
            )

        # -------------------------------------------------------
        # Crop-specific context
        # -------------------------------------------------------

        if crop == "banana":

            if (
                humidity is not None
                and humidity >= 75
            ):

                observations.append(
                    "For banana, humid conditions justify "
                    "closer monitoring of leaf disease symptoms."
                )

        if crop == "maize":

            if (
                temperature is not None
                and temperature >= 32
            ):

                observations.append(
                    "For maize, elevated temperature increases "
                    "the importance of monitoring water availability."
                )

        return {

            "available":
                bool(
                    observations
                ),

            "observations":
                observations,

            "warnings":
                warnings,

            "evidence_count":
                len(
                    observations
                )
        }


    # ===========================================================
    # CONTEXT RECOMMENDATIONS
    # ===========================================================

    def generate_context_recommendations(
        self,
        crop,
        disease,
        nutrient,
        context,
        context_analysis
    ):

        recommendations = []

        # -------------------------------------------------------
        # General
        # -------------------------------------------------------

        recommendations.append(
            "Use the AI result as screening and "
            "decision-support information rather than "
            "a confirmed agricultural diagnosis."
        )

        recommendations.append(
            "Inspect the affected plant and nearby plants "
            "for consistent symptoms."
        )

        # -------------------------------------------------------
        # Disease
        # -------------------------------------------------------

        if disease is not None:

            disease_name = str(
                disease.get(
                    "name",
                    "Uncertain"
                )
            )

            raw_disease = str(
                disease.get(
                    "raw_prediction",
                    disease_name
                )
            )

            disease_confidence = safe_float(
                disease.get(
                    "confidence"
                ),
                0.0
            )

            if disease_name.lower() == "uncertain":

                recommendations.append(
                    f"The disease model's top visual class "
                    f"was {raw_disease}, but confidence was "
                    f"{disease_confidence * 100:.1f}%. "
                    f"Capture another clear image focused on "
                    f"the affected leaf."
                )

            elif disease_name.lower() == "healthy":

                recommendations.append(
                    "The disease model did not identify a "
                    "specific disease pattern with sufficient "
                    "confidence."
                )

            else:

                recommendations.append(
                    f"The image shows visual patterns "
                    f"associated with {disease_name}. "
                    f"Confirm the condition before applying "
                    f"disease-specific treatment."
                )

        # -------------------------------------------------------
        # Nutrient
        # -------------------------------------------------------

        if nutrient is not None:

            nutrient_name = str(
                nutrient.get(
                    "name",
                    "Uncertain"
                )
            )

            raw_nutrient = str(
                nutrient.get(
                    "raw_prediction",
                    nutrient_name
                )
            )

            nutrient_confidence = safe_float(
                nutrient.get(
                    "confidence"
                ),
                0.0
            )

            if nutrient_name.lower() == "uncertain":

                recommendations.append(
                    f"The nutrient model's top class was "
                    f"{raw_nutrient}, but confidence was "
                    f"{nutrient_confidence * 100:.1f}%. "
                    f"Do not apply fertilizer based on this "
                    f"prediction alone."
                )

            elif nutrient_name.lower() == "healthy":

                recommendations.append(
                    "The nutrient model did not identify a "
                    "specific nutrient deficiency pattern."
                )

            else:

                recommendations.append(
                    f"The nutrient model indicates a possible "
                    f"{nutrient_name}. Confirm nutrient status "
                    f"using soil or plant-tissue testing before "
                    f"applying fertilizer."
                )

        # -------------------------------------------------------
        # Context observations
        # -------------------------------------------------------

        for observation in (
            context_analysis.get(
                "observations",
                []
            )
        ):

            recommendations.append(
                observation
            )

        # -------------------------------------------------------
        # Context warnings
        # -------------------------------------------------------

        for warning in (
            context_analysis.get(
                "warnings",
                []
            )
        ):

            recommendations.append(
                warning
            )

        return recommendations


    # ===========================================================
    # MAIN PREDICTION
    # ===========================================================

    def predict_crop_health(
        self,
        image_path,
        crop,
        context=None
    ):

        original_crop = crop

        crop = normalize_crop_name(
            crop
        )

        if not self.is_supported_crop(
            crop
        ):

            raise ValueError(
                f"Unsupported crop: {original_crop}\n"
                f"Supported crops: "
                f"{self.get_supported_crops()}"
            )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "NEW AI CROP HEALTH ANALYSIS"
        )

        print(
            "=" * 70
        )

        print(
            "Crop:",
            crop
        )

        print(
            "Image:",
            image_path
        )

        # -------------------------------------------------------
        # Normalize context
        # -------------------------------------------------------

        context = self.normalize_context(
            context
        )

        # -------------------------------------------------------
        # Disease
        # -------------------------------------------------------

        disease_result = None

        if (
            crop in DISEASE_SUPPORTED_CROPS
            and self.disease_model is not None
        ):

            disease_result = (
                self.predict_disease(
                    image_path,
                    crop
                )
            )

        # -------------------------------------------------------
        # Nutrient
        # -------------------------------------------------------

        nutrient_result = None

        if crop == "banana":

            if self.banana_model is not None:

                nutrient_result = (
                    self.predict_banana_nutrient(
                        image_path
                    )
                )

        elif crop == "maize":

            if self.maize_model is not None:

                nutrient_result = (
                    self.predict_maize_nutrient(
                        image_path
                    )
                )

        # -------------------------------------------------------
        # Confidence values
        # -------------------------------------------------------

        confidence_values = []

        if disease_result is not None:

            disease_confidence = safe_float(
                disease_result.get(
                    "confidence"
                )
            )

            if disease_confidence is not None:

                confidence_values.append(
                    disease_confidence
                )

        if nutrient_result is not None:

            nutrient_confidence = safe_float(
                nutrient_result.get(
                    "confidence"
                )
            )

            if nutrient_confidence is not None:

                confidence_values.append(
                    nutrient_confidence
                )

        if confidence_values:

            overall_confidence = min(
                confidence_values
            )

        else:

            overall_confidence = 0.0

        # -------------------------------------------------------
        # Status
        # -------------------------------------------------------

        disease_name = ""

        if disease_result:

            disease_name = str(
                disease_result.get(
                    "name",
                    ""
                )
            ).lower()

        nutrient_name = ""

        if nutrient_result:

            nutrient_name = str(
                nutrient_result.get(
                    "name",
                    ""
                )
            ).lower()

        verification_required = (
            overall_confidence
            < MEDIUM_CONFIDENCE
        )

        if (
            disease_name == "healthy"
            and (
                nutrient_result is None
                or nutrient_name == "healthy"
            )
            and not verification_required
        ):

            status = "Healthy"

        elif verification_required:

            status = (
                "Requires verification"
            )

        elif (
            disease_name == "uncertain"
            or nutrient_name == "uncertain"
        ):

            status = (
                "Requires verification"
            )

        else:

            status = (
                "Potential health issue detected"
            )

        # -------------------------------------------------------
        # Context analysis
        # -------------------------------------------------------

        context_analysis = (
            self.analyze_context(
                crop,
                context
            )
        )

        # -------------------------------------------------------
        # Recommendations
        # -------------------------------------------------------

        recommendations = (
            self.generate_context_recommendations(
                crop,
                disease_result,
                nutrient_result,
                context,
                context_analysis
            )
        )

        # -------------------------------------------------------
        # Final message
        # -------------------------------------------------------

        if verification_required:

            overall_message = (
                "The available image evidence has "
                "low confidence. Additional plant "
                "inspection or testing is recommended."
            )

        else:

            overall_message = (
                "The AI system found visual evidence. "
                "Environmental information is provided "
                "as supporting decision context and does "
                "not override the image-based prediction."
            )

        # -------------------------------------------------------
        # Final result
        # -------------------------------------------------------

        result = {

            "crop":
                crop.title(),

            "image_path":
                image_path,

            "context":
                context,

            "disease":
                disease_result,

            "nutrient":
                nutrient_result,

            "overall_confidence":
                round(
                    overall_confidence,
                    4
                ),

            "overall_confidence_percent":
                round(
                    overall_confidence * 100,
                    2
                ),

            "status":
                status,

            "verification_required":
                verification_required,

            "overall_message":
                overall_message,

            "context_analysis":
                context_analysis,

            "recommendations":
                recommendations
        }

        print(
            "\n"
            + "=" * 70
        )

        print(
            "PREDICTION COMPLETED"
        )

        print(
            "=" * 70
        )

        print(
            "Disease:",
            (
                disease_result.get(
                    "name"
                )
                if disease_result
                else "Unavailable"
            )
        )

        if disease_result:

            print(
                "Disease confidence:",
                f"{safe_float(disease_result.get('confidence'), 0) * 100:.2f}%"
            )

        print(
            "Nutrient:",
            (
                nutrient_result.get(
                    "name"
                )
                if nutrient_result
                else "Unavailable"
            )
        )

        if nutrient_result:

            print(
                "Nutrient confidence:",
                f"{safe_float(nutrient_result.get('confidence'), 0) * 100:.2f}%"
            )

        print(
            "Overall confidence:",
            f"{overall_confidence * 100:.2f}%"
        )

        print(
            "Status:",
            status
        )

        print(
            "=" * 70
        )

        return result


# ===============================================================
# SIMPLE CLI RESULT PRINTER
# ===============================================================

def print_prediction_result(
    result
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CROP HEALTH PREDICTION"
    )

    print(
        "=" * 70
    )

    print(
        "\nCrop:",
        result.get(
            "crop"
        )
    )

    disease = result.get(
        "disease"
    )

    nutrient = result.get(
        "nutrient"
    )

    if disease:

        print(
            "\nDisease / Health:"
        )

        print(
            "  Prediction:",
            disease.get(
                "name"
            )
        )

        print(
            "  Raw prediction:",
            disease.get(
                "raw_prediction"
            )
        )

        print(
            "  Confidence:",
            disease.get(
                "confidence_percent"
            ),
            "%"
        )

    if nutrient:

        print(
            "\nNutrient:"
        )

        print(
            "  Prediction:",
            nutrient.get(
                "name"
            )
        )

        print(
            "  Raw prediction:",
            nutrient.get(
                "raw_prediction"
            )
        )

        print(
            "  Confidence:",
            nutrient.get(
                "confidence_percent"
            ),
            "%"
        )

    print(
        "\nOverall:",
        result.get(
            "status"
        )
    )

    print(
        "Overall confidence:",
        result.get(
            "overall_confidence_percent"
        ),
        "%"
    )

    print(
        "\nRecommendations:"
    )

    for recommendation in (
        result.get(
            "recommendations",
            []
        )
    ):

        print(
            " -",
            recommendation
        )


# ===============================================================
# DIRECT TEST
# ===============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CROP HEALTH PREDICTOR TEST"
    )

    print(
        "=" * 70
    )

    print(
        "\nModels are loaded when "
        "CropHealthPredictor() is created."
    )

    print(
        "Use the API or call "
        "predict_crop_health() with an image."
    )