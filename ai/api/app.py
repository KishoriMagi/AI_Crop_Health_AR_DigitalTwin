import os
import sys
import uuid
import json
import traceback
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

AI_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(AI_DIR)

if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)


# ============================================================
# IMPORTS
# ============================================================

from inference.crop_health_predictor import (
    CropHealthPredictor,
    ALL_SUPPORTED_CROPS,
    DISEASE_SUPPORTED_CROPS,
    NUTRIENT_SUPPORTED_CROPS,
    normalize_crop_name,
)

from inference.recommendation_engine import (
    RecommendationEngine,
)

from services.weather_service import (
    WeatherService,
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "*"
        }
    }
)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_FOLDER = os.path.join(
    AI_DIR,
    "uploads"
)

DIGITAL_TWIN_FOLDER = os.path.join(
    AI_DIR,
    "digital_twin"
)

DIGITAL_TWIN_FILE = os.path.join(
    DIGITAL_TWIN_FOLDER,
    "farm_history.json"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DIGITAL_TWIN_FOLDER, exist_ok=True)


# ============================================================
# INITIALIZE AI SERVICES
# ============================================================

print("=" * 70)
print("INITIALIZING AI CROP HEALTH API")
print("=" * 70)

print("\nLoading AI predictor...")
predictor = CropHealthPredictor()

print("\nLoading recommendation engine...")
recommendation_engine = RecommendationEngine()

print("\nLoading weather service...")
weather_service = WeatherService()

print("\nAll AI services loaded successfully.")
print("=" * 70)


# ============================================================
# GENERAL HELPERS
# ============================================================

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_float(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value):
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_visible_symptoms(value):
    if not value:
        return []

    if isinstance(value, list):
        return value

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def json_safe(value):
    """
    Convert common NumPy/TensorFlow scalar values into
    normal Python JSON-compatible values.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


# ============================================================
# DIGITAL TWIN STORAGE
# ============================================================

def load_digital_twins():
    """
    Persistent local farm history.

    This intentionally uses one JSON file so the project can
    demonstrate persistence without introducing a database yet.
    """

    if not os.path.exists(DIGITAL_TWIN_FILE):
        return []

    try:
        with open(
            DIGITAL_TWIN_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except (
        OSError,
        json.JSONDecodeError,
        TypeError
    ) as error:

        print(
            "Digital twin history could not be loaded:",
            error
        )

        return []


def save_digital_twins(twins):
    """
    Atomically write farm history to disk.
    """

    temporary_file = (
        DIGITAL_TWIN_FILE + ".tmp"
    )

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_safe(twins),
            file,
            indent=2,
            ensure_ascii=False
        )

    os.replace(
        temporary_file,
        DIGITAL_TWIN_FILE
    )


def create_field_id():
    return (
        "FIELD-"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:6].upper()
    )


def find_digital_twin(field_id):
    twins = load_digital_twins()

    for twin in twins:
        if twin.get("fieldId") == field_id:
            return twin

    return None


def save_digital_twin(twin):
    twins = load_digital_twins()

    # Replace an existing field record when the same field ID
    # is explicitly supplied.
    replaced = False

    for index, existing in enumerate(twins):
        if existing.get("fieldId") == twin.get("fieldId"):
            twins[index] = twin
            replaced = True
            break

    if not replaced:
        twins.append(twin)

    save_digital_twins(twins)

    return twin


# ============================================================
# CONTEXT HELPERS
# ============================================================

def get_coordinates_from_request():
    latitude = parse_float(
        request.form.get("latitude")
    )

    longitude = parse_float(
        request.form.get("longitude")
    )

    if latitude is None or longitude is None:
        return None

    return {
        "latitude": latitude,
        "longitude": longitude
    }


def get_coordinates_from_query():
    latitude = parse_float(
        request.args.get("latitude")
    )

    longitude = parse_float(
        request.args.get("longitude")
    )

    if latitude is None or longitude is None:
        return None

    return {
        "latitude": latitude,
        "longitude": longitude
    }


def get_real_time_weather_from_coordinates(
    coordinates
):
    if not coordinates:
        return None

    try:
        weather = weather_service.get_weather(
            coordinates["latitude"],
            coordinates["longitude"]
        )

        if not isinstance(weather, dict):
            return None

        if not weather.get("success", False):
            return None

        return weather

    except Exception as error:
        print(
            "Weather retrieval error:",
            error
        )
        traceback.print_exc()
        return None


def build_context():
    """
    Build the agricultural + environmental context used by
    crop_health_predictor.py and recommendation_engine.py.

    Weather supplied by the WeatherService takes precedence over
    empty/manual weather values.
    """

    coordinates = (
        get_coordinates_from_request()
    )

    context = {
        # Agricultural context
        "growth_stage":
            request.form.get(
                "growth_stage"
            ) or "vegetative",

        "soil_condition":
            request.form.get(
                "soil_condition"
            ) or "unknown",

        "soil_ph":
            parse_float(
                request.form.get(
                    "soil_ph"
                )
            ),

        "recent_rainfall":
            request.form.get(
                "recent_rainfall"
            ) or "unknown",

        "irrigation":
            request.form.get(
                "irrigation"
            ) or "unknown",

        "visible_symptoms":
            parse_visible_symptoms(
                request.form.get(
                    "visible_symptoms"
                )
            ),

        "location":
            request.form.get(
                "location"
            ) or "GPS location",

        # Weather
        "temperature":
            parse_float(
                request.form.get(
                    "temperature"
                )
            ),

        "humidity":
            parse_float(
                request.form.get(
                    "humidity"
                )
            ),

        "weather_condition":
            request.form.get(
                "weather_condition"
            ) or "unknown",

        "wind_speed":
            parse_float(
                request.form.get(
                    "wind_speed"
                )
            ),

        "weather_code":
            None,

        # GPS
        "latitude":
            coordinates["latitude"]
            if coordinates
            else None,

        "longitude":
            coordinates["longitude"]
            if coordinates
            else None,
    }

    # --------------------------------------------------------
    # REAL-TIME WEATHER
    # --------------------------------------------------------

    weather = (
        get_real_time_weather_from_coordinates(
            coordinates
        )
    )

    if weather:
        context["_weather"] = weather

        if weather.get("temperature") is not None:
            context["temperature"] = parse_float(
                weather.get("temperature")
            )

        if weather.get("humidity") is not None:
            context["humidity"] = parse_float(
                weather.get("humidity")
            )

        rainfall = (
            weather.get("rainfall_status")
            or weather.get("recent_rainfall")
        )

        if rainfall is not None:
            context["recent_rainfall"] = rainfall

        condition = (
            weather.get("condition")
            or weather.get("weather_condition")
        )

        if condition is not None:
            context["weather_condition"] = condition

        if weather.get("wind_speed") is not None:
            context["wind_speed"] = parse_float(
                weather.get("wind_speed")
            )

        if weather.get("weather_code") is not None:
            context["weather_code"] = weather.get(
                "weather_code"
            )

    return context


def public_context(context):
    """
    Remove internal fields before sending context to React.
    """

    result = dict(context or {})
    result.pop("_weather", None)

    return json_safe(result)


# ============================================================
# RECOMMENDATION HELPERS
# ============================================================

def generate_recommendations(
    prediction,
    context
):
    """
    Use the actual RecommendationEngine already present in the
    project. The compatibility fallback keeps the API working
    if the method is called with only prediction.
    """

    try:

        result = (
            recommendation_engine
            .generate_recommendations(
                prediction,
                context
            )
        )

    except TypeError:

        print(
            "RecommendationEngine compatibility mode."
        )

        result = (
            recommendation_engine
            .generate_recommendations(
                prediction
            )
        )

    if isinstance(result, dict):

        recommendations = result.get(
            "recommendations",
            []
        )

    elif isinstance(result, list):

        recommendations = result

        result = {
            "recommendations":
                recommendations
        }

    else:

        recommendations = []

        result = {
            "recommendations":
                []
        }

    if not isinstance(
        recommendations,
        list
    ):

        recommendations = [
            str(recommendations)
        ]

    cleaned = []

    for item in recommendations:

        if item is None:
            continue

        text = str(item).strip()

        if text and text not in cleaned:
            cleaned.append(text)

    return (
        cleaned,
        json_safe(result)
    )


# ============================================================
# DIGITAL TWIN BUILDER
# ============================================================

def build_digital_twin(
    crop,
    results,
    context,
    field_id=None
):
    """
    Convert the image-level AI results into a persistent
    virtual representation of the analyzed field.

    Each uploaded crop image becomes one plant/observation node.
    """

    if not field_id:
        field_id = create_field_id()

    public_env = {
        "temperature":
            context.get("temperature"),

        "humidity":
            context.get("humidity"),

        "rainfall":
            context.get("recent_rainfall"),

        "weatherCondition":
            context.get("weather_condition"),

        "windSpeed":
            context.get("wind_speed"),

        "soilCondition":
            context.get("soil_condition"),

        "soilPH":
            context.get("soil_ph"),

        "irrigation":
            context.get("irrigation"),

        "growthStage":
            context.get("growth_stage"),
    }

    plants = []

    for index, item in enumerate(
        results,
        start=1
    ):

        prediction = (
            item.get("prediction")
            or {}
        )

        disease = (
            prediction.get("disease")
            or {}
        )

        nutrient = (
            prediction.get("nutrient")
            or {}
        )

        overall = (
            prediction.get("overall")
            or {}
        )

        plant = {
            "id": index,

            "imageName":
                item.get("imageName"),

            "crop":
                item.get("crop")
                or prediction.get(
                    "crop",
                    crop
                ),

            "disease":
                disease.get(
                    "name",
                    "Not available"
                ),

            "diseaseConfidence":
                disease.get(
                    "confidence_percent"
                ),

            "nutrient":
                nutrient.get(
                    "name",
                    "Not evaluated"
                ),

            "nutrientConfidence":
                nutrient.get(
                    "confidence_percent"
                ),

            "overallConfidence":
                prediction.get(
                    "overall_confidence_percent",
                    overall.get(
                        "confidence"
                    )
                ),

            "status":
                prediction.get(
                    "status",
                    overall.get(
                        "status",
                        "Requires verification"
                    )
                ),

            "latitude":
                context.get("latitude"),

            "longitude":
                context.get("longitude"),

            "environment":
                public_env,
        }

        plants.append(
            plant
        )

    twin = {
        "fieldId": field_id,

        "createdAt":
            now_iso(),

        "updatedAt":
            now_iso(),

        "crop":
            crop,

        "plantCount":
            len(plants),

        "location": {
            "latitude":
                context.get("latitude"),

            "longitude":
                context.get("longitude"),
        },

        "weather": {
            "temperature":
                context.get("temperature"),

            "humidity":
                context.get("humidity"),

            "rainfall_status":
                context.get(
                    "recent_rainfall"
                ),

            "condition":
                context.get(
                    "weather_condition"
                ),

            "wind_speed":
                context.get(
                    "wind_speed"
                ),
        },

        "environment":
            public_env,

        "plants":
            plants,

        "history": [
            {
                "timestamp":
                    now_iso(),

                "event":
                    "field_analysis",

                "plantCount":
                    len(plants),

                "crop":
                    crop,
            }
        ],
    }

    return json_safe(twin)


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({
        "application":
            "AI Crop Health AR Digital Twin",

        "status":
            "running",

        "service":
            "AI Prediction API",

        "digital_twin":
            True,
    })


# ============================================================
# STATUS
# ============================================================

@app.route(
    "/api/status",
    methods=["GET"]
)
def status():

    return jsonify({

        "status":
            "ready",

        "supported_crops":
            ALL_SUPPORTED_CROPS,

        "disease_supported_crops":
            DISEASE_SUPPORTED_CROPS,

        "nutrient_supported_crops":
            NUTRIENT_SUPPORTED_CROPS,

        "disease_model": {

            "available":
                predictor.disease_model is not None,

            "name":
                "MobileNetV3",

            "classes":
                len(
                    predictor.disease_classes
                )
        },

        "nutrient_models": {

            "banana":
                predictor.banana_model is not None,

            "maize":
                predictor.maize_model is not None
        },

        "weather_service": {

            "provider":
                "Open-Meteo",

            "available":
                weather_service is not None
        },

        "recommendation_engine": {

            "available":
                recommendation_engine is not None,

            "name":
                recommendation_engine.name
        },

        "digital_twin": {

            "enabled":
                True,

            "persistent_storage":
                DIGITAL_TWIN_FILE,

            "field_count":
                len(
                    load_digital_twins()
                )
        },

        "context_features": [

            "latitude",
            "longitude",
            "temperature",
            "humidity",
            "recent_rainfall",
            "weather_condition",
            "wind_speed",
            "growth_stage",
            "soil_condition",
            "soil_ph",
            "irrigation",
            "location",
            "visible_symptoms"
        ],

        "important_note":
            (
                "Disease and nutrient predictions are model-based. "
                "Environmental context is decision-support information "
                "and does not replace field or laboratory verification."
            )
    })


# ============================================================
# WEATHER ENDPOINT
# ============================================================

@app.route(
    "/api/weather",
    methods=["GET"]
)
def weather_endpoint():

    coordinates = (
        get_coordinates_from_query()
    )

    if not coordinates:

        return jsonify({
            "success":
                False,

            "error":
                "latitude and longitude are required."
        }), 400

    try:

        weather = (
            weather_service.get_weather(
                coordinates["latitude"],
                coordinates["longitude"]
            )
        )

        if not isinstance(
            weather,
            dict
        ):

            return jsonify({
                "success":
                    False,

                "error":
                    "Weather service returned invalid data."
            }), 502

        if not weather.get(
            "success",
            False
        ):

            return jsonify(
                weather
            ), 502

        return jsonify(
            json_safe(weather)
        )

    except Exception as error:

        print(
            "\nWEATHER ENDPOINT ERROR:",
            error
        )

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(error),

            "error_type":
                type(error).__name__
        }), 500


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    filepath = None

    try:

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "success":
                    False,

                "error":
                    "No image uploaded."
            }), 400

        image = request.files["image"]

        if not image.filename:

            return jsonify({
                "success":
                    False,

                "error":
                    "Invalid image filename."
            }), 400

        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        crop_input = request.form.get(
            "crop",
            ""
        ).strip()

        if not crop_input:

            return jsonify({

                "success":
                    False,

                "error":
                    "Please select the crop type.",

                "supported_crops":
                    ALL_SUPPORTED_CROPS
            }), 400

        crop = normalize_crop_name(
            crop_input
        )

        if crop not in ALL_SUPPORTED_CROPS:

            return jsonify({

                "success":
                    False,

                "error":
                    f"Unsupported crop: {crop_input}",

                "supported_crops":
                    ALL_SUPPORTED_CROPS
            }), 400

        # ----------------------------------------------------
        # IMAGE FORMAT
        # ----------------------------------------------------

        extension = os.path.splitext(
            image.filename
        )[1].lower()

        allowed_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }

        if extension not in allowed_extensions:

            return jsonify({

                "success":
                    False,

                "error":
                    "Unsupported image format. "
                    "Use JPG, JPEG, PNG or WEBP."
            }), 400

        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        safe_filename = (
            uuid.uuid4().hex
            + extension
        )

        filepath = os.path.join(
            UPLOAD_FOLDER,
            safe_filename
        )

        image.save(
            filepath
        )

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        context = build_context()

        print("\n" + "=" * 70)
        print("NEW CROP HEALTH PREDICTION")
        print("=" * 70)
        print("Crop:", crop)
        print("Image:", safe_filename)

        print("\nCONTEXT")
        print("-" * 70)

        for key, value in public_context(
            context
        ).items():

            print(
                f"{key}: {value}"
            )

        # ----------------------------------------------------
        # AI PREDICTION
        # ----------------------------------------------------

        prediction = (
            predictor.predict_crop_health(
                filepath,
                crop,
                context
            )
        )

        prediction = json_safe(
            prediction
        )

        # ----------------------------------------------------
        # CROP-SPECIFIC RECOMMENDATIONS
        # ----------------------------------------------------

        (
            actual_recommendations,
            recommendation_result
        ) = generate_recommendations(
            prediction,
            context
        )

        # Keep the recommendation engine output in the API
        # response while making the frontend-friendly array
        # available at the top level.
        recommendation_details = {

            "engine":
                recommendation_engine.name,

            "assessment":
                recommendation_result.get(
                    "assessment",
                    {}
                ),

            "context_analysis":
                recommendation_result.get(
                    "context_analysis",
                    {}
                ),

            "system_note":
                recommendation_result.get(
                    "system_note"
                )
        }

        response = {

            "success":
                True,

            "crop":
                prediction.get(
                    "crop",
                    crop
                ),

            "context":
                public_context(
                    context
                ),

            "weather":
                json_safe(
                    context.get("_weather")
                ),

            "prediction":
                prediction,

            "recommendations":
                actual_recommendations,

            "recommendation_details":
                json_safe(
                    recommendation_details
                ),

            "image":
                {
                    "filename":
                        safe_filename,

                    "original_name":
                        image.filename,
                }
        }

        print(
            "\nRecommendations sent:",
            len(
                actual_recommendations
            )
        )

        return jsonify(
            json_safe(response)
        )

    except Exception as error:

        print(
            "\n" + "=" * 70
        )
        print("PREDICTION ERROR")
        print("=" * 70)
        print(
            type(error).__name__,
            ":",
            str(error)
        )

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(error),

            "error_type":
                type(error).__name__,

            "stage":
                "prediction"
        }), 500


# ============================================================
# SERVE UPLOADED IMAGE
# ============================================================

@app.route(
    "/api/uploads/<path:filename>",
    methods=["GET"]
)
def uploaded_image(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# DIGITAL TWIN — CREATE / SAVE
# ============================================================

@app.route(
    "/api/digital-twin",
    methods=["POST"]
)
def create_digital_twin_endpoint():

    try:

        payload = request.get_json(
            silent=True
        ) or {}

        crop = str(
            payload.get(
                "crop",
                ""
            )
        ).strip()

        if not crop:

            return jsonify({

                "success":
                    False,

                "error":
                    "crop is required."
            }), 400

        crop = normalize_crop_name(
            crop
        )

        if crop not in ALL_SUPPORTED_CROPS:

            return jsonify({

                "success":
                    False,

                "error":
                    f"Unsupported crop: {crop}",

                "supported_crops":
                    ALL_SUPPORTED_CROPS
            }), 400

        # ----------------------------------------------------
        # The frontend can send already analyzed plants.
        # ----------------------------------------------------

        plants = payload.get(
            "plants",
            []
        )

        if not isinstance(
            plants,
            list
        ):
            plants = []

        context = payload.get(
            "context",
            {}
        )

        if not isinstance(
            context,
            dict
        ):
            context = {}

        location = payload.get(
            "location"
        )

        weather = payload.get(
            "weather"
        )

        field_id = payload.get(
            "fieldId"
        )

        twin = build_digital_twin(
            crop=crop,
            results=[
                {
                    "prediction": {
                        "crop":
                            item.get(
                                "crop",
                                crop
                            ),

                        "disease": {
                            "name":
                                item.get(
                                    "disease",
                                    "Not available"
                                ),

                            "confidence_percent":
                                item.get(
                                    "diseaseConfidence"
                                )
                        },

                        "nutrient": {
                            "name":
                                item.get(
                                    "nutrient",
                                    "Not evaluated"
                                ),

                            "confidence_percent":
                                item.get(
                                    "nutrientConfidence"
                                )
                        },

                        "overall": {
                            "confidence":
                                item.get(
                                    "overallConfidence"
                                ),

                            "status":
                                item.get(
                                    "status",
                                    "Requires verification"
                                )
                        }
                    },

                    "imageName":
                        item.get(
                            "imageName"
                        ),

                    "crop":
                        item.get(
                            "crop",
                            crop
                        )
                }

                for item in plants
                if isinstance(
                    item,
                    dict
                )
            ],
            context={
                **context,

                "latitude":
                    (
                        location or {}
                    ).get(
                        "latitude",
                        context.get(
                            "latitude"
                        )
                    ),

                "longitude":
                    (
                        location or {}
                    ).get(
                        "longitude",
                        context.get(
                            "longitude"
                        )
                    ),

                "temperature":
                    (
                        weather or {}
                    ).get(
                        "temperature",
                        context.get(
                            "temperature"
                        )
                    ),

                "humidity":
                    (
                        weather or {}
                    ).get(
                        "humidity",
                        context.get(
                            "humidity"
                        )
                    ),

                "recent_rainfall":
                    (
                        weather or {}
                    ).get(
                        "rainfall_status",
                        context.get(
                            "recent_rainfall"
                        )
                    ),

                "weather_condition":
                    (
                        weather or {}
                    ).get(
                        "condition",
                        context.get(
                            "weather_condition"
                        )
                    ),

                "wind_speed":
                    (
                        weather or {}
                    ).get(
                        "wind_speed",
                        context.get(
                            "wind_speed"
                        )
                    ),

                "soil_condition":
                    context.get(
                        "soil_condition",
                        "unknown"
                    ),

                "soil_ph":
                    context.get(
                        "soil_ph"
                    ),

                "irrigation":
                    context.get(
                        "irrigation",
                        "unknown"
                    ),

                "growth_stage":
                    context.get(
                        "growth_stage",
                        "vegetative"
                    ),
            },
            field_id=field_id
        )

        # Preserve explicit frontend weather/location objects.
        if location is not None:
            twin["location"] = json_safe(
                location
            )

        if weather is not None:
            twin["weather"] = json_safe(
                weather
            )

        twin["updatedAt"] = now_iso()

        save_digital_twin(
            twin
        )

        return jsonify({

            "success":
                True,

            "digitalTwin":
                twin
        })

    except Exception as error:

        print(
            "\nDIGITAL TWIN CREATE ERROR:",
            error
        )

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(error),

            "error_type":
                type(error).__name__,

            "stage":
                "digital_twin"
        }), 500


# ============================================================
# DIGITAL TWIN — LIST / HISTORY
# ============================================================

@app.route(
    "/api/digital-twin",
    methods=["GET"]
)
def list_digital_twins():

    try:

        twins = load_digital_twins()

        # Most recent field first.
        twins = list(
            reversed(twins)
        )

        return jsonify({

            "success":
                True,

            "count":
                len(twins),

            "digitalTwins":
                json_safe(twins)
        })

    except Exception as error:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(error)
        }), 500


@app.route(
    "/api/digital-twin/history",
    methods=["GET"]
)
def digital_twin_history():

    try:

        twins = load_digital_twins()

        history = []

        for twin in twins:

            history.extend(
                twin.get(
                    "history",
                    []
                )
                if isinstance(
                    twin.get(
                        "history",
                        []
                    ),
                    list
                )
                else []
            )

        history.sort(
            key=lambda item:
                item.get(
                    "timestamp",
                    ""
                ),
            reverse=True
        )

        return jsonify({

            "success":
                True,

            "count":
                len(history),

            "history":
                json_safe(history)
        })

    except Exception as error:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(error)
        }), 500


# ============================================================
# DIGITAL TWIN — GET ONE FIELD
# ============================================================

@app.route(
    "/api/digital-twin/<field_id>",
    methods=["GET"]
)
def get_digital_twin(field_id):

    twin = find_digital_twin(
        field_id
    )

    if twin is None:

        return jsonify({

            "success":
                False,

            "error":
                "Digital twin field not found.",

            "fieldId":
                field_id
        }), 404

    return jsonify({

        "success":
            True,

        "digitalTwin":
            json_safe(twin)
    })


# ============================================================
# DIGITAL TWIN — DELETE ONE FIELD
# ============================================================

@app.route(
    "/api/digital-twin/<field_id>",
    methods=["DELETE"]
)
def delete_digital_twin(field_id):

    twins = load_digital_twins()

    filtered = [
        twin
        for twin in twins
        if twin.get(
            "fieldId"
        ) != field_id
    ]

    if len(filtered) == len(twins):

        return jsonify({

            "success":
                False,

            "error":
                "Digital twin field not found.",

            "fieldId":
                field_id
        }), 404

    save_digital_twins(
        filtered
    )

    return jsonify({

        "success":
            True,

        "message":
            "Digital twin deleted.",

        "fieldId":
            field_id
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 70
    )

    print(
        "AI CROP HEALTH AR DIGITAL TWIN API"
    )

    print(
        "=" * 70
    )

    print(
        "Server:",
        "http://127.0.0.1:5000"
    )

    print(
        "\nDigital twin storage:",
        DIGITAL_TWIN_FILE
    )

    print(
        "\nAll supported crops:"
    )

    for crop_name in ALL_SUPPORTED_CROPS:
        print(
            " -",
            crop_name
        )

    print(
        "\nDisease-supported crops:"
    )

    for crop_name in DISEASE_SUPPORTED_CROPS:
        print(
            " -",
            crop_name
        )

    print(
        "\nNutrient-supported crops:"
    )

    for crop_name in NUTRIENT_SUPPORTED_CROPS:
        print(
            " -",
            crop_name
        )

    print(
        "\nRecommendation engine:",
        recommendation_engine.name
    )

    print(
        "\nDigital twin endpoints:"
    )

    print(
        " - POST   /api/digital-twin"
    )

    print(
        " - GET    /api/digital-twin"
    )

    print(
        " - GET    /api/digital-twin/history"
    )

    print(
        " - GET    /api/digital-twin/<field_id>"
    )

    print(
        " - DELETE /api/digital-twin/<field_id>"
    )

    print(
        "\nStarting Flask server..."
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )
