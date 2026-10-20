"""
================================================================
AI CROP HEALTH
CROP-SPECIFIC RECOMMENDATION ENGINE
================================================================

Purpose
-------
1. Receive disease and nutrient predictions.
2. Generate crop-specific actionable recommendations.
3. Give practical "what to do" guidance.
4. Use environmental context as supporting information.
5. Never convert None directly to float.
6. Preserve compatibility with existing app.py.
7. Do not modify trained AI models.
8. Do not treat an AI prediction as a confirmed diagnosis.

IMPORTANT
---------
The disease model contains shared disease classes.
Therefore, the engine checks whether a predicted disease
matches the selected crop.

For example:

Banana + Maize Leaf Blight
    -> possible crop/prediction mismatch
    -> verification required
    -> do NOT blindly recommend maize treatment.

Maize + Maize Leaf Blight
    -> crop/disease match
    -> provide maize-specific management.

================================================================
"""

from typing import Optional, Dict, List


# ================================================================
# SAFE VALUE HELPERS
# ================================================================

def safe_float(value, default=None):

    if value is None:
        return default

    if isinstance(value, str):

        value = value.strip()

        if value == "":
            return default

    try:
        return float(value)

    except (TypeError, ValueError):

        return default


def normalize_confidence(value):

    """
    Accepts:

        0.73
        73
        "73"
        "73%"

    Returns:

        0.73
    """

    if isinstance(value, str):

        value = value.strip()

        if value.endswith("%"):

            value = value[:-1]

    value = safe_float(value)

    if value is None:

        return None

    if value > 1:

        value = value / 100.0

    return max(
        0.0,
        min(1.0, value)
    )


def get_confidence_level(confidence):

    confidence = normalize_confidence(
        confidence
    )

    if confidence is None:

        return "Unknown"

    if confidence >= 0.80:

        return "High"

    if confidence >= 0.60:

        return "Medium"

    return "Low"


def confidence_message(confidence):

    confidence = normalize_confidence(
        confidence
    )

    if confidence is None:

        return (
            "No valid confidence value was available."
        )

    if confidence >= 0.80:

        return (
            "High confidence prediction. "
            "Independent verification is still recommended."
        )

    if confidence >= 0.60:

        return (
            "Moderate confidence prediction. "
            "Verification is recommended."
        )

    return (
        "Low confidence prediction. "
        "Additional verification is required."
    )


# ================================================================
# CROP NORMALIZATION
# ================================================================

CROP_ALIASES = {

    "corn": "maize",

    "corn maize": "maize",

    "corn (maize)": "maize",

    "maize": "maize",

    "banana": "banana",

    "apple": "apple",

    "potato": "potato",

    "rice": "rice",

    "tomato": "tomato",

    "grape": "grape",

    "orange": "orange",

    "peach": "peach",

    "squash": "squash",

    "strawberry": "strawberry",

    "blueberry": "blueberry",

    "raspberry": "raspberry",

    "cherry": "cherry",

    "bell pepper": "bell pepper",

    "pepper": "bell pepper"

}


def normalize_crop(crop):

    if crop is None:

        return ""

    crop = str(crop).strip().lower()

    crop = " ".join(
        crop.split()
    )

    return CROP_ALIASES.get(
        crop,
        crop
    )


def display_crop(crop):

    names = {

        "banana": "Banana",

        "maize": "Maize",

        "apple": "Apple",

        "potato": "Potato",

        "rice": "Rice",

        "tomato": "Tomato",

        "grape": "Grape",

        "orange": "Orange",

        "peach": "Peach",

        "squash": "Squash",

        "strawberry": "Strawberry",

        "blueberry": "Blueberry",

        "raspberry": "Raspberry",

        "cherry": "Cherry",

        "bell pepper": "Bell Pepper"

    }

    return names.get(
        crop,
        crop.title()
    )


# ================================================================
# CROP-SPECIFIC DISEASE MAP
# ================================================================
#
# This is intentionally conservative.
#
# If a disease is not listed for a crop, the engine does NOT
# assume that the disease belongs to that crop.
#
# This prevents:
#
# Banana -> Maize Leaf Blight
#
# from being treated as a confirmed banana disease.
# ================================================================

CROP_DISEASES = {

    "apple": {

        "apple rust",

        "apple scab",

        "black rot",

        "powdery mildew",

        "fire blight"

    },

    "maize": {

        "maize common rust",

        "maize gray leaf spot",

        "maize leaf blight",

        "maize northern leaf blight",

        "bacterial leaf blight",

        "bacterial leaf streak"

    },

    "potato": {

        "potato early blight",

        "potato late blight",

        "early blight",

        "late blight"

    },

    "rice": {

        "rice blast",

        "rice brown spot",

        "rice tungro",

        "bacterial leaf blight",

        "bacterial leaf streak",

        "bacterial panicle blight"

    },

    "tomato": {

        "tomato bacterial spot",

        "tomato early blight",

        "tomato late blight",

        "tomato leaf mold",

        "tomato mosaic virus",

        "tomato septoria leaf spot",

        "tomato yellow leaf curl virus",

        "bacterial spot",

        "early blight",

        "late blight",

        "leaf mold",

        "septoria leaf spot"

    },

    "squash": {

        "squash powdery mildew",

        "powdery mildew"

    },

    "orange": {

        "citrus greening",

        "citrus canker",

        "powdery mildew"

    },

    "grape": {

        "black rot",

        "esca",

        "leaf blight",

        "downy mildew",

        "powdery mildew"

    },

    "peach": {

        "bacterial spot"

    },

    "bell pepper": {

        "bacterial spot"

    }

}


# ================================================================
# DISEASE MANAGEMENT KNOWLEDGE BASE
# ================================================================

DISEASE_MANAGEMENT = {

    # ------------------------------------------------------------
    # MAIZE
    # ------------------------------------------------------------

    "maize common rust": [

        "Inspect additional maize leaves for orange-brown rust pustules.",

        "Remove severely affected leaves where practical and manage infected crop residue.",

        "Maintain balanced crop nutrition and avoid unnecessary plant stress.",

        "Improve field airflow and avoid prolonged leaf wetness.",

        "Monitor nearby maize plants because rust can spread within the field.",

        "If Maize Common Rust is confirmed and disease pressure is significant, use a locally registered fungicide according to its product label and local agricultural guidance."

    ],

    "maize gray leaf spot": [

        "Inspect lower and middle maize leaves for expanding gray or tan lesions.",

        "Manage infected maize crop residue where practical.",

        "Maintain appropriate plant spacing and field airflow.",

        "Avoid unnecessary overhead irrigation that keeps foliage wet.",

        "Scout nearby maize plants regularly for new lesions.",

        "If Gray Leaf Spot is confirmed, follow the locally recommended disease-management program and use only a registered fungicide according to its label."

    ],

    "maize leaf blight": [

        "Inspect additional maize leaves for expanding brown or gray leaf lesions.",

        "Remove severely affected leaves where practical.",

        "Manage infected maize crop residue to reduce potential disease carryover.",

        "Improve field airflow and avoid unnecessarily dense, wet foliage.",

        "Avoid unnecessary overhead irrigation and prolonged leaf wetness.",

        "Maintain balanced irrigation and crop nutrition so the plants are not under additional stress.",

        "Monitor nearby maize plants for similar lesions.",

        "If Maize Leaf Blight is confirmed, use the locally recommended disease-management program and apply a locally registered fungicide only according to its product label."

    ],

    "maize northern leaf blight": [

        "Inspect maize leaves for long, expanding gray-green or brown lesions.",

        "Remove or manage severely infected plant material where practical.",

        "Manage infected crop residue to reduce disease carryover.",

        "Improve field airflow and reduce prolonged leaf wetness.",

        "Avoid unnecessary overhead irrigation.",

        "Monitor surrounding maize plants frequently.",

        "If Northern Leaf Blight is confirmed, follow local disease-management recommendations and use a registered fungicide according to the product label when appropriate."

    ],

    # ------------------------------------------------------------
    # POTATO
    # ------------------------------------------------------------

    "potato early blight": [

        "Inspect older potato leaves for dark circular lesions and concentric ring patterns.",

        "Remove severely affected leaves where practical.",

        "Manage infected crop residue after harvest.",

        "Avoid prolonged leaf wetness and unnecessary overhead irrigation.",

        "Maintain balanced plant nutrition and avoid crop stress.",

        "Monitor nearby potato plants for expanding lesions.",

        "If Early Blight is confirmed, follow local potato disease-management guidance and use a registered fungicide according to its label when appropriate."

    ],

    "potato late blight": [

        "Inspect the entire potato crop for rapidly expanding dark lesions.",

        "Remove severely affected plant material where practical.",

        "Avoid prolonged leaf wetness and unnecessary overhead irrigation.",

        "Increase scouting frequency during cool, wet or humid weather.",

        "Manage infected plant residue appropriately.",

        "If Late Blight is confirmed, seek prompt local agricultural guidance because the disease can develop rapidly.",

        "Use only locally registered disease-control products according to their labels."

    ],

    # ------------------------------------------------------------
    # RICE
    # ------------------------------------------------------------

    "rice blast": [

        "Inspect rice leaves, nodes and panicles for additional blast symptoms.",

        "Review nitrogen management and avoid excessive nitrogen application.",

        "Maintain appropriate water and field-management practices.",

        "Manage infected crop residue where practical.",

        "Monitor nearby rice plants for spreading symptoms.",

        "If Rice Blast is confirmed, follow local rice disease-management recommendations and use registered treatments according to their labels."

    ],

    "rice brown spot": [

        "Inspect additional rice leaves for brown circular or oval lesions.",

        "Review overall crop nutrition and soil fertility.",

        "Maintain appropriate water management.",

        "Manage infected crop residue where practical.",

        "Monitor nearby plants for additional lesions.",

        "If Brown Spot is confirmed, correct confirmed nutritional or field-management problems and follow local disease-management guidance."

    ],

    "rice tungro": [

        "Inspect nearby rice plants for yellow-orange discoloration and stunting.",

        "Check surrounding plants because tungro can occur in field patches.",

        "Use healthy planting material for future plantings.",

        "Monitor and manage relevant insect vectors according to local agricultural guidance.",

        "Confirm suspected Rice Tungro using appropriate diagnostic assessment before major treatment decisions."

    ],

    "bacterial leaf blight": [

        "Inspect nearby plants for water-soaked or yellowing leaf lesions.",

        "Reduce unnecessary overhead irrigation and prolonged leaf wetness.",

        "Maintain field sanitation and manage infected crop residues appropriately.",

        "Avoid practices that unnecessarily injure plants.",

        "Monitor disease spread after rainfall or prolonged humidity.",

        "Confirm the disease before selecting a bactericide or other disease-specific treatment.",

        "Use only locally registered products according to their labels."

    ],

    "bacterial leaf streak": [

        "Inspect nearby plants for narrow, streak-like lesions.",

        "Reduce unnecessary overhead irrigation and prolonged leaf wetness.",

        "Maintain field sanitation.",

        "Manage infected crop residues where practical.",

        "Monitor disease development after rainfall or humid conditions.",

        "Confirm the diagnosis before disease-specific treatment."

    ],

    "bacterial panicle blight": [

        "Inspect affected panicles and nearby plants for similar symptoms.",

        "Maintain appropriate field sanitation.",

        "Avoid unnecessary plant stress and prolonged wetness.",

        "Monitor the crop closely during favorable weather.",

        "Confirm the disease before applying disease-specific treatment.",

        "Follow locally approved agricultural management guidance."

    ],

    # ------------------------------------------------------------
    # APPLE
    # ------------------------------------------------------------

    "apple scab": [

        "Inspect new leaves and fruit for additional olive-brown scab lesions.",

        "Remove and manage heavily affected fallen leaves where practical.",

        "Improve canopy airflow.",

        "Reduce prolonged leaf wetness where practical.",

        "Monitor new growth during humid or wet periods.",

        "If Apple Scab is confirmed, follow local orchard disease-management guidance and use a registered fungicide according to its label."

    ],

    "apple rust": [

        "Inspect nearby apple leaves for additional rust symptoms.",

        "Remove severely affected leaves where practical.",

        "Improve canopy airflow.",

        "Monitor the crop during humid weather.",

        "Confirm the diagnosis before disease-specific treatment.",

        "If confirmed, follow locally recommended rust-management practices and registered product labels."

    ],

    "black rot": [

        "Remove severely affected plant material where practical.",

        "Manage infected fallen leaves and crop debris.",

        "Improve canopy airflow.",

        "Avoid prolonged leaf wetness.",

        "Inspect nearby plants for additional symptoms.",

        "Confirm the disease before applying fungicides."

    ],

    # ------------------------------------------------------------
    # TOMATO
    # ------------------------------------------------------------

    "tomato bacterial spot": [

        "Remove severely affected leaves where practical.",

        "Avoid unnecessary overhead irrigation.",

        "Keep foliage as dry as practical.",

        "Sanitize tools and avoid spreading contaminated plant material.",

        "Monitor nearby tomato plants for additional lesions.",

        "Confirm the disease before applying any disease-specific treatment.",

        "Use only locally registered products according to their labels."

    ],

    "tomato early blight": [

        "Remove severely affected lower leaves where practical.",

        "Manage infected crop residue.",

        "Improve airflow around the crop.",

        "Avoid prolonged leaf wetness.",

        "Avoid unnecessary overhead irrigation.",

        "Maintain balanced crop nutrition.",

        "If confirmed, follow local tomato disease-management recommendations and registered fungicide labels."

    ],

    "tomato late blight": [

        "Inspect nearby tomato plants immediately for similar lesions.",

        "Remove severely affected plant material where practical.",

        "Avoid prolonged leaf wetness.",

        "Avoid unnecessary overhead irrigation.",

        "Increase scouting during cool, humid or wet weather.",

        "Confirm the diagnosis promptly.",

        "If confirmed, follow local late-blight management recommendations and registered product labels."

    ],

    "tomato leaf mold": [

        "Improve greenhouse or field ventilation where possible.",

        "Reduce prolonged humidity around foliage.",

        "Avoid unnecessary overhead irrigation.",

        "Remove severely affected leaves where practical.",

        "Monitor nearby tomato plants.",

        "Confirm the diagnosis before applying a fungicide."

    ],

    "tomato mosaic virus": [

        "Inspect nearby tomato plants for mosaic patterns and abnormal growth.",

        "Remove severely affected plants where practical when infection is confirmed.",

        "Disinfect tools between plants.",

        "Avoid handling healthy plants immediately after infected plants.",

        "Use healthy planting material for future plantings.",

        "Manage relevant insect vectors according to local agricultural guidance.",

        "Do not assume fungicides will control a viral disease."

    ],

    "tomato yellow leaf curl virus": [

        "Inspect nearby tomato plants for yellowing, curling and stunting.",

        "Check for whitefly activity.",

        "Manage confirmed insect vectors according to local integrated pest-management guidance.",

        "Remove severely affected plants where practical when infection is confirmed.",

        "Use healthy planting material for future plantings.",

        "Do not treat a confirmed viral infection as a fungal disease."

    ],

    "tomato septoria leaf spot": [

        "Remove severely affected lower leaves where practical.",

        "Avoid overhead irrigation that keeps foliage wet.",

        "Improve airflow around plants.",

        "Manage infected crop residue.",

        "Monitor new leaves for additional lesions.",

        "Confirm the disease before applying a fungicide."

    ],

    # ------------------------------------------------------------
    # GENERAL DISEASE CLASSES
    # ------------------------------------------------------------

    "powdery mildew": [

        "Inspect leaves for characteristic powdery fungal growth.",

        "Improve canopy airflow.",

        "Reduce excessive humidity around foliage where practical.",

        "Remove severely affected leaves where practical.",

        "Avoid excessive nitrogen application unless supported by crop requirements.",

        "Monitor nearby plants.",

        "If confirmed, use a locally registered fungicide according to its product label."

    ],

    "downy mildew": [

        "Inspect the underside of leaves for additional symptoms.",

        "Reduce prolonged leaf wetness.",

        "Improve airflow through the crop.",

        "Avoid unnecessary overhead irrigation.",

        "Remove severely affected plant material where practical.",

        "Monitor surrounding plants during humid weather.",

        "If confirmed, follow local disease-management recommendations and registered product labels."

    ],

    "leaf mold": [

        "Improve ventilation and canopy airflow.",

        "Reduce prolonged humidity around foliage.",

        "Avoid unnecessary overhead irrigation.",

        "Remove severely affected leaves where practical.",

        "Monitor nearby plants.",

        "Confirm the disease before applying disease-specific treatment."

    ],

    "leaf blight": [

        "Inspect additional leaves to determine the extent of the lesions.",

        "Remove severely affected leaves where practical.",

        "Improve field airflow.",

        "Reduce prolonged leaf wetness.",

        "Avoid unnecessary overhead irrigation.",

        "Monitor nearby plants.",

        "Confirm the exact disease and crop before applying disease-specific treatment."

    ],

    "leaf scorch": [

        "Inspect irrigation and soil-moisture conditions.",

        "Check for heat, drought, salinity or other environmental stress.",

        "Maintain appropriate soil moisture without excessive irrigation.",

        "Inspect roots and soil conditions if symptoms continue.",

        "Do not assume Leaf Scorch is caused by a pathogen without further verification."

    ],

    "citrus greening": [

        "Inspect additional citrus leaves and branches for uneven yellowing and associated symptoms.",

        "Inspect the crop for signs of relevant insect vectors.",

        "Manage confirmed vectors according to local integrated pest-management guidance.",

        "Use healthy planting material for future plantings.",

        "Confirm the diagnosis with appropriate plant-health testing.",

        "Do not assume a fungicide will cure a confirmed bacterial/vascular disease."

    ],

    "esca": [

        "Inspect the plant for additional symptoms and affected woody tissue.",

        "Remove severely affected material only according to appropriate crop-management practices.",

        "Maintain plant hygiene.",

        "Avoid unnecessary injuries to woody tissues.",

        "Confirm the condition with appropriate plant-health assessment.",

        "Seek crop-specific specialist guidance for confirmed woody-tissue disease."

    ],

    "target spot": [

        "Inspect nearby leaves and fruit for expanding target-like lesions.",

        "Remove severely affected plant material where practical.",

        "Improve canopy airflow.",

        "Reduce prolonged leaf wetness.",

        "Avoid unnecessary overhead irrigation.",

        "Confirm the disease before applying a fungicide."

    ],

    "septoria leaf spot": [

        "Remove severely affected leaves where practical.",

        "Improve airflow around the crop.",

        "Avoid prolonged leaf wetness.",

        "Avoid unnecessary overhead irrigation.",

        "Manage infected crop residue.",

        "Confirm the disease before applying a fungicide."

    ]

}


# ================================================================
# NUTRIENT MANAGEMENT
# ================================================================

NUTRIENT_MANAGEMENT = {

    "potassium deficiency": {

        "banana": [

            "Inspect older banana leaves for yellowing or necrosis along the leaf margins.",

            "Check whether symptoms are strongest on older leaves.",

            "Confirm potassium deficiency using soil or plant-tissue testing.",

            "Check soil potassium availability and the current fertilizer program.",

            "Review irrigation because both excessive water stress and poor root-zone conditions can affect nutrient uptake.",

            "If potassium deficiency is confirmed, apply a suitable potassium fertilizer according to soil-test recommendations and the product label.",

            "Maintain consistent soil moisture to support nutrient uptake.",

            "Do not apply excessive potassium when soil potassium is already adequate."

        ],

        "maize": [

            "Inspect older maize leaves for marginal yellowing, chlorosis or necrosis.",

            "Confirm potassium status using soil or plant-tissue testing.",

            "Check soil potassium availability.",

            "Review the current fertilizer program and nutrient balance.",

            "Review irrigation and soil moisture because nutrient uptake depends on suitable root-zone conditions.",

            "If potassium deficiency is confirmed, apply an appropriate potassium fertilizer according to soil-test recommendations and the product label.",

            "Maintain consistent soil moisture.",

            "Avoid excessive potassium application when soil levels are already adequate."

        ],

        "default": [

            "Inspect older leaves for marginal yellowing or necrosis.",

            "Confirm potassium deficiency through soil or plant-tissue testing.",

            "Review soil potassium availability.",

            "Review the crop fertilizer program.",

            "If confirmed, correct potassium deficiency according to soil-test recommendations and the product label.",

            "Maintain appropriate soil moisture."

        ]

    },

    "nitrogen deficiency": {

        "maize": [

            "Inspect older maize leaves for characteristic nitrogen-deficiency yellowing.",

            "Confirm nitrogen status using soil or plant-tissue testing.",

            "Review the crop growth stage and current nitrogen-management program.",

            "Check whether soil moisture is adequate for nutrient uptake.",

            "If nitrogen deficiency is confirmed, apply nitrogen according to soil-test results, crop requirement and the fertilizer label.",

            "Avoid excessive nitrogen application because unnecessary nitrogen can increase environmental and crop-management risks."

        ],

        "default": [

            "Inspect older leaves for nitrogen-deficiency symptoms.",

            "Confirm nitrogen status using soil or plant-tissue testing.",

            "Review the crop growth stage and current fertilizer program.",

            "If confirmed, correct nitrogen deficiency according to soil-test recommendations and crop requirements."

        ]

    },

    "phosphorus deficiency": {

        "maize": [

            "Inspect plants for symptoms consistent with phosphorus deficiency.",

            "Confirm phosphorus status using soil or plant-tissue testing.",

            "Check soil pH because phosphorus availability is affected by soil conditions.",

            "Review the existing phosphorus fertilizer program.",

            "If deficiency is confirmed, apply phosphorus according to soil-test recommendations and the product label.",

            "Maintain suitable soil moisture to support nutrient uptake."

        ],

        "default": [

            "Confirm phosphorus status using soil or plant-tissue testing.",

            "Check soil pH and phosphorus availability.",

            "Review the existing fertilizer program.",

            "If confirmed, correct phosphorus deficiency according to soil-test recommendations."

        ]

    },

    "magnesium deficiency": {

        "banana": [

            "Inspect older banana leaves for interveinal chlorosis.",

            "Confirm magnesium status using soil or plant-tissue testing.",

            "Review the balance between magnesium, potassium and calcium.",

            "Check soil pH and root-zone conditions.",

            "If deficiency is confirmed, apply an appropriate magnesium source according to soil-test recommendations and the product label."

        ],

        "maize": [

            "Inspect older maize leaves for interveinal chlorosis.",

            "Confirm magnesium status through soil or plant-tissue testing.",

            "Review magnesium, potassium and calcium balance.",

            "Check soil pH and root-zone conditions.",

            "If confirmed, correct magnesium deficiency according to soil-test recommendations."

        ],

        "default": [

            "Inspect older leaves for interveinal chlorosis.",

            "Confirm magnesium status using soil or plant-tissue testing.",

            "Review soil pH and nutrient balance.",

            "Correct confirmed deficiency according to soil-test recommendations."

        ]

    },

    "calcium deficiency": {

        "banana": [

            "Inspect young leaves and developing tissues for symptoms associated with calcium deficiency.",

            "Confirm calcium status using soil and plant-tissue testing.",

            "Review soil pH and root-zone conditions.",

            "Check whether irrigation or root-zone problems are affecting nutrient uptake.",

            "If deficiency is confirmed, correct calcium availability according to soil-test recommendations."

        ],

        "default": [

            "Inspect young leaves and developing tissues.",

            "Confirm calcium status using soil or plant-tissue testing.",

            "Review soil pH and root-zone conditions.",

            "Correct confirmed deficiency according to soil-test recommendations."

        ]

    },

    "boron deficiency": {

        "banana": [

            "Inspect young leaves and growing tissues for symptoms associated with boron deficiency.",

            "Confirm boron status using soil or plant-tissue testing.",

            "Review soil pH and boron availability.",

            "If deficiency is confirmed, apply boron only according to soil-test recommendations and the product label.",

            "Avoid excessive boron application because the safe range between deficiency and toxicity can be narrow."

        ],

        "default": [

            "Confirm boron status using soil or plant-tissue testing.",

            "Review soil pH and boron availability.",

            "Correct confirmed boron deficiency according to soil-test recommendations.",

            "Avoid excessive boron application."

        ]

    },

    "iron deficiency": {

        "banana": [

            "Inspect young banana leaves for interveinal chlorosis.",

            "Check soil pH because iron availability is strongly affected by soil conditions.",

            "Confirm iron status using soil or plant-tissue analysis.",

            "Review root-zone conditions.",

            "Correct confirmed iron deficiency according to soil-test recommendations."

        ],

        "default": [

            "Inspect young leaves for interveinal chlorosis.",

            "Check soil pH.",

            "Confirm iron status using soil or plant-tissue analysis.",

            "Correct confirmed deficiency according to soil-test recommendations."

        ]

    },

    "manganese deficiency": {

        "banana": [

            "Inspect leaves for patterns associated with manganese deficiency.",

            "Confirm manganese status using soil or plant-tissue testing.",

            "Review soil pH and nutrient availability.",

            "Correct confirmed deficiency according to soil-test recommendations."

        ],

        "default": [

            "Inspect leaves for manganese-deficiency patterns.",

            "Confirm manganese status using soil or plant-tissue testing.",

            "Review soil pH and nutrient balance.",

            "Correct confirmed deficiency according to soil-test recommendations."

        ]

    },

    "sulphur deficiency": {

        "banana": [

            "Inspect younger banana leaves for uniform chlorosis.",

            "Confirm sulphur status using soil or plant-tissue testing.",

            "Review soil organic matter and current fertilizer management.",

            "Correct confirmed sulphur deficiency according to soil-test recommendations."

        ],

        "default": [

            "Inspect younger leaves for uniform chlorosis.",

            "Confirm sulphur status using soil or plant-tissue testing.",

            "Review soil organic matter and fertilizer management.",

            "Correct confirmed deficiency according to soil-test recommendations."

        ]

    },

    "zinc deficiency": {

        "banana": [

            "Inspect young leaves for symptoms associated with zinc deficiency.",

            "Confirm zinc status using soil or plant-tissue testing.",

            "Review soil pH and zinc availability.",

            "Correct confirmed zinc deficiency according to soil-test recommendations."

        ],

        "default": [

            "Inspect young leaves for zinc-deficiency symptoms.",

            "Confirm zinc status using soil or plant-tissue testing.",

            "Review soil pH and zinc availability.",

            "Correct confirmed deficiency according to soil-test recommendations."

        ]

    }

}


# ================================================================
# NUTRIENT ALIASES
# ================================================================

NUTRIENT_ALIASES = {

    "potassium": "potassium deficiency",

    "potassium deficiency":
        "potassium deficiency",

    "nitrogen": "nitrogen deficiency",

    "nitrogen deficiency":
        "nitrogen deficiency",

    "phosphorus": "phosphorus deficiency",

    "phosphorus deficiency":
        "phosphorus deficiency",

    "magnesium": "magnesium deficiency",

    "magnesium deficiency":
        "magnesium deficiency",

    "calcium": "calcium deficiency",

    "calcium deficiency":
        "calcium deficiency",

    "boron": "boron deficiency",

    "boron deficiency":
        "boron deficiency",

    "iron": "iron deficiency",

    "iron deficiency":
        "iron deficiency",

    "manganese": "manganese deficiency",

    "manganese deficiency":
        "manganese deficiency",

    "sulphur": "sulphur deficiency",

    "sulfur": "sulphur deficiency",

    "sulphur deficiency":
        "sulphur deficiency",

    "sulfur deficiency":
        "sulphur deficiency",

    "zinc": "zinc deficiency",

    "zinc deficiency":
        "zinc deficiency"

}


# ================================================================
# ENGINE
# ================================================================

class RecommendationEngine:

    def __init__(self):

        self.name = (
            "AI Crop Health "
            "Crop-Specific Recommendation Engine"
        )

        self.high_confidence = 0.80

        self.medium_confidence = 0.60


    # ============================================================
    # CONTEXT
    # ============================================================

    def process_context(self, context):

        if not isinstance(context, dict):

            context = {}

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

            "visible_symptoms":
                context.get(
                    "visible_symptoms",
                    []
                ),

            "location":
                context.get(
                    "location"
                )

        }


    # ============================================================
    # DISEASE-CROP VALIDATION
    # ============================================================

    def disease_matches_crop(
        self,
        crop,
        disease
    ):

        crop = normalize_crop(
            crop
        )

        disease_key = str(
            disease or ""
        ).strip().lower()

        if not disease_key:

            return False

        if disease_key == "healthy":

            return True

        if crop not in CROP_DISEASES:

            return None

        allowed = CROP_DISEASES[
            crop
        ]

        return disease_key in allowed


    # ============================================================
    # DISEASE MANAGEMENT
    # ============================================================

    def get_disease_management(
        self,
        crop,
        disease
    ):

        if not disease:

            return []

        disease_key = str(
            disease
        ).strip().lower()

        if disease_key in (
            "",
            "healthy",
            "uncertain"
        ):

            return []

        crop = normalize_crop(
            crop
        )

        match = self.disease_matches_crop(
            crop,
            disease
        )

        # --------------------------------------------------------
        # Crop mismatch
        # --------------------------------------------------------

        if match is False:

            return [

                (
                    f"IMPORTANT: The predicted condition "
                    f"'{disease}' does not match the selected "
                    f"crop '{display_crop(crop)}'."
                ),

                (
                    "Do not apply disease-specific treatment "
                    "from this prediction yet."
                ),

                (
                    "Capture another clear image of the affected "
                    "crop and verify the crop/disease combination "
                    "with an agricultural expert or reliable "
                    "diagnostic method."
                )

            ]

        # --------------------------------------------------------
        # Exact disease knowledge
        # --------------------------------------------------------

        if disease_key in DISEASE_MANAGEMENT:

            return DISEASE_MANAGEMENT[
                disease_key
            ].copy()

        # --------------------------------------------------------
        # Flexible disease matching
        # --------------------------------------------------------

        for key, actions in DISEASE_MANAGEMENT.items():

            if (
                key in disease_key
                or disease_key in key
            ):

                return actions.copy()

        # --------------------------------------------------------
        # Safe fallback
        # --------------------------------------------------------

        return [

            (
                f"Inspect the {display_crop(crop)} crop carefully "
                f"for symptoms associated with {disease}."
            ),

            "Remove severely affected plant material where practical.",

            "Improve field sanitation and reduce prolonged leaf wetness.",

            "Monitor nearby plants for similar symptoms.",

            (
                "Confirm the exact disease before applying "
                "disease-specific treatment."
            )

        ]


    # ============================================================
    # NUTRIENT MANAGEMENT
    # ============================================================

    def get_nutrient_management(
        self,
        crop,
        nutrient
    ):

        if not nutrient:

            return []

        nutrient_key = str(
            nutrient
        ).strip().lower()

        if nutrient_key in (
            "",
            "healthy",
            "uncertain"
        ):

            return []

        canonical = NUTRIENT_ALIASES.get(
            nutrient_key
        )

        if canonical is None:

            for alias, value in NUTRIENT_ALIASES.items():

                if (
                    alias in nutrient_key
                    or nutrient_key in alias
                ):

                    canonical = value

                    break

        if canonical is None:

            return [

                (
                    f"Confirm the suspected {nutrient} "
                    "through soil or plant-tissue testing."
                ),

                "Review the current crop nutrition program.",

                (
                    "Correct a confirmed deficiency according "
                    "to soil-test results and crop requirements."
                ),

                "Avoid excessive fertilizer application."

            ]

        management = NUTRIENT_MANAGEMENT.get(
            canonical,
            {}
        )

        if not isinstance(
            management,
            dict
        ):

            return list(
                management
            )

        crop = normalize_crop(
            crop
        )

        if crop in management:

            return management[
                crop
            ].copy()

        return management.get(
            "default",
            []
        ).copy()


    # ============================================================
    # ENVIRONMENTAL RECOMMENDATIONS
    # ============================================================

    def get_environmental_recommendations(
        self,
        crop,
        context,
        disease=None,
        nutrient=None
    ):
        """
        Generate CONTEXT-CONDITIONED recommendations.

        Important:
        Context must change the recommended action, not merely
        be displayed to the user.

        Inputs considered:
            - crop
            - disease
            - nutrient
            - temperature
            - humidity
            - rainfall
            - weather condition
            - wind speed
            - soil condition
            - soil pH
            - irrigation
            - growth stage
        """

        recommendations = []

        # ============================================================
        # NORMALIZE INPUT
        # ============================================================

        crop_name = str(
            crop or "Unknown"
        ).strip()

        crop_lower = crop_name.lower()

        context = (
            context
            if isinstance(context, dict)
            else {}
        )

        disease_name = ""

        if isinstance(disease, dict):
            disease_name = str(
                disease.get(
                    "name",
                    ""
                )
            ).strip()

        disease_lower = disease_name.lower()

        nutrient_name = ""

        if isinstance(nutrient, dict):
            nutrient_name = str(
                nutrient.get(
                    "name",
                    ""
                )
            ).strip()

        nutrient_lower = nutrient_name.lower()

        # ============================================================
        # CONTEXT VALUES
        # ============================================================

        temperature = context.get(
            "temperature"
        )

        humidity = context.get(
            "humidity"
        )

        rainfall = str(
            context.get(
                "recent_rainfall",
                ""
            )
        ).strip().lower()

        weather_condition = str(
            context.get(
                "weather_condition",
                ""
            )
        ).strip().lower()

        soil_condition = str(
            context.get(
                "soil_condition",
                ""
            )
        ).strip().lower()

        irrigation = str(
            context.get(
                "irrigation",
                ""
            )
        ).strip().lower()

        growth_stage = str(
            context.get(
                "growth_stage",
                ""
            )
        ).strip().lower()

        soil_ph = context.get(
            "soil_ph"
        )

        wind_speed = context.get(
            "wind_speed"
        )

        # ============================================================
        # SAFE NUMERIC CONVERSION
        # ============================================================

        try:
            temperature = (
                float(temperature)
                if temperature is not None
                else None
            )
        except (
            TypeError,
            ValueError
        ):
            temperature = None

        try:
            humidity = (
                float(humidity)
                if humidity is not None
                else None
            )
        except (
            TypeError,
            ValueError
        ):
            humidity = None

        try:
            soil_ph = (
                float(soil_ph)
                if soil_ph is not None
                and str(soil_ph).strip() != ""
                else None
            )
        except (
            TypeError,
            ValueError
        ):
            soil_ph = None

        try:
            wind_speed = (
                float(wind_speed)
                if wind_speed is not None
                else None
            )
        except (
            TypeError,
            ValueError
        ):
            wind_speed = None

        # ============================================================
        # DETECT ENVIRONMENTAL CONDITIONS
        # ============================================================

        high_humidity = (
            humidity is not None
            and humidity >= 80
        )

        very_high_humidity = (
            humidity is not None
            and humidity >= 90
        )

        hot_condition = (
            temperature is not None
            and temperature >= 30
        )

        cool_condition = (
            temperature is not None
            and temperature < 20
        )

        wet_soil = soil_condition in {
            "wet",
            "very wet",
            "waterlogged",
            "saturated"
        }

        moist_soil = soil_condition in {
            "moist",
            "moderately moist"
        }

        dry_soil = soil_condition in {
            "dry",
            "very dry"
        }

        heavy_rain = rainfall in {
            "heavy",
            "high",
            "frequent",
            "very high"
        }

        recent_rain = rainfall in {
            "moderate",
            "heavy",
            "high",
            "frequent",
            "very high"
        }

        storm_condition = (
            "thunderstorm" in weather_condition
            or "storm" in weather_condition
        )

        rainy_weather = (
            "rain" in weather_condition
            or "drizzle" in weather_condition
            or "shower" in weather_condition
            or storm_condition
        )

        strong_wind = (
            wind_speed is not None
            and wind_speed >= 25
        )

        # ============================================================
        # 1. HUMIDITY + DISEASE INTERACTION
        # ============================================================

        disease_present = (
            disease_lower not in {
                "",
                "uncertain",
                "unknown",
                "healthy",
                "not available",
                "none"
            }
        )

        if high_humidity and disease_present:

            if very_high_humidity:

                recommendations.append(
                    f"CONTEXT CONDITION — {crop_name}: "
                    f"relative humidity is very high "
                    f"({humidity:.0f}%). Because a possible "
                    f"{disease_name} condition is present, "
                    f"increase inspection frequency and check "
                    f"nearby leaves for new lesions or disease spread."
                )

            else:

                recommendations.append(
                    f"CONTEXT CONDITION — {crop_name}: "
                    f"humidity is high ({humidity:.0f}%). "
                    f"Because a possible {disease_name} condition "
                    f"is present, monitor foliage more frequently "
                    f"for disease development."
                )

            recommendations.append(
                "Because foliage disease risk can increase when "
                "leaves remain wet for long periods, avoid "
                "unnecessary overhead irrigation and improve "
                "air movement around the crop."
            )

        elif very_high_humidity:

            recommendations.append(
                f"CONTEXT CONDITION — {crop_name}: "
                f"humidity is very high ({humidity:.0f}%). "
                f"Monitor leaves closely for new fungal or "
                f"bacterial symptoms even though no specific "
                f"disease has been confirmed."
            )

        # ============================================================
        # 2. WEATHER + RAINFALL INTERACTION
        # ============================================================

        if storm_condition:

            recommendations.append(
                f"WEATHER ACTION — {crop_name}: current weather "
                f"is {context.get('weather_condition')}. "
                f"Do not add irrigation while rainfall is occurring "
                f"or immediately after a wet period unless the "
                f"root zone actually requires water."
            )

            recommendations.append(
                "After the storm, inspect leaves for prolonged "
                "wetness, physical damage and newly developing "
                "lesions before applying disease-specific treatment."
            )

        elif heavy_rain:

            recommendations.append(
                f"RAINFALL ACTION — {crop_name}: recent rainfall "
                f"is reported as {rainfall}. Check drainage and "
                f"avoid unnecessary irrigation until the root zone "
                f"has returned to an appropriate moisture level."
            )

        elif recent_rain:

            recommendations.append(
                f"RAINFALL ACTION — {crop_name}: recent rainfall "
                f"is elevated ({rainfall}). Recheck affected leaves "
                f"after wet periods and monitor for disease spread."
            )

        # ============================================================
        # 3. SOIL MOISTURE + IRRIGATION INTERACTION
        # ============================================================

        if wet_soil:

            recommendations.append(
                f"SOIL/WATER ACTION — {crop_name}: soil is reported "
                f"as {soil_condition}. Do not increase irrigation "
                f"while the root zone remains excessively wet. "
                f"Check drainage before adding more water."
            )

        elif moist_soil:

            if irrigation in {
                "normal",
                "regular",
                "adequate"
            }:

                recommendations.append(
                    f"SOIL/WATER ACTION — {crop_name}: soil is moist "
                    f"and irrigation is reported as {irrigation}. "
                    f"Maintain the current irrigation pattern rather "
                    f"than increasing water solely because a nutrient "
                    f"deficiency was detected."
                )

            else:

                recommendations.append(
                    f"SOIL/WATER ACTION — {crop_name}: soil is already "
                    f"moist. Avoid unnecessary additional irrigation "
                    f"and continue monitoring root-zone moisture."
                )

        elif dry_soil:

            recommendations.append(
                f"SOIL/WATER ACTION — {crop_name}: soil is reported "
                f"as dry. Check root-zone moisture before treatment "
                f"and restore appropriate irrigation if the crop is "
                f"experiencing water stress."
            )

        # ============================================================
        # 4. POTASSIUM DEFICIENCY + CONTEXT
        # ============================================================

        potassium_deficiency = (
            "potassium deficiency"
            in nutrient_lower
        )

        if potassium_deficiency:

            # Moist/wet soil
            if wet_soil:

                recommendations.append(
                    f"NUTRIENT + SOIL ACTION — {crop_name}: "
                    f"potassium deficiency is suspected, but the soil "
                    f"is currently {soil_condition}. Do not increase "
                    f"irrigation to correct the nutrient problem. "
                    f"Check drainage and confirm potassium status "
                    f"through soil or plant-tissue testing."
                )

            elif moist_soil:

                recommendations.append(
                    f"NUTRIENT + SOIL ACTION — {crop_name}: "
                    f"potassium deficiency is suspected and soil is "
                    f"currently moist. Maintain suitable soil moisture "
                    f"rather than increasing irrigation."
                )

            elif dry_soil:

                recommendations.append(
                    f"NUTRIENT + SOIL ACTION — {crop_name}: "
                    f"potassium deficiency is suspected while soil is "
                    f"dry. Address verified water stress and confirm "
                    f"potassium status before applying fertilizer."
                )

            # pH interaction
            if soil_ph is not None:

                if soil_ph < 5.5:

                    recommendations.append(
                        f"NUTRIENT + pH ACTION — {crop_name}: "
                        f"reported soil pH is {soil_ph:.2f}, which is "
                        f"strongly acidic. Confirm the pH with a soil "
                        f"test because unsuitable soil pH can affect "
                        f"nutrient availability. Do not apply a large "
                        f"fertilizer correction based only on the image."
                    )

                elif soil_ph > 7.5:

                    recommendations.append(
                        f"NUTRIENT + pH ACTION — {crop_name}: "
                        f"reported soil pH is {soil_ph:.2f}. Confirm "
                        f"soil pH with a soil test and review nutrient "
                        f"availability before changing the fertilizer "
                        f"program."
                    )

                else:

                    recommendations.append(
                        f"NUTRIENT + pH ACTION — {crop_name}: "
                        f"reported soil pH is {soil_ph:.2f}. Keep the "
                        f"current pH information in the nutrient "
                        f"assessment and confirm potassium status "
                        f"before fertilizer application."
                    )

            # High humidity interaction
            if high_humidity:

                recommendations.append(
                    f"NUTRIENT + WEATHER ACTION — {crop_name}: "
                    f"humidity is {humidity:.0f}% while potassium "
                    f"deficiency is suspected. Avoid unnecessary "
                    f"overhead irrigation because additional leaf "
                    f"wetness is undesirable under these conditions."
                )

            # Hot weather interaction
            if hot_condition:

                recommendations.append(
                    f"NUTRIENT + TEMPERATURE ACTION — {crop_name}: "
                    f"temperature is {temperature:.1f}°C. Monitor "
                    f"soil moisture and plant water stress because "
                    f"heat can increase crop water demand."
                )

        # ============================================================
        # 5. GROWTH STAGE
        # ============================================================

        if growth_stage in {
            "vegetative",
            "vegetative stage",
            "growth",
            "growing"
        }:

            if nutrient_lower not in {
                "",
                "uncertain",
                "unknown"
            }:

                recommendations.append(
                    f"GROWTH-STAGE ACTION — {crop_name}: the crop is "
                    f"in the vegetative stage. Continue monitoring "
                    f"new and older leaves so nutrient-related symptoms "
                    f"can be tracked as the plant develops."
                )

            elif disease_present:

                recommendations.append(
                    f"GROWTH-STAGE ACTION — {crop_name}: the crop is "
                    f"in the vegetative stage. Inspect newly developing "
                    f"leaves regularly for progression of the suspected "
                    f"{disease_name} condition."
                )

        elif growth_stage in {
            "flowering",
            "flowering stage"
        }:

            recommendations.append(
                f"GROWTH-STAGE ACTION — {crop_name}: the crop is "
                f"flowering. Avoid unnecessary environmental stress "
                f"and continue monitoring the crop closely during "
                f"this sensitive growth stage."
            )

        elif growth_stage in {
            "fruiting",
            "fruit development",
            "fruiting stage"
        }:

            recommendations.append(
                f"GROWTH-STAGE ACTION — {crop_name}: the crop is "
                f"in the fruit-development stage. Maintain stable "
                f"water and nutrient management and monitor symptoms "
                f"for progression."
            )

        # ============================================================
        # 6. WIND
        # ============================================================

        if strong_wind:

            recommendations.append(
                f"WEATHER ACTION — {crop_name}: wind speed is "
                f"approximately {wind_speed:.1f} km/h. Inspect plants "
                f"after the windy period for physical leaf damage and "
                f"avoid unnecessary foliar applications during strong "
                f"wind conditions."
            )

        # ============================================================
        # 7. COMBINED HIGH-RISK CONDITION
        #
        # This is the important context-aware part:
        # multiple context variables interact.
        # ============================================================

        if (
            high_humidity
            and (
                wet_soil
                or moist_soil
            )
            and (
                rainy_weather
                or recent_rain
                or storm_condition
            )
        ):

            recommendations.append(
                f"COMBINED CONTEXT ACTION — {crop_name}: high "
                f"humidity + moist/wet soil + recent wet weather "
                f"indicate a wet growing environment. Prioritize "
                f"drainage, avoid unnecessary irrigation, reduce "
                f"prolonged leaf wetness and increase crop scouting."
            )

        # ============================================================
        # 8. TEMPERATURE INFORMATION SHOULD AFFECT ACTION
        # ============================================================

        if cool_condition:

            recommendations.append(
                f"TEMPERATURE ACTION — {crop_name}: current "
                f"temperature is {temperature:.1f}°C. Avoid making "
                f"major irrigation or fertilizer changes solely from "
                f"the image prediction; monitor plant response under "
                f"the current cooler conditions."
            )

        elif hot_condition:

            recommendations.append(
                f"TEMPERATURE ACTION — {crop_name}: current "
                f"temperature is {temperature:.1f}°C. Monitor plant "
                f"water status and avoid allowing verified water stress "
                f"to persist."
            )

        # ============================================================
        # 9. FALLBACK
        # ============================================================

        if not recommendations:

            recommendations.append(
                f"Context information for {crop_name} is available, "
                f"but no additional context-specific action was "
                f"triggered. Continue monitoring the crop and use "
                f"the disease/nutrient-specific recommendations."
            )

        return recommendations




    # ============================================================
    # MAIN RECOMMENDATION GENERATOR
    # ============================================================

    def generate_recommendations(
        self,
        prediction,
        context=None
    ):

        if not isinstance(
            prediction,
            dict
        ):

            prediction = {}

        context = self.process_context(
            context
        )

        crop = normalize_crop(
            prediction.get(
                "crop",
                ""
            )
        )

        disease = prediction.get(
            "disease"
        )

        nutrient = prediction.get(
            "nutrient"
        )

        if not isinstance(
            disease,
            dict
        ):

            disease = {}

        if not isinstance(
            nutrient,
            dict
        ):

            nutrient = {}

        disease_name = str(
            disease.get(
                "name",
                disease.get(
                    "prediction",
                    ""
                )
            )
        ).strip()

        nutrient_name = str(
            nutrient.get(
                "name",
                nutrient.get(
                    "prediction",
                    ""
                )
            )
        ).strip()

        disease_confidence = normalize_confidence(

            disease.get(
                "confidence",
                disease.get(
                    "confidence_percent"
                )
            )

        )

        nutrient_confidence = normalize_confidence(

            nutrient.get(
                "confidence",
                nutrient.get(
                    "confidence_percent"
                )
            )

        )

        recommendations = []

        # ========================================================
        # SECTION 1 — CROP
        # ========================================================

        if crop:

            recommendations.append(

                (
                    f"CROP-SPECIFIC ACTION PLAN — "
                    f"{display_crop(crop)}"
                )

            )

        # ========================================================
        # SECTION 2 — DISEASE
        # ========================================================

        if disease_name:

            if disease_name.lower() == "healthy":

                recommendations.append(

                    "Disease assessment: No specific disease "
                    "pattern was identified. Continue routine "
                    "crop monitoring."

                )

            elif disease_name.lower() == "uncertain":

                recommendations.append(

                    (
                        "Disease assessment: The disease prediction "
                        "is uncertain. Capture another clear image "
                        "showing the affected area before applying "
                        "disease-specific treatment."
                    )

                )

            else:

                recommendations.append(

                    (
                        f"DISEASE SOLUTION — "
                        f"{disease_name}"
                    )

                )

                disease_actions = self.get_disease_management(

                    crop,

                    disease_name

                )

                recommendations.extend(
                    disease_actions
                )

                # Confidence safety

                if (
                    disease_confidence is not None
                    and disease_confidence < 0.60
                ):

                    recommendations.append(

                        (
                            "Disease confidence is low. Do not "
                            "start disease-specific treatment from "
                            "the image prediction alone."
                        )

                    )

                elif (
                    disease_confidence is not None
                    and disease_confidence < 0.80
                ):

                    recommendations.append(

                        (
                            "Disease confidence is moderate. "
                            "Verify the condition before applying "
                            "disease-specific treatment."
                        )

                    )

                else:

                    recommendations.append(

                        (
                            "Disease confidence is high, but the "
                            "result is still an AI screening result. "
                            "Confirm the condition before treatment."
                        )

                    )

        # ========================================================
        # SECTION 3 — NUTRIENT
        # ========================================================

        if nutrient_name:

            if nutrient_name.lower() == "healthy":

                recommendations.append(

                    (
                        "NUTRIENT SOLUTION — "
                        "No specific nutrient deficiency detected."
                    )

                )

                recommendations.append(

                    (
                        "Continue balanced crop nutrition and "
                        "monitor the crop."
                    )

                )

            elif nutrient_name.lower() == "uncertain":

                recommendations.append(

                    (
                        "NUTRIENT SOLUTION — Uncertain prediction. "
                        "Confirm nutrient status using soil or "
                        "plant-tissue testing before applying "
                        "fertilizer."
                    )

                )

            else:

                recommendations.append(

                    (
                        f"NUTRIENT SOLUTION — "
                        f"{nutrient_name}"
                    )

                )

                nutrient_actions = (
                    self.get_nutrient_management(
                        crop,
                        nutrient_name
                    )
                )

                recommendations.extend(
                    nutrient_actions
                )

                if (
                    nutrient_confidence is not None
                    and nutrient_confidence < 0.60
                ):

                    recommendations.append(

                        (
                            "Nutrient prediction confidence is "
                            "low. Confirm the deficiency through "
                            "soil or plant-tissue testing before "
                            "applying fertilizer."
                        )

                    )

                elif (
                    nutrient_confidence is not None
                    and nutrient_confidence < 0.80
                ):

                    recommendations.append(

                        (
                            "Nutrient prediction confidence is "
                            "moderate. Confirm the deficiency "
                            "before making major fertilizer changes."
                        )

                    )

        # ========================================================
        # SECTION 4 — ENVIRONMENT
        # ========================================================

        environmental = (
            self.get_environmental_recommendations(
                crop,
                context,
                disease,
                nutrient
            )
        )

        if environmental:

            recommendations.append(
                "ENVIRONMENT-SPECIFIC ACTIONS"
            )

            recommendations.extend(
                environmental
            )

        # ========================================================
        # SECTION 5 — GENERAL SAFETY
        # ========================================================

        recommendations.append(

            (
                "IMPORTANT: This is AI-assisted crop-health "
                "screening and decision support, not a confirmed "
                "agricultural diagnosis."
            )

        )

        # Remove duplicate recommendations

        cleaned = []

        for item in recommendations:

            item = str(
                item
            ).strip()

            if not item:

                continue

            if item not in cleaned:

                cleaned.append(
                    item
                )

        return cleaned


    # ============================================================
    # COMPATIBILITY METHODS
    # ============================================================

    def generate(
        self,
        crop,
        disease_prediction=None,
        disease_confidence=None,
        nutrient_prediction=None,
        nutrient_confidence=None,
        context=None
    ):

        prediction = {

            "crop":
                crop,

            "disease": {

                "name":
                    disease_prediction,

                "confidence":
                    disease_confidence

            }
            if disease_prediction
            else {},

            "nutrient": {

                "name":
                    nutrient_prediction,

                "confidence":
                    nutrient_confidence

            }
            if nutrient_prediction
            else {}

        }

        return self.generate_recommendations(
            prediction,
            context
        )


    def recommend(
        self,
        crop,
        disease_prediction=None,
        disease_confidence=None,
        nutrient_prediction=None,
        nutrient_confidence=None,
        context=None
    ):

        return self.generate(
            crop,
            disease_prediction,
            disease_confidence,
            nutrient_prediction,
            nutrient_confidence,
            context
        )


    def get_recommendation(
        self,
        crop,
        disease_prediction=None,
        disease_confidence=None,
        nutrient_prediction=None,
        nutrient_confidence=None,
        context=None
    ):

        return self.generate(
            crop,
            disease_prediction,
            disease_confidence,
            nutrient_prediction,
            nutrient_confidence,
            context
        )


    # ============================================================
    # PRINT RESULT
    # ============================================================

    def print_recommendations(
        self,
        result
    ):

        print()
        print("=" * 70)
        print(
            "AI CROP HEALTH "
            "CROP-SPECIFIC RECOMMENDATION ENGINE"
        )
        print("=" * 70)

        print()

        if isinstance(
            result,
            dict
        ):

            print(
                "Crop:",
                result.get(
                    "crop",
                    "Unknown"
                )
            )

            disease = result.get(
                "disease",
                {}
            )

            nutrient = result.get(
                "nutrient",
                {}
            )

            if disease:

                print()
                print(
                    "-" * 70
                )
                print(
                    "DISEASE"
                )
                print(
                    "-" * 70
                )

                print(
                    "Prediction:",
                    disease.get(
                        "name"
                    )
                )

                print(
                    "Confidence:",
                    disease.get(
                        "confidence_percent",
                        disease.get(
                            "confidence"
                        )
                    )
                )

            if nutrient:

                print()
                print(
                    "-" * 70
                )
                print(
                    "NUTRIENT"
                )
                print(
                    "-" * 70
                )

                print(
                    "Prediction:",
                    nutrient.get(
                        "name"
                    )
                )

                print(
                    "Confidence:",
                    nutrient.get(
                        "confidence_percent",
                        nutrient.get(
                            "confidence"
                        )
                    )
                )

            print()
            print(
                "-" * 70
            )
            print(
                "ACTIONABLE SOLUTIONS"
            )
            print(
                "-" * 70
            )

            recommendations = result.get(
                "recommendations",
                []
            )

            for index, recommendation in enumerate(
                recommendations,
                start=1
            ):

                print(
                    f"{index}. {recommendation}"
                )

            print()

            print(
                "Total recommendations:",
                len(
                    recommendations
                )
            )

            print(
                "=" * 70
            )


# ================================================================
# LEGACY FUNCTION
# ================================================================

def generate_recommendation(
    crop: str,
    disease_prediction: Optional[str] = None,
    disease_confidence: Optional[float] = None,
    nutrient_prediction: Optional[str] = None,
    nutrient_confidence: Optional[float] = None,
    context: Optional[Dict] = None
):

    engine = RecommendationEngine()

    prediction = {

        "crop":
            crop,

        "disease": {

            "name":
                disease_prediction,

            "confidence":
                disease_confidence,

            "confidence_percent":
                (
                    normalize_confidence(
                        disease_confidence
                    ) * 100
                    if normalize_confidence(
                        disease_confidence
                    ) is not None
                    else None
                )

        }
        if disease_prediction
        else {},

        "nutrient": {

            "name":
                nutrient_prediction,

            "confidence":
                nutrient_confidence,

            "confidence_percent":
                (
                    normalize_confidence(
                        nutrient_confidence
                    ) * 100
                    if normalize_confidence(
                        nutrient_confidence
                    ) is not None
                    else None
                )

        }
        if nutrient_prediction
        else {}

    }

    recommendations = (
        engine.generate_recommendations(
            prediction,
            context
        )
    )

    return {

        "crop":
            crop,

        "disease":
            prediction.get(
                "disease"
            ),

        "nutrient":
            prediction.get(
                "nutrient"
            ),

        "recommendations":
            recommendations

    }


# ================================================================
# DIRECT TEST
# ================================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "TESTING CROP-SPECIFIC "
        "RECOMMENDATION ENGINE"
    )

    print("=" * 70)

    engine = RecommendationEngine()

    # ------------------------------------------------------------
    # TEST 1
    # ------------------------------------------------------------

    test_prediction = {

        "crop":
            "Maize",

        "disease": {

            "name":
                "Maize Leaf Blight",

            "confidence":
                0.73,

            "confidence_percent":
                73.0

        },

        "nutrient": {

            "name":
                "Potassium Deficiency",

            "confidence":
                0.937,

            "confidence_percent":
                93.7

        }

    }

    test_context = {

        "growth_stage":
            "Vegetative",

        "soil_condition":
            "Moist",

        "soil_ph":
            6.5,

        "recent_rainfall":
            "low",

        "irrigation":
            "normal",

        "temperature":
            27,

        "humidity":
            88,

        "weather_condition":
            "Overcast",

        "wind_speed":
            9.1,

        "visible_symptoms":
            [],

        "location":
            "Example region"

    }

    result = engine.generate_recommendations(

        test_prediction,

        test_context

    )

    print()

    print(
        "CROP:",
        test_prediction["crop"]
    )

    print()

    print(
        "DISEASE:",
        test_prediction["disease"]["name"]
    )

    print(
        "NUTRIENT:",
        test_prediction["nutrient"]["name"]
    )

    print()

    print(
        "-" * 70
    )

    print(
        "CROP-SPECIFIC ACTIONABLE SOLUTIONS"
    )

    print(
        "-" * 70
    )

    for index, recommendation in enumerate(

        result,

        start=1

    ):

        print(
            f"{index}. {recommendation}"
        )

    print()

    print(
        "Total recommendations:",
        len(result)
    )

    print()

    # ------------------------------------------------------------
    # TEST 2 - MISMATCH CHECK
    # ------------------------------------------------------------

    print("=" * 70)

    print(
        "TESTING CROP / DISEASE MISMATCH"
    )

    print("=" * 70)

    mismatch_prediction = {

        "crop":
            "Banana",

        "disease": {

            "name":
                "Maize Leaf Blight",

            "confidence":
                0.73

        },

        "nutrient": {

            "name":
                "Potassium Deficiency",

            "confidence":
                0.937

        }

    }

    mismatch_result = (
        engine.generate_recommendations(
            mismatch_prediction,
            test_context
        )
    )

    print()

    for index, recommendation in enumerate(

        mismatch_result,

        start=1

    ):

        print(
            f"{index}. {recommendation}"
        )

    print()

    print("=" * 70)

    print(
        "RECOMMENDATION ENGINE TEST COMPLETED"
    )

    print("=" * 70)