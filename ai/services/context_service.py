"""
Context Service
============================================================

Combines:
    - Real-time weather information
    - Crop information
    - Soil information
    - Irrigation information
    - Growth stage
    - Visible symptoms

The weather is automatically obtained using
latitude and longitude.

This service does NOT diagnose disease.
It provides environmental/context information
to the recommendation layer.
"""

from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ai.services.weather_service import WeatherService


# ============================================================
# CONTEXT SERVICE
# ============================================================

class ContextService:

    def __init__(self):

        print("=" * 70)
        print("CONTEXT SERVICE")
        print("=" * 70)

        self.weather_service = WeatherService()

        print("Real-time weather integration enabled.")
        print("=" * 70)


    # ========================================================
    # GET COMPLETE CONTEXT
    # ========================================================

    def get_context(
        self,
        latitude,
        longitude,
        growth_stage=None,
        soil_condition=None,
        soil_ph=None,
        irrigation=None,
        visible_symptoms=None,
        location=None
    ):

        # ----------------------------------------------------
        # GET WEATHER
        # ----------------------------------------------------

        weather_result = self.weather_service.get_weather(
            latitude,
            longitude
        )


        # ----------------------------------------------------
        # CHECK WEATHER RESULT
        # ----------------------------------------------------

        if not weather_result.get("success", False):

            return {
                "success": False,

                "error": weather_result.get(
                    "error",
                    "Unable to retrieve weather."
                ),

                "context": {

                    "temperature": None,
                    "humidity": None,
                    "recent_rainfall": None,
                    "weather_condition": None,
                    "wind_speed": None,

                    "growth_stage": growth_stage,
                    "soil_condition": soil_condition,
                    "soil_ph": soil_ph,
                    "irrigation": irrigation,
                    "location": location,

                    "visible_symptoms": (
                        visible_symptoms
                        if visible_symptoms
                        else []
                    )
                }
            }


        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------
        # weather_service.py stores the environmental
        # information inside:
        #
        # weather_result["context"]
        #
        # NOT directly inside weather_result.
        # ----------------------------------------------------

        weather_context = weather_result.get(
            "context",
            {}
        )


        # ----------------------------------------------------
        # BUILD COMPLETE CONTEXT
        # ----------------------------------------------------

        context = {

            # Real-time weather
            "temperature":
                weather_context.get("temperature"),

            "humidity":
                weather_context.get("humidity"),

            "recent_rainfall":
                weather_context.get("recent_rainfall"),

            "weather_condition":
                weather_context.get("weather_condition"),

            "wind_speed":
                weather_context.get("wind_speed"),


            # User/crop context
            "growth_stage":
                growth_stage,

            "soil_condition":
                soil_condition,

            "soil_ph":
                soil_ph,

            "irrigation":
                irrigation,

            "location":
                location,

            "visible_symptoms":
                visible_symptoms
                if visible_symptoms
                else []
        }


        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        return {

            "success": True,

            "location": weather_result.get(
                "location",
                {
                    "latitude": latitude,
                    "longitude": longitude
                }
            ),

            "weather": weather_result.get(
                "weather",
                {}
            ),

            "context": context,

            "source": weather_result.get(
                "source",
                "Open-Meteo"
            ),

            "retrieved_at": weather_result.get(
                "retrieved_at"
            )
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("TESTING CONTEXT SERVICE")
    print("=" * 70)


    service = ContextService()


    result = service.get_context(

        # Example coordinates
        latitude=12.9716,
        longitude=77.5946,

        # Crop information
        growth_stage="vegetative",

        # Soil information
        soil_condition="moist",
        soil_ph=6.5,

        # Irrigation
        irrigation="normal",

        # Location
        location="Test Location",

        # Symptoms
        visible_symptoms=[
            "small brown spots on leaves"
        ]
    )


    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)


    if result.get("success"):

        print("\nLocation:")
        print(result["location"])


        print("\nWeather:")

        weather = result.get(
            "weather",
            {}
        )

        for key, value in weather.items():

            print(
                f"  {key}: {value}"
            )


        print("\nContext:")

        context = result.get(
            "context",
            {}
        )

        for key, value in context.items():

            print(
                f"  {key}: {value}"
            )


    else:

        print(
            "\nWeather/context error:"
        )

        print(
            result.get(
                "error"
            )
        )


    print()
    print("=" * 70)
    print("CONTEXT SERVICE TEST COMPLETED")
    print("=" * 70)