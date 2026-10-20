"""
Weather Service
============================================================

Provides real-time environmental context for the
AI Crop Health system.

Uses latitude/longitude to obtain:

    - Temperature
    - Humidity
    - Rainfall
    - Wind speed
    - Weather condition
    - Weather code

Provider:
    Open-Meteo

The returned information can be passed to the
AI crop-health predictor and recommendation engine.

IMPORTANT:
This service provides environmental context.
It does NOT diagnose plant disease.
"""

import requests
from datetime import datetime, timezone


# ============================================================
# CONFIGURATION
# ============================================================

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

REQUEST_TIMEOUT = 15


# ============================================================
# WEATHER SERVICE
# ============================================================

class WeatherService:

    def __init__(self):

        print("=" * 70)
        print("WEATHER SERVICE")
        print("=" * 70)

        print("Provider: Open-Meteo")
        print("Real-time weather context enabled")

        print("=" * 70)


    # ========================================================
    # GET WEATHER
    # ========================================================

    def get_weather(
        self,
        latitude,
        longitude
    ):

        # ----------------------------------------------------
        # VALIDATE COORDINATES
        # ----------------------------------------------------

        try:

            latitude = float(latitude)
            longitude = float(longitude)

        except (
            TypeError,
            ValueError
        ):

            return {

                "success": False,

                "error":
                    "Invalid latitude or longitude."
            }


        # ----------------------------------------------------
        # VALIDATE COORDINATE RANGE
        # ----------------------------------------------------

        if not -90 <= latitude <= 90:

            return {

                "success": False,

                "error":
                    "Latitude must be between -90 and 90."
            }


        if not -180 <= longitude <= 180:

            return {

                "success": False,

                "error":
                    "Longitude must be between -180 and 180."
            }


        # ----------------------------------------------------
        # OPEN-METEO PARAMETERS
        # ----------------------------------------------------

        params = {

            "latitude":
                latitude,

            "longitude":
                longitude,

            "current":
                (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "precipitation,"
                    "rain,"
                    "showers,"
                    "weather_code,"
                    "wind_speed_10m"
                ),

            "hourly":
                (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "precipitation,"
                    "rain"
                ),

            "timezone":
                "auto",

            "past_days":
                1,

            "forecast_days":
                2
        }


        # ----------------------------------------------------
        # REQUEST OPEN-METEO
        # ----------------------------------------------------

        try:

            print()
            print("Fetching weather from Open-Meteo...")
            print(
                "Latitude:",
                latitude
            )
            print(
                "Longitude:",
                longitude
            )


            response = requests.get(

                OPEN_METEO_URL,

                params=params,

                timeout=REQUEST_TIMEOUT
            )


            response.raise_for_status()


            data = response.json()


        except requests.RequestException as error:

            print()
            print(
                "Weather request failed:",
                str(error)
            )


            return {

                "success": False,

                "error":
                    (
                        "Weather service request failed: "
                        + str(error)
                    )
            }


        except ValueError as error:

            print()
            print(
                "Invalid weather JSON:",
                str(error)
            )


            return {

                "success": False,

                "error":
                    "Weather service returned invalid JSON."
            }


        # ----------------------------------------------------
        # CHECK RESPONSE
        # ----------------------------------------------------

        if not isinstance(
            data,
            dict
        ):

            return {

                "success": False,

                "error":
                    "Invalid response from Open-Meteo."
            }


        current = data.get(
            "current",
            {}
        )


        if not isinstance(
            current,
            dict
        ):

            return {

                "success": False,

                "error":
                    "Current weather data is missing."
            }


        # ====================================================
        # CURRENT WEATHER VALUES
        # ====================================================

        temperature = current.get(
            "temperature_2m"
        )


        humidity = current.get(
            "relative_humidity_2m"
        )


        precipitation = current.get(
            "precipitation",
            0
        )


        rain = current.get(
            "rain",
            0
        )


        showers = current.get(
            "showers",
            0
        )


        wind_speed = current.get(
            "wind_speed_10m"
        )


        weather_code = current.get(
            "weather_code"
        )


        # ====================================================
        # NORMALIZE NUMERIC VALUES
        # ====================================================

        temperature = self._safe_float(
            temperature
        )


        humidity = self._safe_float(
            humidity
        )


        precipitation = self._safe_float(
            precipitation,
            default=0.0
        )


        rain = self._safe_float(
            rain,
            default=0.0
        )


        showers = self._safe_float(
            showers,
            default=0.0
        )


        wind_speed = self._safe_float(
            wind_speed
        )


        weather_code = self._safe_int(
            weather_code
        )


        # ====================================================
        # MAKE SURE REQUIRED VALUES EXIST
        # ====================================================

        if temperature is None:

            return {

                "success": False,

                "error":
                    "Open-Meteo did not return temperature."
            }


        if humidity is None:

            return {

                "success": False,

                "error":
                    "Open-Meteo did not return humidity."
            }


        # ====================================================
        # INTERPRET WEATHER
        # ====================================================

        rainfall_status = (
            self._classify_rainfall(

                precipitation,

                rain,

                showers
            )
        )


        weather_condition = (
            self._weather_code_to_text(
                weather_code
            )
        )


        humidity_status = (
            self._classify_humidity(
                humidity
            )
        )


        temperature_status = (
            self._classify_temperature(
                temperature
            )
        )


        # ====================================================
        # WEATHER OBJECT
        # ====================================================

        weather_data = {

            "temperature":
                temperature,

            "temperature_unit":
                "°C",

            "temperature_status":
                temperature_status,

            "humidity":
                humidity,

            "humidity_unit":
                "%",

            "humidity_status":
                humidity_status,

            "precipitation":
                precipitation,

            "rain":
                rain,

            "showers":
                showers,

            "rainfall_status":
                rainfall_status,

            "wind_speed":
                wind_speed,

            "wind_speed_unit":
                "km/h",

            "weather_code":
                weather_code,

            "condition":
                weather_condition
        }


        # ====================================================
        # CONTEXT OBJECT
        # ====================================================

        context_data = {

            "temperature":
                temperature,

            "humidity":
                humidity,

            "recent_rainfall":
                rainfall_status,

            "weather_condition":
                weather_condition,

            "wind_speed":
                wind_speed
        }


        # ====================================================
        # FINAL RESULT
        #
        # IMPORTANT:
        #
        # Values are provided BOTH:
        #
        # result["weather"]["temperature"]
        #
        # AND
        #
        # result["temperature"]
        #
        # This makes the service compatible with
        # the current app.py.
        # ====================================================

        result = {

            "success":
                True,


            # ------------------------------------------------
            # LOCATION
            # ------------------------------------------------

            "location": {

                "latitude":
                    latitude,

                "longitude":
                    longitude
            },


            # ------------------------------------------------
            # NESTED WEATHER
            # ------------------------------------------------

            "weather":
                weather_data,


            # ------------------------------------------------
            # NESTED CONTEXT
            # ------------------------------------------------

            "context":
                context_data,


            # ------------------------------------------------
            # TOP-LEVEL VALUES
            #
            # These are important for your current app.py.
            # ------------------------------------------------

            "temperature":
                temperature,

            "temperature_unit":
                "°C",

            "temperature_status":
                temperature_status,


            "humidity":
                humidity,

            "humidity_unit":
                "%",

            "humidity_status":
                humidity_status,


            "precipitation":
                precipitation,

            "rain":
                rain,

            "showers":
                showers,


            "rainfall_status":
                rainfall_status,


            "wind_speed":
                wind_speed,

            "wind_speed_unit":
                "km/h",


            "weather_code":
                weather_code,


            "condition":
                weather_condition,


            # ------------------------------------------------
            # SOURCE
            # ------------------------------------------------

            "source":
                "Open-Meteo",


            # ------------------------------------------------
            # RETRIEVAL TIME
            # ------------------------------------------------

            "retrieved_at":
                datetime.now(
                    timezone.utc
                ).isoformat()
        }


        # ====================================================
        # DEBUG OUTPUT
        # ====================================================

        print()
        print(
            "Weather retrieved successfully."
        )

        print()

        print(
            "Temperature:",
            temperature,
            "°C"
        )

        print(
            "Humidity:",
            humidity,
            "%"
        )

        print(
            "Rainfall:",
            rainfall_status
        )

        print(
            "Condition:",
            weather_condition
        )

        print(
            "Wind:",
            wind_speed,
            "km/h"
        )

        print(
            "Weather code:",
            weather_code
        )

        print()


        return result


    # ========================================================
    # SAFE FLOAT
    # ========================================================

    def _safe_float(
        self,
        value,
        default=None
    ):

        if value is None:

            return default


        try:

            return float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return default


    # ========================================================
    # SAFE INTEGER
    # ========================================================

    def _safe_int(
        self,
        value,
        default=None
    ):

        if value is None:

            return default


        try:

            return int(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return default


    # ========================================================
    # RAINFALL CLASSIFICATION
    # ========================================================

    def _classify_rainfall(
        self,
        precipitation,
        rain,
        showers
    ):

        precipitation = (
            self._safe_float(
                precipitation,
                0.0
            )
        )


        rain = (
            self._safe_float(
                rain,
                0.0
            )
        )


        showers = (
            self._safe_float(
                showers,
                0.0
            )
        )


        total = max(

            precipitation,

            rain,

            showers
        )


        if total <= 0:

            return "none"


        elif total < 2.5:

            return "low"


        elif total < 10:

            return "moderate"


        else:

            return "high"


    # ========================================================
    # HUMIDITY CLASSIFICATION
    # ========================================================

    def _classify_humidity(
        self,
        humidity
    ):

        if humidity is None:

            return "unknown"


        if humidity < 40:

            return "low"


        elif humidity < 70:

            return "moderate"


        elif humidity < 85:

            return "high"


        else:

            return "very_high"


    # ========================================================
    # TEMPERATURE CLASSIFICATION
    # ========================================================

    def _classify_temperature(
        self,
        temperature
    ):

        if temperature is None:

            return "unknown"


        if temperature < 10:

            return "cold"


        elif temperature < 20:

            return "cool"


        elif temperature < 30:

            return "moderate"


        elif temperature < 35:

            return "high"


        else:

            return "very_high"


    # ========================================================
    # WEATHER CODE → TEXT
    # ========================================================

    def _weather_code_to_text(
        self,
        code
    ):

        weather_codes = {

            0:
                "Clear sky",

            1:
                "Mainly clear",

            2:
                "Partly cloudy",

            3:
                "Overcast",

            45:
                "Fog",

            48:
                "Depositing rime fog",

            51:
                "Light drizzle",

            53:
                "Moderate drizzle",

            55:
                "Dense drizzle",

            56:
                "Light freezing drizzle",

            57:
                "Dense freezing drizzle",

            61:
                "Slight rain",

            63:
                "Moderate rain",

            65:
                "Heavy rain",

            66:
                "Light freezing rain",

            67:
                "Heavy freezing rain",

            71:
                "Slight snow",

            73:
                "Moderate snow",

            75:
                "Heavy snow",

            77:
                "Snow grains",

            80:
                "Slight rain showers",

            81:
                "Moderate rain showers",

            82:
                "Violent rain showers",

            85:
                "Slight snow showers",

            86:
                "Heavy snow showers",

            95:
                "Thunderstorm",

            96:
                "Thunderstorm with slight hail",

            99:
                "Thunderstorm with heavy hail"
        }


        return weather_codes.get(

            code,

            "Unknown"
        )


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 70)
    print("TESTING WEATHER SERVICE")
    print("=" * 70)


    # Example coordinates.
    #
    # You can replace these with your actual coordinates.

    latitude = 12.9716

    longitude = 77.5946


    service = WeatherService()


    result = service.get_weather(

        latitude,

        longitude
    )


    print()


    if result.get("success"):

        print(
            "Weather retrieved successfully."
        )


        print()
        print("Location:")

        print(
            result["location"]
        )


        print()
        print("Top-level weather values:")

        print(
            "Temperature:",
            result["temperature"],
            "°C"
        )

        print(
            "Humidity:",
            result["humidity"],
            "%"
        )

        print(
            "Rainfall:",
            result["rainfall_status"]
        )

        print(
            "Condition:",
            result["condition"]
        )

        print(
            "Wind:",
            result["wind_speed"],
            "km/h"
        )


        print()
        print("Nested weather:")

        for key, value in result["weather"].items():

            print(
                f"  {key}: {value}"
            )


        print()
        print("Context:")

        for key, value in result["context"].items():

            print(
                f"  {key}: {value}"
            )


    else:

        print(
            "Weather retrieval failed:"
        )

        print(
            result.get(
                "error",
                "Unknown error"
            )
        )


    print()

    print("=" * 70)
    print("WEATHER SERVICE TEST COMPLETED")
    print("=" * 70)