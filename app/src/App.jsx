import { useEffect, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:5000/api";


// ============================================================
// WEATHER CODE
// ============================================================

function getWeatherCondition(code) {
  const conditions = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",

    45: "Fog",
    48: "Depositing rime fog",

    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",

    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",

    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",

    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",

    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
  };

  return conditions[Number(code)] || "Unknown";
}


// ============================================================
// SAFE NUMBER
// ============================================================

function safeNumber(value, fallback = 0) {
  const n = Number(value);

  return Number.isFinite(n) ? n : fallback;
}


// ============================================================
// RECOMMENDATION TEXT
// ============================================================

function recommendationText(item) {
  if (typeof item === "string") {
    return item.trim();
  }

  if (!item || typeof item !== "object") {
    return "";
  }

  return String(
    item.recommendation ||
    item.action ||
    item.solution ||
    item.message ||
    item.text ||
    item.description ||
    item.instruction ||
    ""
  ).trim();
}


// ============================================================
// GENERIC ENVIRONMENTAL TEXT CHECK
//
// IMPORTANT:
// We use this ONLY for old/generic fallback structures.
//
// We DO NOT filter the main backend recommendation array.
// Your recommendation_engine.py already generates the correct
// crop-specific recommendations.
// ============================================================

function isGenericContextRecommendation(text) {
  if (!text) {
    return true;
  }

  const value = String(text).trim().toLowerCase();

  const genericPatterns = [
    "current temperature",
    "current relative humidity",
    "relative humidity is approximately",
    "recent rainfall condition",
    "reported soil condition",
    "reported soil ph",
    "soil ph is",
    "irrigation condition",
    "weather condition",
    "current wind",
    "location:",
    "for banana, humid conditions",
    "for maize, humid conditions",
    "high humidity can favor",
    "environmental action",
    "environment-specific actions",
  ];

  return genericPatterns.some((pattern) =>
    value.includes(pattern)
  );
}


// ============================================================
// IMPORTANT: EXTRACT BACKEND RECOMMENDATIONS
//
// YOUR PYTHON TEST PROVED:
//
// generate_recommendations(...) -> LIST
//
// Example:
//
// [
//   "CROP-SPECIFIC ACTION PLAN — Maize",
//   "DISEASE SOLUTION — Maize Leaf Blight",
//   "Inspect additional maize leaves...",
//   ...
// ]
//
// Flask then sends:
//
// "recommendations": [ ... ]
//
// Therefore result.recommendations itself is an ARRAY.
//
// This function handles that FIRST.
// ============================================================

function extractBackendSpecificRecommendations(result) {
  if (!result) {
    return [];
  }

  const output = [];

  const prediction = result.prediction || {};

  const root = result.recommendations;


  // ==========================================================
  // CASE 1 — CURRENT BACKEND
  //
  // recommendations = [...]
  //
  // THIS IS YOUR CURRENT STRUCTURE.
  // ==========================================================

  if (Array.isArray(root)) {
    root.forEach((item) => {
      const text = recommendationText(item);

      if (text) {
        output.push(text);
      }
    });

    return [...new Set(output)];
  }


  // ==========================================================
  // CASE 2 — OLD/NESTED STRUCTURE
  //
  // recommendations = {
  //     recommendations: [...]
  // }
  // ==========================================================

  if (root && typeof root === "object") {

    const possibleArrays = [

      root.recommendations,

      root.crop_specific_recommendations,

      root.crop_specific,

      root.actionable_solutions,

      root.specific_recommendations,

      root.disease_recommendations,

      root.nutrient_recommendations,

      root.solutions,

      root.actions,

    ];


    possibleArrays.forEach((arr) => {

      if (Array.isArray(arr)) {

        arr.forEach((item) => {

          const text = recommendationText(item);

          if (text) {
            output.push(text);
          }

        });

      }

    });


    // --------------------------------------------------------
    // Nested objects
    // --------------------------------------------------------

    const nestedObjects = [

      root.disease,

      root.nutrient,

      root.disease_solution,

      root.nutrient_solution,

      root.solution,

    ];


    nestedObjects.forEach((obj) => {

      if (!obj) {
        return;
      }


      if (Array.isArray(obj)) {

        obj.forEach((item) => {

          const text =
            recommendationText(item);

          if (text) {
            output.push(text);
          }

        });

        return;
      }


      if (typeof obj === "object") {

        const text =
          recommendationText(obj);

        if (text) {
          output.push(text);
        }

      }

    });

  }


  // ==========================================================
  // CASE 3 — prediction.recommendations
  // ==========================================================

  if (
    Array.isArray(
      prediction.recommendations
    )
  ) {

    prediction.recommendations.forEach(
      (item) => {

        const text =
          recommendationText(item);

        if (text) {
          output.push(text);
        }

      }
    );

  }


  // ==========================================================
  // CASE 4 — recommendation_details
  // ==========================================================

  const details =
    result.recommendation_details;

  if (
    details &&
    Array.isArray(
      details.recommendations
    )
  ) {

    details.recommendations.forEach(
      (item) => {

        const text =
          recommendationText(item);

        if (text) {
          output.push(text);
        }

      }
    );

  }


  return [
    ...new Set(
      output.filter(Boolean)
    )
  ];
}


// ============================================================
// FRONTEND FALLBACK
//
// This is ONLY used if the backend somehow does not return
// recommendations.
//
// The normal path is still the Python recommendation engine.
// ============================================================

function buildSpecificFallbackRecommendations(
  crop,
  disease,
  nutrient
) {

  const recommendations = [];

  const cropName =
    String(crop || "crop")
      .trim()
      .toLowerCase();

  const diseaseName =
    String(
      disease?.name ||
      disease?.label ||
      ""
    )
      .trim()
      .toLowerCase();

  const nutrientName =
    String(
      nutrient?.name ||
      ""
    )
      .trim()
      .toLowerCase();


  // ==========================================================
  // DISEASE
  // ==========================================================

  if (
    diseaseName &&
    diseaseName !== "healthy" &&
    diseaseName !== "unknown"
  ) {

    // --------------------------------------------------------
    // MAIZE
    // --------------------------------------------------------

    if (cropName === "maize") {

      if (
        diseaseName.includes("leaf blight") ||
        diseaseName.includes("northern leaf blight")
      ) {

        recommendations.push(
          "Inspect additional maize leaves for expanding brown or gray elongated lesions."
        );

        recommendations.push(
          "Remove severely affected maize leaves where practical and manage infected crop residue."
        );

        recommendations.push(
          "Improve airflow around maize plants and avoid unnecessarily prolonged leaf wetness."
        );

        recommendations.push(
          "Avoid unnecessary overhead irrigation when foliage remaining wet can favor disease development."
        );

        recommendations.push(
          "Monitor nearby maize plants for similar lesions."
        );

        recommendations.push(
          "If Maize Leaf Blight is confirmed, follow the locally recommended disease-management program and use only a registered fungicide according to its product label."
        );

      } else {

        recommendations.push(
          `Inspect additional maize leaves for symptoms consistent with ${disease?.name || "the detected disease"}.`
        );

        recommendations.push(
          "Remove severely affected leaves and manage infected crop residue where practical."
        );

        recommendations.push(
          "Maintain good field airflow and avoid unnecessary prolonged leaf wetness."
        );

        recommendations.push(
          `If ${disease?.name || "the suspected disease"} is confirmed, follow locally recommended disease-management practices and use only registered treatment products according to their labels.`
        );

      }

    }


    // --------------------------------------------------------
    // BANANA
    // --------------------------------------------------------

    else if (cropName === "banana") {

      if (
        diseaseName.includes("sigatoka") ||
        diseaseName.includes("leaf streak") ||
        diseaseName.includes("leaf spot")
      ) {

        recommendations.push(
          "Remove severely infected banana leaves and safely manage infected leaf material to reduce disease carryover."
        );

        recommendations.push(
          "Improve air circulation between banana plants and avoid unnecessary prolonged leaf wetness."
        );

        recommendations.push(
          "Avoid unnecessary overhead irrigation, especially during humid conditions."
        );

        recommendations.push(
          "Monitor newly emerging banana leaves for expansion of streaks or leaf lesions."
        );

        recommendations.push(
          "If confirmed, follow the locally recommended banana disease-management program and use only registered products according to the product label."
        );

      } else {

        recommendations.push(
          `Inspect additional banana leaves for symptoms consistent with ${disease?.name || "the detected disease"}.`
        );

        recommendations.push(
          "Remove or manage severely affected banana leaves and infected residue where practical."
        );

        recommendations.push(
          "Improve airflow around banana plants and avoid unnecessary prolonged leaf wetness."
        );

        recommendations.push(
          `If ${disease?.name || "the suspected disease"} is confirmed, follow locally recommended banana disease-management practices.`
        );

      }

    }


    // --------------------------------------------------------
    // GENERAL CROP
    // --------------------------------------------------------

    else {

      recommendations.push(
        `Inspect additional ${cropName} leaves for symptoms consistent with ${disease?.name || "the detected disease"}.`
      );

      recommendations.push(
        `Remove or manage severely affected ${cropName} leaves and infected crop residue where practical.`
      );

      recommendations.push(
        `Improve airflow around the ${cropName} crop and avoid unnecessary prolonged leaf wetness.`
      );

      recommendations.push(
        `If ${disease?.name || "the suspected disease"} is confirmed, follow locally recommended disease-management practices and use only registered treatment products according to their labels.`
      );

    }

  }


  // ==========================================================
  // NUTRIENT
  // ==========================================================

  if (
    nutrientName &&
    nutrientName !== "healthy" &&
    nutrientName !== "not evaluated" &&
    nutrientName !== "unknown"
  ) {

    if (
      nutrientName.includes("potassium")
    ) {

      recommendations.push(
        `Inspect older ${cropName} leaves for marginal yellowing, chlorosis or necrosis associated with potassium deficiency.`
      );

      recommendations.push(
        "Confirm potassium status using soil or plant-tissue testing before applying fertilizer."
      );

      recommendations.push(
        "Check soil potassium availability and review the current fertilizer program."
      );

      recommendations.push(
        "Review irrigation and soil moisture because suitable root-zone conditions are important for nutrient uptake."
      );

      recommendations.push(
        "If potassium deficiency is confirmed, apply an appropriate potassium fertilizer according to soil-test recommendations and the product label."
      );

      recommendations.push(
        "Avoid excessive potassium application when soil potassium is already adequate."
      );

    } else {

      recommendations.push(
        `Confirm the suspected ${nutrient?.name || "nutrient deficiency"} using soil or plant-tissue testing before applying fertilizer.`
      );

      recommendations.push(
        `Inspect ${cropName} leaves for symptoms consistent with ${nutrient?.name || "the detected nutrient deficiency"}.`
      );

      recommendations.push(
        "Review soil nutrient availability and the current fertilizer program."
      );

      recommendations.push(
        `If ${nutrient?.name || "the deficiency"} is confirmed, correct the nutrient imbalance according to soil-test recommendations and the product label.`
      );

    }

  }


  return [
    ...new Set(
      recommendations
        .filter(Boolean)
        .map((item) => String(item).trim())
        .filter(Boolean)
    )
  ];
}


// ============================================================
// FINAL RECOMMENDATIONS
//
// PRIORITY:
//
// 1. Python RecommendationEngine output
// 2. Crop-specific frontend fallback
// 3. Old predictor output
//
// Environmental context is NOT allowed to replace the actual
// crop-specific solution list.
// ============================================================

function getFinalRecommendations(result) {

  const prediction =
    result?.prediction || {};

  const disease =
    prediction.disease || {};

  const nutrient =
    prediction.nutrient || {};

  const crop =
    result?.crop ||
    prediction.crop ||
    "crop";


  // ==========================================================
  // 1. PYTHON RECOMMENDATION ENGINE
  // ==========================================================

  const backendRecommendations =
    extractBackendSpecificRecommendations(
      result
    );


  console.log(
    "=============================================="
  );

  console.log(
    "FRONTEND RECOMMENDATION CHECK"
  );

  console.log(
    "Crop:",
    crop
  );

  console.log(
    "Disease:",
    disease.name
  );

  console.log(
    "Nutrient:",
    nutrient.name
  );

  console.log(
    "Backend recommendations:",
    backendRecommendations
  );

  console.log(
    "Backend recommendation count:",
    backendRecommendations.length
  );

  console.log(
    "=============================================="
  );


  // ==========================================================
  // THIS IS THE IMPORTANT LINE
  //
  // If Python returned 23 recommendations, use those 23.
  // ==========================================================

  if (
    backendRecommendations.length > 0
  ) {

    return backendRecommendations;
  }


  // ==========================================================
  // 2. FRONTEND SPECIFIC FALLBACK
  // ==========================================================

  const fallback =
    buildSpecificFallbackRecommendations(
      crop,
      disease,
      nutrient
    );


  if (
    fallback.length > 0
  ) {

    return fallback;
  }


  // ==========================================================
  // 3. LAST FALLBACK
  // ==========================================================

  let oldRecommendations = [];


  if (
    Array.isArray(
      prediction.recommendations
    )
  ) {

    oldRecommendations =
      prediction.recommendations;

  }


  return oldRecommendations
    .map(
      recommendationText
    )
    .filter(
      (text) =>
        text &&
        !isGenericContextRecommendation(text)
    );
}


// ============================================================
// MAIN APP
// ============================================================

function App() {

  // ==========================================================
  // IMAGES
  // ==========================================================

  const [images, setImages] =
    useState([]);


  // ==========================================================
  // CROP
  // ==========================================================

  const [crop, setCrop] =
    useState("banana");


  // ==========================================================
  // LOCATION
  // ==========================================================

  const [location, setLocation] =
    useState(null);

  const [locationLoading, setLocationLoading] =
    useState(false);


  // ==========================================================
  // WEATHER
  // ==========================================================

  const [weather, setWeather] =
    useState(null);

  const [weatherLoading, setWeatherLoading] =
    useState(false);


  // ==========================================================
  // RESULTS
  // ==========================================================

  const [results, setResults] =
    useState([]);

  const [loading, setLoading] =
    useState(false);


  // ==========================================================
  // DIGITAL TWIN
  // ==========================================================

  const [digitalTwin, setDigitalTwin] =
    useState(null);

  // Persistent Digital Twin field identity.
  // Keeping this ID allows later analyses to update the same farm
  // instead of creating a completely new field every time.
  const [digitalTwinFieldId, setDigitalTwinFieldId] =
    useState(null);

  // Digital Twin is generated only after the user explicitly
  // requests it from the completed crop-health analysis.
  const [digitalTwinReady, setDigitalTwinReady] =
    useState(false);


  // ==========================================================
  // STATUS
  // ==========================================================

  const [error, setError] =
    useState("");

  const [message, setMessage] =
    useState("");

  const [apiConnected, setApiConnected] =
    useState(false);


  // ==========================================================
  // INITIALIZATION
  // ==========================================================

  useEffect(() => {
    checkAPI();
    loadSavedDigitalTwin();
  }, []);


  // ==========================================================
  // LOAD SAVED DIGITAL TWIN
  // ==========================================================

  async function loadSavedDigitalTwin() {

    try {

      const response =
        await fetch(
          `${API_BASE}/digital-twin`
        );

      const data =
        await response.json();

      if (
        !response.ok ||
        data.success === false
      ) {

        throw new Error(
          data.error ||
          "Unable to load saved Digital Twin."
        );
      }

      const twins =
        Array.isArray(
          data.digitalTwins
        )
          ? data.digitalTwins
          : [];

      if (
        twins.length > 0
      ) {

        const latestTwin =
          twins[0];

        setDigitalTwin(
          latestTwin
        );

        setDigitalTwinFieldId(
          latestTwin.fieldId ||
          null
        );

        console.log(
          "Loaded saved Digital Twin:",
          latestTwin
        );

      }

    } catch (err) {

      // Loading saved history must not stop the normal
      // crop-analysis workflow.
      console.warn(
        "Saved Digital Twin could not be loaded:",
        err
      );

    }
  }


  // ==========================================================
  // CHECK API
  // ==========================================================

  async function checkAPI() {

    try {

      const response =
        await fetch(
          `${API_BASE}/status`
        );

      if (!response.ok) {
        throw new Error(
          "Flask API unavailable"
        );
      }

      const data =
        await response.json();

      console.log(
        "Flask API status:",
        data
      );

      setApiConnected(true);

    } catch (err) {

      console.error(
        "API connection error:",
        err
      );

      setApiConnected(false);

    }

  }


  // ==========================================================
  // IMAGE UPLOAD
  // ==========================================================

  function handleImageChange(event) {

    const files =
      Array.from(
        event.target.files || []
      );

    if (!files.length) {
      return;
    }


    const validTypes = [
      "image/jpeg",
      "image/jpg",
      "image/png",
      "image/webp",
    ];


    const validFiles =
      files.filter(
        (file) =>
          validTypes.includes(
            file.type
          )
      );


    if (!validFiles.length) {

      setError(
        "Please upload JPG, PNG or WEBP images."
      );

      return;
    }


    images.forEach(
      (item) => {

        if (item.preview) {

          URL.revokeObjectURL(
            item.preview
          );

        }

      }
    );


    const preparedImages =
      validFiles.map(
        (file) => ({

          file,

          preview:
            URL.createObjectURL(
              file
            ),

          id:
            `${file.name}-${file.lastModified}-${Math.random()}`

        })
      );


    setImages(
      preparedImages
    );

    setResults([]);

    setDigitalTwin(null);
    setDigitalTwinReady(false);

    setError("");

    setMessage(
      `${preparedImages.length} crop image(s) selected.`
    );


    setTimeout(
      () => {
        getLocation();
      },
      300
    );

  }


  // ==========================================================
  // GET LOCATION
  // ==========================================================

  function getLocation() {

    setError("");

    if (
      !navigator.geolocation
    ) {

      setError(
        "Geolocation is not supported by this browser."
      );

      return;
    }


    setLocationLoading(
      true
    );


    navigator.geolocation.getCurrentPosition(

      async (position) => {

        const coords = {

          latitude:
            position.coords.latitude,

          longitude:
            position.coords.longitude,

        };


        setLocation(
          coords
        );

        setLocationLoading(
          false
        );


        setMessage(
          "Location obtained. Retrieving real-time weather..."
        );


        await getWeather(
          coords
        );

      },


      (err) => {

        console.error(
          "Location error:",
          err
        );


        setLocationLoading(
          false
        );


        setError(
          "Location permission was not granted. Please allow location access in your browser."
        );

      },


      {

        enableHighAccuracy:
          true,

        timeout:
          15000,

        maximumAge:
          300000,

      }

    );

  }


  // ==========================================================
  // GET WEATHER
  // ==========================================================

  async function getWeather(
    coords
  ) {

    setWeatherLoading(
      true
    );

    setError("");


    try {

      const url =
        `${API_BASE}/weather` +
        `?latitude=${encodeURIComponent(
          coords.latitude
        )}` +
        `&longitude=${encodeURIComponent(
          coords.longitude
        )}`;


      console.log(
        "Backend weather URL:",
        url
      );


      const response =
        await fetch(
          url
        );


      const responseText =
        await response.text();


      let data;


      try {

        data =
          JSON.parse(
            responseText
          );

      } catch {

        throw new Error(
          `Weather API returned invalid JSON: ${responseText.slice(
            0,
            300
          )}`
        );

      }


      if (
        !response.ok ||
        data.success === false
      ) {

        throw new Error(
          data.error ||
          data.message ||
          "Weather retrieval failed."
        );

      }


      const context =
        data.context ||
        data.weather ||
        data;


      const weatherObject = {

        temperature:
          safeNumber(
            context.temperature ??
            data.temperature
          ),

        humidity:
          safeNumber(
            context.humidity ??
            data.humidity
          ),

        rainfall_status:
          context.recent_rainfall ??
          context.rainfall_status ??
          data.rainfall_status ??
          "none",

        condition:
          context.weather_condition ??
          context.condition ??
          data.condition ??
          "Unknown",

        wind_speed:
          safeNumber(
            context.wind_speed ??
            data.wind_speed
          ),

        source:
          data.source ||
          "Open-Meteo",

      };


      console.log(
        "FINAL WEATHER:",
        weatherObject
      );


      setWeather(
        weatherObject
      );


      setMessage(
        "Real-time environmental information retrieved successfully."
      );


    } catch (err) {

      console.error(
        "WEATHER ERROR:",
        err
      );


      setWeather(
        null
      );


      setError(
        `Weather retrieval failed: ${err.message}`
      );


    } finally {

      setWeatherLoading(
        false
      );

    }

  }


  // ==========================================================
  // ANALYZE CROPS
  // ==========================================================

  async function analyzeCrops() {

    if (
      !images.length
    ) {

      setError(
        "Please upload at least one crop image."
      );

      return;
    }


    if (
      !location
    ) {

      setError(
        "Please allow location access first."
      );

      return;
    }


    if (
      !weather
    ) {

      setError(
        "Please wait for weather information before analysis."
      );

      return;
    }


    if (
      !apiConnected
    ) {

      setError(
        "Flask API is not connected. Start ai/api/app.py first."
      );

      return;
    }


    setLoading(
      true
    );

    setError("");

    setResults([]);


    try {

      const newResults = [];


      for (
        const item of images
      ) {

        const formData =
          new FormData();


        formData.append(
          "image",
          item.file
        );


        formData.append(
          "crop",
          crop
        );


        // ==================================================
        // AGRICULTURAL CONTEXT
        // ==================================================

        formData.append(
          "growth_stage",
          "vegetative"
        );


        formData.append(
          "soil_condition",
          "moist"
        );


        formData.append(
          "soil_ph",
          "6.5"
        );


        formData.append(
          "irrigation",
          "normal"
        );


        formData.append(
          "recent_rainfall",
          weather.rainfall_status ||
          "low"
        );


        formData.append(
          "visible_symptoms",
          ""
        );


        formData.append(
          "location",
          `${location.latitude},${location.longitude}`
        );


        formData.append(
          "latitude",
          String(
            location.latitude
          )
        );


        formData.append(
          "longitude",
          String(
            location.longitude
          )
        );


        // ==================================================
        // DEBUG
        // ==================================================

        console.log(
          "================================================"
        );

        console.log(
          "SENDING PREDICTION REQUEST"
        );

        console.log(
          "Image:",
          item.file.name
        );

        console.log(
          "Crop:",
          crop
        );

        console.log(
          "Location:",
          location
        );

        console.log(
          "Weather:",
          weather
        );


        // ==================================================
        // SEND TO FLASK
        // ==================================================

        const response =
          await fetch(
            `${API_BASE}/predict`,
            {

              method:
                "POST",

              body:
                formData,

            }
          );


        const responseText =
          await response.text();


        console.log(
          "RAW FLASK RESPONSE:",
          responseText
        );


        let data;


        try {

          data =
            JSON.parse(
              responseText
            );

        } catch {

          throw new Error(
            `Flask returned invalid JSON: ${responseText.slice(
              0,
              500
            )}`
          );

        }


        console.log(
          "COMPLETE PREDICTION RESPONSE:",
          data
        );


        // ==================================================
        // CRITICAL DEBUG
        //
        // This will show exactly what Flask returned.
        // ==================================================

        console.log(
          "=============================================="
        );

        console.log(
          "RECOMMENDATION DATA FROM FLASK:"
        );

        console.log(
          "Type:",
          Array.isArray(
            data.recommendations
          )
            ? "ARRAY"
            : typeof data.recommendations
        );

        console.log(
          "Recommendations:",
          data.recommendations
        );

        console.log(
          "Recommendation count:",
          Array.isArray(
            data.recommendations
          )
            ? data.recommendations.length
            : "NOT ARRAY"
        );

        console.log(
          "=============================================="
        );


        if (
          !response.ok ||
          data.success === false
        ) {

          throw new Error(
            data.error ||
            data.message ||
            `Prediction failed for ${item.file.name}`
          );

        }


        // ==================================================
        // PRESERVE COMPLETE BACKEND RESPONSE
        // ==================================================

        newResults.push({

          ...data,

          imageName:
            item.file.name,

          preview:
            item.preview,

        });

      }


      setResults(
        newResults
      );


      // Analysis is complete. Do NOT create the Digital Twin
      // automatically. The user can review the results first
      // and explicitly generate the farm Digital Twin.
      setDigitalTwinReady(true);


      setMessage(
        `Analysis completed successfully for ${newResults.length} crop image(s). Review the results and generate the Digital Twin when ready.`
      );


    } catch (err) {

      console.error(
        "ANALYSIS ERROR:",
        err
      );


      setError(
        err.message ||
        "Crop analysis failed."
      );


    } finally {

      setLoading(
        false
      );

    }

  }


  // ==========================================================
  // DIGITAL TWIN
  // ==========================================================

  function createDigitalTwin(
    analysisResults
  ) {

    if (
      !analysisResults ||
      !analysisResults.length
    ) {

      return;
    }


    const plants =
      analysisResults.map(
        (result, index) => {

          const prediction =
            result.prediction ||
            {};


          const disease =
            prediction.disease ||
            {};


          const nutrient =
            prediction.nutrient ||
            {};


          return {

            plantId:
              `PLANT-${index + 1}`,

            image:
              result.imageName,

            crop:
              result.crop ||
              crop,

            disease:
              disease.name ||
              "Unknown",

            diseaseConfidence:
              disease.confidence_percent ??
              disease.confidence ??
              0,

            nutrient:
              nutrient.name ||
              "Not evaluated",

            nutrientConfidence:
              nutrient.confidence_percent ??
              nutrient.confidence ??
              0,

            overallConfidence:
              prediction.overall_confidence_percent ??
              0,

            status:
              prediction.status ||
              "Unknown",

            latitude:
              location?.latitude ??
              null,

            longitude:
              location?.longitude ??
              null,

            weather:
              weather
                ? {

                    temperature:
                      weather.temperature,

                    humidity:
                      weather.humidity,

                    rainfall:
                      weather.rainfall_status,

                    condition:
                      weather.condition,

                    wind_speed:
                      weather.wind_speed,

                  }
                : null,

            agriculturalContext: {

              growthStage:
                "vegetative",

              soilCondition:
                "moist",

              soilPH:
                6.5,

              irrigation:
                "normal",

            },

          };

        }
      );


    const fieldId =
      digitalTwinFieldId ||
      `FIELD-${Date.now()}`;

    const twin = {

      fieldId,

      createdAt:
        digitalTwin?.createdAt ||
        new Date().toISOString(),

      updatedAt:
        new Date().toISOString(),

      crop,

      plantCount:
        plants.length,

      location,

      weather,

      plants,

    };


    console.log(
      "DIGITAL TWIN:",
      twin
    );


    // Update the UI immediately.
    setDigitalTwin(
      twin
    );

    setDigitalTwinFieldId(
      fieldId
    );

    // Persist the same Digital Twin in Flask.
    // The backend stores it in farm_history.json.
    void saveDigitalTwin(
      twin
    );

    // The current analysis has now been converted into a
    // Digital Twin, so hide the generation action until a
    // new analysis is performed.
    setDigitalTwinReady(false);

  }


  // ==========================================================
  // SAVE DIGITAL TWIN TO FLASK
  // ==========================================================

  async function saveDigitalTwin(
    twin
  ) {

    if (
      !twin
    ) {
      return;
    }

    try {

      const response =
        await fetch(
          `${API_BASE}/digital-twin`,
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(
                twin
              ),
          }
        );

      const data =
        await response.json();

      if (
        !response.ok ||
        data.success === false
      ) {

        throw new Error(
          data.error ||
          "Digital Twin could not be saved."
        );
      }

      const savedTwin =
        data.digitalTwin ||
        twin;

      setDigitalTwin(
        savedTwin
      );

      setDigitalTwinFieldId(
        savedTwin.fieldId ||
        twin.fieldId
      );

      console.log(
        "Digital Twin saved successfully:",
        savedTwin
      );

    } catch (err) {

      // Prediction has already completed, so a storage
      // failure should not make the crop analysis fail.
      console.error(
        "DIGITAL TWIN SAVE ERROR:",
        err
      );

      setMessage(
        "Crop analysis completed, but the Digital Twin could not be saved."
      );

    }
  }


  // ==========================================================
  // DOWNLOAD DIGITAL TWIN
  // ==========================================================

  function downloadDigitalTwin() {

    if (
      !digitalTwin
    ) {

      return;
    }


    const blob =
      new Blob(
        [
          JSON.stringify(
            digitalTwin,
            null,
            2
          ),
        ],
        {
          type:
            "application/json",
        }
      );


    const url =
      URL.createObjectURL(
        blob
      );


    const link =
      document.createElement(
        "a"
      );


    link.href =
      url;


    link.download =
      "crop_digital_twin.json";


    document.body.appendChild(
      link
    );


    link.click();


    document.body.removeChild(
      link
    );


    URL.revokeObjectURL(
      url
    );

  }


  // ==========================================================
  // STATUS CLASS
  // ==========================================================

  function getStatusClass(
    status
  ) {

    if (
      !status
    ) {

      return "";
    }


    const value =
      String(
        status
      )
        .toLowerCase();


    if (
      value.includes(
        "healthy"
      ) ||
      value.includes(
        "normal"
      )
    ) {

      return "healthy";
    }


    if (
      value.includes(
        "verification"
      ) ||
      value.includes(
        "uncertain"
      ) ||
      value.includes(
        "moderate"
      ) ||
      value.includes(
        "potential"
      )
    ) {

      return "warning";
    }


    return "danger";

  }


  // ==========================================================
  // RENDER
  // ==========================================================

  return (

    <div className="app">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="header">

        <div className="brand">

          <div className="logo">
            🌱
          </div>

          <div>

            <h1>
              AI Crop Health
            </h1>

            <p>
              AR Digital Twin
            </p>

          </div>

        </div>


        <div className="api-status">

          <span
            className={
              apiConnected
                ? "status-dot"
                : "status-dot offline"
            }
          ></span>

          {apiConnected
            ? "AI API Connected"
            : "AI API Offline"}

        </div>

      </header>


      {/* ======================================================
          HERO
      ====================================================== */}

      <section className="hero">

        <div className="hero-content">

          <p className="eyebrow">
            INTELLIGENT AGRICULTURE
          </p>

          <h2>
            AI-Powered
            <span>
              {" "}Crop Health Detection
            </span>
          </h2>

          <p className="hero-description">

            Upload one or multiple crop images.
            Allow location access to retrieve
            real-time environmental information,
            then analyze crop health and generate
            a field digital twin.

          </p>

        </div>

      </section>


      <main className="main-container">


        {/* =====================================================
            LEFT PANEL
        ===================================================== */}

        <section className="card upload-card">

          <div className="section-title">

            <div className="section-icon">
              📷
            </div>

            <div>

              <h3>
                Analyze Crop
              </h3>

              <p>
                Upload one or multiple crop images
              </p>

            </div>

          </div>


          {/* CROP */}

          <label className="label">
            Select Crop
          </label>


          <select
            value={crop}
            onChange={
              (e) =>
                setCrop(
                  e.target.value
                )
            }
            className="crop-select"
          >

            <option value="banana">
              🍌 Banana
            </option>

            <option value="maize">
              🌽 Maize
            </option>

            <option value="apple">
              🍎 Apple
            </option>

            <option value="tomato">
              🍅 Tomato
            </option>

            <option value="potato">
              🥔 Potato
            </option>

            <option value="rice">
              🌾 Rice
            </option>

            <option value="squash">
              🎃 Squash
            </option>

          </select>


          {/* IMAGE */}

          <label className="label">
            Crop Image
          </label>


          <label className="upload-area">

            {images.length > 0 ? (

              <img
                src={
                  images[0].preview
                }
                alt="Crop preview"
                className="preview-image"
              />

            ) : (

              <div>

                <div className="upload-icon">
                  📷
                </div>

                <p>
                  Click to upload crop image
                </p>

                <small>
                  JPG, PNG or WEBP
                </small>

              </div>

            )}


            <input
              type="file"
              accept="image/jpeg,image/jpg,image/png,image/webp"
              multiple
              onChange={
                handleImageChange
              }
              hidden
            />

          </label>


          {/* SELECTED IMAGES */}

          {images.length > 0 && (

            <div className="selected-images">

              <p>
                {images.length}
                {" "}
                image(s) selected
              </p>


              <div className="thumbnail-grid">

                {images.map(
                  (item) => (

                    <img
                      key={
                        item.id
                      }
                      src={
                        item.preview
                      }
                      alt={
                        item.file.name
                      }
                      className="thumbnail"
                    />

                  )
                )}

              </div>

            </div>

          )}


          {/* LOCATION */}

          <div className="context-card">

            <h4>
              📍 Environmental Context
            </h4>

            <p>
              Your browser location is used to
              retrieve real-time weather.
            </p>


            <button
              type="button"
              className="secondary-btn"
              onClick={
                getLocation
              }
              disabled={
                locationLoading
              }
            >

              {locationLoading
                ? "Getting location..."
                : "Refresh Location"}

            </button>


            {location && (

              <div>

                <p>
                  <strong>
                    Latitude:
                  </strong>{" "}
                  {Number(
                    location.latitude
                  ).toFixed(5)}
                </p>


                <p>
                  <strong>
                    Longitude:
                  </strong>{" "}
                  {Number(
                    location.longitude
                  ).toFixed(5)}
                </p>

              </div>

            )}


            {weatherLoading && (

              <p>
                🌦️ Retrieving real-time weather...
              </p>

            )}


            {weather && (

              <div className="weather-summary">

                <p>
                  🌡️ Temperature:
                  {" "}
                  {weather.temperature}
                  °C
                </p>

                <p>
                  💧 Humidity:
                  {" "}
                  {weather.humidity}
                  %
                </p>

                <p>
                  🌧️ Rainfall:
                  {" "}
                  {weather.rainfall_status}
                </p>

                <p>
                  ☁️ Condition:
                  {" "}
                  {weather.condition}
                </p>

                <p>
                  💨 Wind:
                  {" "}
                  {weather.wind_speed}
                  km/h
                </p>

              </div>

            )}

          </div>


          {/* ERROR */}

          {error && (

            <div className="error-box">

              ⚠️{" "}
              {error}

            </div>

          )}


          {/* MESSAGE */}

          {message && !error && (

            <div className="message-box">

              {message}

            </div>

          )}


          {/* ANALYZE */}

          <button
            className="predict-btn"
            onClick={
              analyzeCrops
            }
            disabled={
              loading ||
              !images.length ||
              !location ||
              !weather ||
              !apiConnected
            }
          >

            {loading ? (

              <>

                <span className="spinner"></span>

                Analyzing...

              </>

            ) : (

              <>
                🔍 Analyze Crop Health
              </>

            )}

          </button>

        </section>


        {/* =====================================================
            RESULTS
        ===================================================== */}

        <section className="card results-card">

          <div className="section-title">

            <div className="section-icon">
              🧠
            </div>

            <div>

              <h3>
                AI Analysis
              </h3>

              <p>
                Prediction, context and
                actionable recommendations
              </p>

            </div>

          </div>


          {results.length === 0 && (

            <div className="empty-state">

              <div>
                🌱
              </div>

              <h3>
                No analysis yet
              </h3>

              <p>
                Upload a crop image and click
                Analyze Crop Health.
              </p>

            </div>

          )}


          {results.length > 0 && (

            <div className="results">

              <div className="result-summary">

                <h3>
                  🌱 Field Analysis
                </h3>

                <p>

                  {results.length}
                  {" "}
                  crop image
                  {results.length !== 1
                    ? "s"
                    : ""}
                  {" "}
                  analyzed.

                </p>

              </div>


              {results.map(
                (
                  result,
                  index
                ) => {

                  const prediction =
                    result.prediction ||
                    {};


                  const disease =
                    prediction.disease ||
                    {};


                  const nutrient =
                    prediction.nutrient ||
                    {};


                  const overall =
                    prediction.overall ||
                    {};


                  // ==================================================
                  // GET THE ACTUAL PYTHON RECOMMENDATION ENGINE LIST
                  // ==================================================

                  const recommendations =
                    getFinalRecommendations(
                      result
                    );


                  console.log(
                    `RESULT ${index + 1} FINAL RECOMMENDATIONS:`,
                    recommendations
                  );


                  return (

                    <div
                      className="result-item"
                      key={
                        result.imageName ||
                        index
                      }
                    >


                      {/* IMAGE */}

                      {result.preview && (

                        <div className="result-image-wrapper">

                          <img
                            src={
                              result.preview
                            }
                            alt={
                              result.imageName ||
                              "Crop"
                            }
                            className="result-image"
                          />

                          <p>
                            {result.imageName}
                          </p>

                        </div>

                      )}


                      {/* =================================================
                          DISEASE / HEALTH
                      ================================================= */}

                      <div className="prediction-card">

                        <h4>
                          🌿 Disease / Health
                        </h4>


                        <h2>

                          {disease.name ||
                            disease.prediction ||
                            "Unknown"}

                        </h2>


                        <p>

                          Confidence:
                          {" "}

                          <strong>

                            {disease.confidence_percent ??
                              disease.confidence ??
                              0}
                            %

                          </strong>

                        </p>


                        {disease.confidence_level && (

                          <p>

                            Level:
                            {" "}

                            <strong>
                              {disease.confidence_level}
                            </strong>

                          </p>

                        )}


                        {disease.message && (

                          <p>
                            {disease.message}
                          </p>

                        )}


                        {disease.status && (

                          <p>
                            {disease.status}
                          </p>

                        )}


                        {/* DISEASE ALTERNATIVES */}

                        {Array.isArray(
                          disease.top_predictions
                        ) &&
                        disease.top_predictions.length > 0 && (

                          <div className="alternatives">

                            <h5>
                              Other possible predictions
                            </h5>


                            <ul>

                              {disease.top_predictions
                                .slice(0, 5)
                                .map(
                                  (
                                    item,
                                    i
                                  ) => (

                                    <li
                                      key={i}
                                    >

                                      {item.name ||
                                        item.class_name ||
                                        item.label ||
                                        "Unknown"}

                                      {" — "}

                                      {item.confidence_percent ??
                                        item.confidence ??
                                        0}
                                      %

                                    </li>

                                  )
                                )}

                            </ul>

                          </div>

                        )}

                      </div>


                      {/* =================================================
                          NUTRIENT
                      ================================================= */}

                      {nutrient &&
                        nutrient.name &&
                        nutrient.name !==
                          "Not evaluated" && (

                        <div className="prediction-card">

                          <h4>
                            🧪 Nutrient Assessment
                          </h4>


                          <h2>

                            {nutrient.name}

                          </h2>


                          <p>

                            Confidence:
                            {" "}

                            <strong>

                              {nutrient.confidence_percent ??
                                nutrient.confidence ??
                                0}
                              %

                            </strong>

                          </p>


                          {nutrient.confidence_level && (

                            <p>

                              Level:
                              {" "}

                              <strong>
                                {nutrient.confidence_level}
                              </strong>

                            </p>

                          )}


                          <p>

                            {nutrient.message ||
                              nutrient.status ||
                              "Nutrient assessment completed."}

                          </p>


                          {/* NUTRIENT ALTERNATIVES */}

                          {Array.isArray(
                            nutrient.top_predictions
                          ) &&
                          nutrient.top_predictions.length > 0 && (

                            <div className="alternatives">

                              <h5>
                                Other possible predictions
                              </h5>


                              <ul>

                                {nutrient.top_predictions
                                  .slice(0, 5)
                                  .map(
                                    (
                                      item,
                                      i
                                    ) => (

                                      <li
                                        key={i}
                                      >

                                        {item.name ||
                                          item.class_name ||
                                          item.label ||
                                          "Unknown"}

                                        {" — "}

                                        {item.confidence_percent ??
                                          item.confidence ??
                                          0}
                                        %

                                      </li>

                                    )
                                  )}

                              </ul>

                            </div>

                          )}

                        </div>

                      )}


                      {/* =================================================
                          OVERALL
                      ================================================= */}

                      <div
                        className={
                          `status-box ${
                            getStatusClass(
                              prediction.status ||
                              overall.status
                            )
                          }`
                        }
                      >

                        <h4>
                          Overall Assessment
                        </h4>


                        <h2>

                          {prediction.status ||
                            overall.status ||
                            "Requires verification"}

                        </h2>


                        <p>

                          Overall confidence:
                          {" "}

                          <strong>

                            {prediction.overall_confidence_percent ??
                              overall.confidence_percent ??
                              overall.confidence ??
                              0}
                            %

                          </strong>

                        </p>


                        {(
                          prediction.verification_required ??
                          overall.verification_required
                        ) && (

                          <p>

                            ⚠️ Independent verification
                            is recommended before
                            treatment decisions.

                          </p>

                        )}

                      </div>


                      {/* =================================================
                          CROP-SPECIFIC RECOMMENDATIONS
                      ================================================= */}

                      <div className="recommendations">

                        <h4>
                          🌱 Crop-Specific Solutions & Recommendations
                        </h4>


                        {recommendations.length > 0 ? (

                          <>

                            <p className="recommendation-intro">

                              Based on the detected{" "}

                              <strong>
                                {disease.name ||
                                  "health condition"}
                              </strong>


                              {nutrient.name &&
                                nutrient.name !==
                                  "Not evaluated" && (

                                <>

                                  {" "}
                                  and possible{" "}

                                  <strong>
                                    {nutrient.name}
                                  </strong>

                                </>

                              )}

                              , the following
                              actions are recommended
                              for the selected crop.

                            </p>


                            <ol>

                              {recommendations.map(
                                (
                                  recommendation,
                                  rIndex
                                ) => (

                                  <li
                                    key={
                                      rIndex
                                    }
                                  >

                                    {recommendation}

                                  </li>

                                )
                              )}

                            </ol>

                          </>

                        ) : (

                          <p>

                            No crop-specific solution
                            was returned for this result.
                            Verify the condition before
                            applying treatment.

                          </p>

                        )}


                        {/* =================================================
                            ENVIRONMENTAL CONTEXT
                        ================================================= */}

                        <div className="environment-summary">

                          <h5>
                            🌦️ Environmental context used
                          </h5>


                          <p>

                            🌡️ Temperature:
                            {" "}

                            {weather?.temperature ??
                              prediction.context?.temperature ??
                              "--"}
                            °C

                          </p>


                          <p>

                            💧 Humidity:
                            {" "}

                            {weather?.humidity ??
                              prediction.context?.humidity ??
                              "--"}
                            %

                          </p>


                          <p>

                            🌧️ Rainfall:
                            {" "}

                            {weather?.rainfall_status ??
                              weather?.recent_rainfall ??
                              prediction.context?.recent_rainfall ??
                              "--"}

                          </p>


                          <p>

                            ☁️ Condition:
                            {" "}

                            {weather?.condition ??
                              weather?.weather_condition ??
                              prediction.context?.weather_condition ??
                              "--"}

                          </p>


                          <p>

                            💨 Wind:
                            {" "}

                            {weather?.wind_speed ??
                              prediction.context?.wind_speed ??
                              "--"}
                            km/h

                          </p>


                          <p>

                            📍 Location:
                            {" "}

                            {location
                              ? `${Number(
                                  location.latitude
                                ).toFixed(5)}, ${Number(
                                  location.longitude
                                ).toFixed(5)}`
                              : "--"}

                          </p>


                          <p>

                            🌱 Growth stage:
                            {" "}
                            Vegetative

                          </p>


                          <p>

                            🧪 Soil condition:
                            {" "}
                            Moist

                          </p>


                          <p>

                            🧪 Soil pH:
                            {" "}
                            6.5

                          </p>


                          <p>

                            💧 Irrigation:
                            {" "}
                            Normal

                          </p>

                        </div>

                      </div>

                    </div>

                  );

                }
              )}

            </div>

          )}


          {/* =====================================================
              GENERATE DIGITAL TWIN
          ===================================================== */}

          {results.length > 0 &&
            digitalTwinReady &&
            !digitalTwin && (

            <section className="card digital-twin-generate-card">

              <div className="section-title">

                <div className="section-icon">
                  🌐
                </div>

                <div>

                  <h3>
                    Build Your Farm Digital Twin
                  </h3>

                  <p>
                    Crop health analysis is complete. Review the
                    disease, nutrient, weather and recommendation
                    results above, then create the virtual farm
                    representation when you are ready.
                  </p>

                </div>

              </div>

              <button
                className="predict-btn"
                type="button"
                onClick={() =>
                  createDigitalTwin(results)
                }
                disabled={
                  !results.length ||
                  !location ||
                  !weather
                }
              >
                🌐 Generate Digital Twin
              </button>

            </section>

          )}


          {/* =====================================================
              DIGITAL TWIN
          ===================================================== */}

          {digitalTwin && (
  <section className="digital-twin-card">

    {/* DIGITAL TWIN HEADER */}
    <div className="digital-twin-header">

      <div className="digital-twin-title">

        <div className="digital-twin-icon">
          🌐
        </div>

        <div>
          <h2>Farm Digital Twin</h2>

          <p>
            Virtual representation of the analyzed field
          </p>
        </div>

      </div>

      <div className="plant-count-badge">
        🌱 {digitalTwin.plantCount} Plants
      </div>

    </div>


    {/* FIELD SUMMARY */}
    <div className="twin-summary-grid">

      <div className="twin-summary-item">
        <span>Field ID</span>
        <strong>
          {digitalTwin.fieldId}
        </strong>
      </div>

      <div className="twin-summary-item">
        <span>Crop</span>
        <strong>
          {digitalTwin.crop}
        </strong>
      </div>

      <div className="twin-summary-item">
        <span>Plants Analyzed</span>
        <strong>
          {digitalTwin.plantCount}
        </strong>
      </div>

      <div className="twin-summary-item">
        <span>Created</span>
        <strong>
          {digitalTwin.createdAt
            ? new Date(
                digitalTwin.createdAt
              ).toLocaleDateString()
            : "--"}
        </strong>
      </div>

    </div>


    {/* ENVIRONMENT */}
    <div className="twin-section">

      <div className="twin-section-heading">
        <span>📍</span>
        <h3>Location & Environment</h3>
      </div>


      <div className="twin-environment-grid">

        <div className="environment-item">
          <span>Latitude</span>

          <strong>
            {digitalTwin.location
              ? Number(
                  digitalTwin.location.latitude
                ).toFixed(5)
              : "--"}
          </strong>
        </div>


        <div className="environment-item">
          <span>Longitude</span>

          <strong>
            {digitalTwin.location
              ? Number(
                  digitalTwin.location.longitude
                ).toFixed(5)
              : "--"}
          </strong>
        </div>


        <div className="environment-item">
          <span>Temperature</span>

          <strong>
            {digitalTwin.weather?.temperature ?? "--"} °C
          </strong>
        </div>


        <div className="environment-item">
          <span>Humidity</span>

          <strong>
            {digitalTwin.weather?.humidity ?? "--"} %
          </strong>
        </div>


        <div className="environment-item">
          <span>Rainfall</span>

          <strong>
            {digitalTwin.weather?.rainfall_status ?? "--"}
          </strong>
        </div>


        <div className="environment-item">
          <span>Condition</span>

          <strong>
            {digitalTwin.weather?.condition ?? "--"}
          </strong>
        </div>

      </div>

    </div>


    {/* CROP MAP */}
    <div className="twin-section">

      <div className="twin-section-heading">
        <span>🌾</span>
        <h3>Crop Map</h3>

        <span className="map-description">
          Individual plant health status
        </span>
      </div>


      <div className="twin-plant-grid">

        {digitalTwin.plants?.map(
          (plant) => {

            const statusClass =
              getStatusClass(
                plant.status
              );

            return (
              <div
                className={`twin-plant-card ${statusClass}`}
                key={plant.id}
              >

                {/* PLANT HEADER */}
                <div className="plant-card-header">

                  <div className="plant-number">
                    {plant.id}
                  </div>

                  <span className="plant-label">
                    Plant {plant.id}
                  </span>

                </div>


                {/* CROP */}
                <h4>
                  {plant.crop}
                </h4>


                {/* DISEASE */}
                <div className="plant-health-row">

                  <span>
                    🌿 Disease / Health
                  </span>

                  <strong>
                    {plant.disease || "Unknown"}
                  </strong>

                </div>


                <div className="confidence-row">

                  <span>
                    Confidence
                  </span>

                  <strong>
                    {safeNumber(
                      plant.diseaseConfidence
                    ).toFixed(2)}%
                  </strong>

                </div>


                <div className="mini-progress">

                  <div
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(
                          0,
                          safeNumber(
                            plant.diseaseConfidence
                          )
                        )
                      )}%`
                    }}
                  />

                </div>


                {/* NUTRIENT */}
                <div className="plant-health-row nutrient-row">

                  <span>
                    🧪 Nutrient
                  </span>

                  <strong>
                    {plant.nutrient ||
                      "Not evaluated"}
                  </strong>

                </div>


                <div className="confidence-row">

                  <span>
                    Nutrient confidence
                  </span>

                  <strong>
                    {safeNumber(
                      plant.nutrientConfidence
                    ).toFixed(2)}%
                  </strong>

                </div>


                {/* OVERALL */}
                <div className="plant-overall">

                  <span>
                    Overall confidence
                  </span>

                  <strong>
                    {safeNumber(
                      plant.overallConfidence
                    ).toFixed(2)}%
                  </strong>

                </div>


                {/* STATUS */}
                <div
                  className={`plant-status ${statusClass}`}
                >
                  {plant.status ||
                    "Requires verification"}
                </div>


              </div>
            );
          }
        )}

      </div>

    </div>


    {/* EXPORT */}
    <div className="twin-actions">

      <button
        className="digital-twin-export"
        onClick={
          downloadDigitalTwin
        }
      >
        ↓ Export Digital Twin JSON
      </button>

    </div>

  </section>
)}


        </section>

      </main>

    </div>

  );
}


export default App;