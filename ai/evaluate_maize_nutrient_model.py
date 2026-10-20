import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_CSV = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "nutrient_test.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ai",
    "models",
    "maize_nutrient_mobilenetv3.keras"
)

CLASS_PATH = os.path.join(
    PROJECT_ROOT,
    "ai",
    "models",
    "maize_nutrient_classes.json"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "ai",
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "maize_nutrient_classification_report.csv"
)

CONFUSION_PATH = os.path.join(
    RESULTS_DIR,
    "maize_nutrient_confusion_matrix.csv"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "maize_nutrient_test_metrics.json"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("MAIZE NUTRIENT MODEL EVALUATION")
print("=" * 70)


# ================================================================
# CHECK FILES
# ================================================================

print("\nChecking required files...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

print("Model found.")

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(
        f"Class mapping not found:\n{CLASS_PATH}"
    )

print("Class mapping found.")

if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_CSV}"
    )

print("Test dataset found.")


# ================================================================
# LOAD CLASS MAPPING
# ================================================================

print("\nLoading class mapping...")

with open(CLASS_PATH, "r") as f:
    class_mapping = json.load(f)

# Handle either:
# {"0": "Healthy", ...}
# or
# {"Healthy": 0, ...}

if all(str(k).isdigit() for k in class_mapping.keys()):
    class_names = [
        class_mapping[str(i)]
        for i in range(len(class_mapping))
    ]
else:
    class_names = [
        name
        for name, index in sorted(
            class_mapping.items(),
            key=lambda x: x[1]
        )
    ]

NUM_CLASSES = len(class_names)

print(f"Number of classes: {NUM_CLASSES}")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ================================================================
# LOAD TEST DATASET
# ================================================================

print("\nLoading test dataset...")

df = pd.read_csv(TEST_CSV)

# Only evaluate maize
df = df[df["crop"].astype(str).str.lower() == "maize"].copy()

print(f"\nTotal Maize test images: {len(df)}")

if len(df) == 0:
    raise ValueError("No Maize images found in test dataset.")

print("\nTest class distribution:")
print(df["label"].value_counts())


# ================================================================
# CHECK IMAGE PATHS
# ================================================================

print("\nChecking image paths...")

missing = []

for path in df["image_path"]:
    if not os.path.exists(path):
        missing.append(path)

print(f"Missing test images: {len(missing)}")

if missing:
    print("\nFirst missing paths:")
    for path in missing[:10]:
        print(path)

    raise FileNotFoundError(
        "Some test images are missing."
    )

print("All Maize test image paths are valid.")


# ================================================================
# LABEL ENCODING
# ================================================================

label_to_index = {
    name: i
    for i, name in enumerate(class_names)
}

df["label_index"] = df["label"].map(label_to_index)

if df["label_index"].isna().any():

    unknown_labels = df[
        df["label_index"].isna()
    ]["label"].unique()

    raise ValueError(
        f"Unknown labels found: {unknown_labels}"
    )

df["label_index"] = df["label_index"].astype(int)


# ================================================================
# IMAGE LOADING FUNCTION
# ================================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(image, tf.float32)

    # MobileNetV3 preprocessing
    image = tf.keras.applications.mobilenet_v3.preprocess_input(
        image
    )

    return image, label


# ================================================================
# CREATE TEST PIPELINE
# ================================================================

print("\nPreparing Maize test pipeline...")

test_paths = df["image_path"].values
test_labels = df["label_index"].values

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ================================================================
# LOAD MODEL
# ================================================================

print("\nLoading trained Maize model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

print("\nModel information:")
print("Input shape: ", model.input_shape)
print("Output shape:", model.output_shape)


# ================================================================
# GENERATE PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("GENERATING MAIZE TEST PREDICTIONS")
print("=" * 70)

predictions = model.predict(
    test_dataset,
    verbose=1
)

print(
    f"\nPrediction shape: {predictions.shape}"
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_labels


# ================================================================
# OVERALL METRICS
# ================================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ================================================================
# PRINT RESULTS
# ================================================================

print("\n" + "=" * 70)
print("OVERALL MAIZE TEST RESULTS")
print("=" * 70)

print(f"\nTest Images:        {len(y_true)}")
print(f"Test Accuracy:      {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro Precision:    {macro_precision:.4f}")
print(f"Macro Recall:       {macro_recall:.4f}")
print(f"Macro F1-Score:     {macro_f1:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall:    {weighted_recall:.4f}")
print(f"Weighted F1-Score:  {weighted_f1:.4f}")


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 70)
print("PER-CLASS MAIZE RESULTS")
print("=" * 70)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0
    )
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm)


# ================================================================
# SAVE CLASSIFICATION REPORT
# ================================================================

report_df.to_csv(
    REPORT_PATH,
    index=True
)


# ================================================================
# SAVE CONFUSION MATRIX
# ================================================================

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(
    CONFUSION_PATH
)


# ================================================================
# SAVE METRICS
# ================================================================

metrics = {
    "model": "MobileNetV3Small",
    "dataset": "Maize Nutrient",
    "test_images": int(len(y_true)),
    "num_classes": int(NUM_CLASSES),

    "accuracy": float(accuracy),

    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),

    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1": float(weighted_f1),

    "classes": class_names
}

with open(
    METRICS_PATH,
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


# ================================================================
# FINAL MESSAGE
# ================================================================

print("\n" + "=" * 70)
print("MAIZE NUTRIENT EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nResults saved to:")

print("\nClassification report:")
print(REPORT_PATH)

print("\nConfusion matrix:")
print(CONFUSION_PATH)

print("\nMetrics:")
print(METRICS_PATH)

print("\n" + "=" * 70)