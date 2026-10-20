import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
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
    "banana_nutrient_mobilenetv3_v2.keras"
)

CLASS_MAPPING_PATH = os.path.join(
    PROJECT_ROOT,
    "ai",
    "models",
    "banana_nutrient_classes.json"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "ai",
    "results"
)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "banana_nutrient_classification_report_v2.csv"
)

CONFUSION_PATH = os.path.join(
    RESULTS_DIR,
    "banana_nutrient_confusion_matrix_v2.csv"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "banana_nutrient_test_metrics_v2.json"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 70)
print("BANANA NUTRIENT MODEL V2 EVALUATION")
print("=" * 70)
print()


# ================================================================
# CHECK REQUIRED FILES
# ================================================================

print("Checking required files...")

if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_CSV}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"V2 model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(CLASS_MAPPING_PATH):
    raise FileNotFoundError(
        f"Class mapping not found:\n{CLASS_MAPPING_PATH}"
    )

print("V2 model found.")
print("Class mapping found.")
print("Test dataset found.")


# ================================================================
# LOAD CLASS MAPPING
# ================================================================

print()
print("Loading class mapping...")

with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)


# Handle either:
# {"0": "Boron Deficiency", ...}
# or
# {"Boron Deficiency": 0, ...}

if all(str(k).isdigit() for k in class_mapping.keys()):

    class_indices = {
        int(k): v
        for k, v in class_mapping.items()
    }

else:

    class_indices = {
        int(v): k
        for k, v in class_mapping.items()
    }


class_names = [
    class_indices[i]
    for i in sorted(class_indices.keys())
]

NUM_CLASSES = len(class_names)

print(f"Number of classes: {NUM_CLASSES}")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ================================================================
# LOAD TEST DATASET
# ================================================================

print()
print("Loading test dataset...")

df = pd.read_csv(TEST_CSV)

# Keep only Banana records
banana_df = df[
    df["crop"].astype(str).str.lower() == "banana"
].copy()

print()
print(f"Total Banana test images: {len(banana_df)}")

print()
print("Test class distribution:")
print(banana_df["label"].value_counts().sort_index())


# ================================================================
# CHECK IMAGE PATHS
# ================================================================

print()
print("Checking image paths...")

missing_images = []

for path in banana_df["image_path"]:
    if not os.path.exists(path):
        missing_images.append(path)

print(f"Missing test images: {len(missing_images)}")

if len(missing_images) > 0:

    print()
    print("First missing images:")

    for path in missing_images[:10]:
        print(path)

    raise FileNotFoundError(
        f"{len(missing_images)} test images are missing."
    )

print("All Banana test image paths are valid.")


# ================================================================
# CREATE LABEL MAPPING
# ================================================================

label_to_index = {
    name: i
    for i, name in enumerate(class_names)
}


# ================================================================
# CONVERT LABELS TO INTEGER
# ================================================================

banana_df["label_index"] = banana_df["label"].map(
    label_to_index
)

unknown_labels = banana_df[
    banana_df["label_index"].isna()
]["label"].unique()

if len(unknown_labels) > 0:

    print()
    print("ERROR: Unknown labels found:")

    for label in unknown_labels:
        print(label)

    raise ValueError(
        "Some test labels are not present in the class mapping."
    )

banana_df["label_index"] = banana_df["label_index"].astype(int)


# ================================================================
# IMAGE LOADING FUNCTION
# ================================================================

def load_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    # MobileNetV3 expects pixel values in [0,255]
    # when include_preprocessing=True.
    image = tf.ensure_shape(
        image,
        [IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
    )

    return image, label


# ================================================================
# CREATE TEST PIPELINE
# ================================================================

print()
print("Preparing Banana test pipeline...")

test_paths = banana_df["image_path"].values
test_labels = banana_df["label_index"].values

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
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

print()
print("Loading V2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("V2 model loaded successfully.")


# ================================================================
# MODEL INFORMATION
# ================================================================

print()
print("Model information:")

print(f"Input shape:  {model.input_shape}")
print(f"Output shape: {model.output_shape}")


# ================================================================
# GENERATE PREDICTIONS
# ================================================================

print()
print("=" * 70)
print("GENERATING V2 TEST PREDICTIONS")
print("=" * 70)

predictions = model.predict(
    test_dataset,
    verbose=1
)

print()
print(f"Prediction shape: {predictions.shape}")


# ================================================================
# CONVERT PREDICTIONS TO CLASS INDICES
# ================================================================

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
# PRINT OVERALL RESULTS
# ================================================================

print()
print("=" * 70)
print("OVERALL V2 TEST RESULTS")
print("=" * 70)

print()
print(f"Test Images:       {len(y_true)}")
print(
    f"Test Accuracy:     {accuracy:.4f} ({accuracy * 100:.2f}%)"
)

print(
    f"Macro Precision:   {macro_precision:.4f}"
)

print(
    f"Macro Recall:      {macro_recall:.4f}"
)

print(
    f"Macro F1-Score:    {macro_f1:.4f}"
)

print(
    f"Weighted Precision:{weighted_precision:.4f}"
)

print(
    f"Weighted Recall:   {weighted_recall:.4f}"
)

print(
    f"Weighted F1-Score: {weighted_f1:.4f}"
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print()
print("=" * 70)
print("PER-CLASS V2 RESULTS")
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

print()
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

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

print()
print("Rows = Actual")
print("Columns = Predicted")
print()

print(cm)


# ================================================================
# CREATE RESULTS DIRECTORY
# ================================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ================================================================
# SAVE CLASSIFICATION REPORT
# ================================================================

report_df.to_csv(
    REPORT_PATH
)

# ================================================================
# SAVE CONFUSION MATRIX
# ================================================================

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.index.name = "Actual"

cm_df.to_csv(
    CONFUSION_PATH
)


# ================================================================
# SAVE OVERALL METRICS
# ================================================================

metrics = {
    "model": "MobileNetV3Small Banana Nutrient V2",

    "test_images": int(len(y_true)),

    "number_of_classes": int(NUM_CLASSES),

    "accuracy": float(accuracy),

    "macro_precision": float(macro_precision),

    "macro_recall": float(macro_recall),

    "macro_f1": float(macro_f1),

    "weighted_precision": float(weighted_precision),

    "weighted_recall": float(weighted_recall),

    "weighted_f1": float(weighted_f1),

    "class_names": class_names
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ================================================================
# FINAL OUTPUT
# ================================================================

print()
print("=" * 70)
print("V2 EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Results saved to:")

print()
print("Classification report:")
print(REPORT_PATH)

print()
print("Confusion matrix:")
print(CONFUSION_PATH)

print()
print("Metrics:")
print(METRICS_PATH)

print()
print("=" * 70)