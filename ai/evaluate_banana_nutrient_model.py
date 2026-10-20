"""
evaluate_banana_nutrient_model.py

Evaluates the Banana Nutrient MobileNetV3 model
on the untouched Banana test dataset.
"""

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


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ai",
    "models"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "ai",
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)


TEST_CSV = os.path.join(
    DATASET_DIR,
    "nutrient_test.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "banana_nutrient_mobilenetv3.keras"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "banana_nutrient_classes.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("BANANA NUTRIENT MODEL EVALUATION")
print("=" * 70)

test_df = pd.read_csv(TEST_CSV)

test_df = test_df[
    test_df["crop"] == "Banana"
].copy()

print(
    f"\nBanana test images: {len(test_df)}"
)

print("\nTest class distribution:")

print(
    test_df["label"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(
    CLASS_PATH,
    "r",
    encoding="utf-8"
) as file:

    index_to_class = json.load(file)


index_to_class = {
    int(k): v
    for k, v in index_to_class.items()
}

class_to_index = {
    v: k
    for k, v in index_to_class.items()
}


# ============================================================
# CONVERT LABELS
# ============================================================

test_df["class_id"] = test_df[
    "label"
].map(class_to_index)


# ============================================================
# CHECK MISSING IMAGES
# ============================================================

missing = []

for path in test_df["image_path"]:

    if not os.path.exists(path):

        missing.append(path)


print(
    f"\nMissing test images: {len(missing)}"
)

if len(missing) > 0:

    print("\nFirst missing paths:")

    for path in missing[:10]:

        print(path)

    raise FileNotFoundError(
        "Some test images are missing."
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_df["image_path"].values,
        test_df["class_id"].values.astype(np.int32)
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


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    test_dataset,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

y_true = test_df[
    "class_id"
].values


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
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


# ============================================================
# PRINT METRICS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL TEST RESULTS")
print("=" * 70)

print(
    f"\nTest Images: {len(test_df)}"
)

print(
    f"Test Accuracy: {accuracy:.4f}"
    f" ({accuracy * 100:.2f}%)"
)

print(
    f"Macro Precision: {precision:.4f}"
)

print(
    f"Macro Recall: {recall:.4f}"
)

print(
    f"Macro F1-Score: {f1:.4f}"
)

print(
    f"Weighted Precision: {weighted_precision:.4f}"
)

print(
    f"Weighted Recall: {weighted_recall:.4f}"
)

print(
    f"Weighted F1-Score: {weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

target_names = [
    index_to_class[i]
    for i in range(len(index_to_class))
]

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(index_to_class))),
    target_names=target_names,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()


# ============================================================
# PRINT CLASS RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(index_to_class))),
        target_names=target_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(index_to_class)))
)

cm_df = pd.DataFrame(
    cm,
    index=target_names,
    columns=target_names
)


# ============================================================
# SAVE RESULTS
# ============================================================

report_path = os.path.join(
    RESULT_DIR,
    "banana_nutrient_classification_report.csv"
)

cm_path = os.path.join(
    RESULT_DIR,
    "banana_nutrient_confusion_matrix.csv"
)

metrics_path = os.path.join(
    RESULT_DIR,
    "banana_nutrient_test_metrics.json"
)


report_df.to_csv(
    report_path
)

cm_df.to_csv(
    cm_path
)


metrics = {

    "test_images": int(len(test_df)),

    "accuracy": float(accuracy),

    "macro_precision": float(precision),

    "macro_recall": float(recall),

    "macro_f1": float(f1),

    "weighted_precision": float(weighted_precision),

    "weighted_recall": float(weighted_recall),

    "weighted_f1": float(weighted_f1)

}


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(
    f"\nClassification report:\n{report_path}"
)

print(
    f"\nConfusion matrix:\n{cm_path}"
)

print(
    f"\nMetrics:\n{metrics_path}"
)