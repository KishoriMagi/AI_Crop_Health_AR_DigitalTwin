import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "ai" / "models" / "disease_mobilenetv3_v2.keras"
CLASS_FILE = BASE_DIR / "ai" / "models" / "disease_classes.json"
TEST_CSV = BASE_DIR / "datasets" / "disease_test_clean.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


print("=" * 70)
print("MOBILENETV3 DISEASE MODEL EVALUATION")
print("=" * 70)


# =========================================================
# 1. CHECK FILES
# =========================================================

print("\nChecking required files...")

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

if not CLASS_FILE.exists():
    raise FileNotFoundError(f"Class mapping not found: {CLASS_FILE}")

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

print("Model found.")
print("Class mapping found.")
print("Test dataset found.")


# =========================================================
# 2. LOAD CLASS MAPPING
# =========================================================

print("\nLoading class mapping...")

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

if all(str(k).isdigit() for k in class_mapping.keys()):
    index_to_class = {
        int(k): v for k, v in class_mapping.items()
    }
else:
    index_to_class = {
        int(v): k for k, v in class_mapping.items()
    }

num_classes = len(index_to_class)

print(f"Number of classes: {num_classes}")

for index in sorted(index_to_class):
    print(f"{index:2d} -> {index_to_class[index]}")


# =========================================================
# 3. LOAD TEST DATA
# =========================================================

print("\nLoading test dataset...")

df = pd.read_csv(TEST_CSV)

print(f"Test images: {len(df):,}")

print("\nCSV columns:")
print(list(df.columns))


# =========================================================
# 4. CREATE LABEL MAPPING
# =========================================================

class_to_index = {
    name: index
    for index, name in index_to_class.items()
}

df["label_index"] = df["condition"].map(class_to_index)

if df["label_index"].isna().any():

    unknown_labels = df.loc[
        df["label_index"].isna(),
        "condition"
    ].unique()

    raise ValueError(
        f"Unknown labels found: {unknown_labels}"
    )

df["label_index"] = df["label_index"].astype(int)


# =========================================================
# 5. CHECK IMAGE PATHS
# =========================================================

print("\nChecking test image paths...")

missing_count = 0

for path in df["image_path"]:

    if not os.path.isfile(path):

        if missing_count < 10:
            print("Missing:", path)

        missing_count += 1

if missing_count > 0:

    raise FileNotFoundError(
        f"{missing_count} image paths are invalid."
    )

print("All test image paths are valid.")


# =========================================================
# 6. IMAGE LOADING FUNCTION
# =========================================================

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
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# =========================================================
# 7. CREATE TEST DATASET
# =========================================================

print("\nPreparing test pipeline...")

test_paths = df["image_path"].values
test_labels = df["label_index"].values

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


# =========================================================
# 8. LOAD MODEL
# =========================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("Model loaded successfully.")


# =========================================================
# 9. GENERATE PREDICTIONS
# =========================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

predictions = model.predict(
    test_dataset,
    verbose=1
)

print("\nPrediction shape:", predictions.shape)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

true_labels = test_labels


# =========================================================
# 10. CALCULATE METRICS
# =========================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision_macro = precision_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)


# =========================================================
# 11. PRINT OVERALL RESULTS
# =========================================================

print("\n" + "=" * 70)
print("OVERALL TEST RESULTS")
print("=" * 70)

print(
    f"Test Accuracy       : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Macro Precision     : {precision_macro:.4f}"
)

print(
    f"Macro Recall        : {recall_macro:.4f}"
)

print(
    f"Macro F1-Score      : {f1_macro:.4f}"
)

print(
    f"Weighted Precision  : {precision_weighted:.4f}"
)

print(
    f"Weighted Recall     : {recall_weighted:.4f}"
)

print(
    f"Weighted F1-Score   : {f1_weighted:.4f}"
)


# =========================================================
# 12. CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 70)
print("PER-CLASS CLASSIFICATION REPORT")
print("=" * 70)

target_names = [
    index_to_class[i]
    for i in range(num_classes)
]

report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(num_classes)),
    target_names=target_names,
    zero_division=0
)

print(report)


# =========================================================
# 13. CONFUSION MATRIX
# =========================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(num_classes))
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm)


# =========================================================
# 14. SAVE RESULTS
# =========================================================

RESULTS_DIR = BASE_DIR / "ai" / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Classification report

report_dict = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(num_classes)),
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_file = RESULTS_DIR / "disease_classification_report_v2.csv"

report_df.to_csv(
    report_file
)


# Confusion matrix

cm_df = pd.DataFrame(
    cm,
    index=target_names,
    columns=target_names
)

cm_file = RESULTS_DIR / "disease_confusion_matrix_v2.csv"

cm_df.to_csv(
    cm_file
)


# Overall metrics

metrics = {
    "test_accuracy": float(accuracy),
    "macro_precision": float(precision_macro),
    "macro_recall": float(recall_macro),
    "macro_f1": float(f1_macro),
    "weighted_precision": float(precision_weighted),
    "weighted_recall": float(recall_weighted),
    "weighted_f1": float(f1_weighted)
}

metrics_file = RESULTS_DIR / "disease_test_metrics_v2.json"

with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# =========================================================
# 15. COMPLETION
# =========================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nResults saved to:")

print(
    f"\nClassification report:"
    f"\n{report_file}"
)

print(
    f"\nConfusion matrix:"
    f"\n{cm_file}"
)

print(
    f"\nOverall metrics:"
    f"\n{metrics_file}"
)

print("\n" + "=" * 70)