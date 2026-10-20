"""
DISEASE MODEL V3 PHASE 1 EVALUATION
Evaluates disease_v3_phase1_best.keras on the test set.
Outputs: accuracy, macro/weighted precision/recall/F1,
         per-class report, confusion matrix.
"""

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
    f1_score,
)


BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_FILE  = BASE_DIR / "ai" / "models" / "disease_v3_phase1_best.keras"
CLASS_FILE  = BASE_DIR / "ai" / "models" / "disease_classes.json"
TEST_CSV    = BASE_DIR / "datasets" / "disease_test_clean.csv"
RESULTS_DIR = BASE_DIR / "ai" / "results"

IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32


print("=" * 70)
print("DISEASE MODEL V3 PHASE 1 EVALUATION — EfficientNetV2-S Head")
print("=" * 70)

# ---- files ----
for f in [MODEL_FILE, CLASS_FILE, TEST_CSV]:
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- class mapping ----
with open(CLASS_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

index_to_class = {int(k): v for k, v in raw.items()}
num_classes    = len(index_to_class)
class_to_index = {v: k for k, v in index_to_class.items()}

print(f"\nClasses ({num_classes}):")
for idx in sorted(index_to_class):
    print(f"  {idx:3d} -> {index_to_class[idx]}")

# ---- test data ----
df = pd.read_csv(TEST_CSV)
df["label_index"] = df["condition"].map(class_to_index).astype(int)
print(f"\nTest images: {len(df):,}")

# ---- tf.data pipeline ----
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label

test_ds = (
    tf.data.Dataset
    .from_tensor_slices((df["image_path"].values, df["label_index"].values))
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ---- load model (register custom loss / schedule if needed) ----
print("\nLoading Phase 1 model...")

def sparse_label_smooth_loss(y_true, y_pred):
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    return tf.keras.losses.categorical_crossentropy(y_true_oh, y_pred, label_smoothing=0.10)

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=1e-3, total_steps=1000, warmup_steps=100, min_lr=1e-7):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.total_steps = float(total_steps)
        self.warmup_steps = float(warmup_steps)
        self.min_lr = float(min_lr)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1.0 + tf.cos(np.pi * tf.minimum(progress, 1.0)))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
    def get_config(self):
        return {"initial_lr": self.initial_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps, "min_lr": self.min_lr}

model = tf.keras.models.load_model(
    MODEL_FILE,
    custom_objects={
        "sparse_label_smooth_loss": sparse_label_smooth_loss,
        "WarmupCosineDecay": WarmupCosineDecay,
    },
)
print("Model loaded successfully.")

# ---- predict ----
print("\nGenerating predictions on test set...")
preds  = model.predict(test_ds, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = df["label_index"].values

# ---- metrics ----
accuracy          = accuracy_score(y_true, y_pred)
macro_precision   = precision_score(y_true, y_pred, average="macro",    zero_division=0)
macro_recall      = recall_score(   y_true, y_pred, average="macro",    zero_division=0)
macro_f1          = f1_score(       y_true, y_pred, average="macro",    zero_division=0)
weighted_precision= precision_score(y_true, y_pred, average="weighted", zero_division=0)
weighted_recall   = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
weighted_f1       = f1_score(       y_true, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 70)
print("OVERALL TEST RESULTS — V3 PHASE 1")
print("=" * 70)
print(f"  Test Accuracy      : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print(f"  Macro Precision    : {macro_precision:.4f}")
print(f"  Macro Recall       : {macro_recall:.4f}")
print(f"  Macro F1-Score     : {macro_f1:.4f}")
print(f"  Weighted Precision : {weighted_precision:.4f}")
print(f"  Weighted Recall    : {weighted_recall:.4f}")
print(f"  Weighted F1-Score  : {weighted_f1:.4f}")

# ---- comparison with v2 ----
v2_accuracy = 0.8776
v2_macro_f1 = 0.7132
print("\n--- Comparison with V2 (MobileNetV3Small) ---")
print(f"  Accuracy delta vs V2 : {(accuracy - v2_accuracy) * 100:+.2f}%")
print(f"  Macro F1 delta vs V2 : {macro_f1 - v2_macro_f1:+.4f}")

# ---- per-class report ----
target_names = [index_to_class[i] for i in range(num_classes)]

print("\n" + "=" * 70)
print("PER-CLASS CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

# ---- confusion matrix ----
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

# ---- save results ----
report_dict = classification_report(
    y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
)
pd.DataFrame(report_dict).transpose().to_csv(
    RESULTS_DIR / "disease_classification_report_v3_phase1.csv"
)
pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
    RESULTS_DIR / "disease_confusion_matrix_v3_phase1.csv"
)

metrics = {
    "model": "EfficientNetV2-S Disease V3 Phase 1",
    "test_images": int(len(df)),
    "num_classes": num_classes,
    "test_accuracy":        round(float(accuracy),           4),
    "macro_precision":      round(float(macro_precision),    4),
    "macro_recall":         round(float(macro_recall),       4),
    "macro_f1":             round(float(macro_f1),           4),
    "weighted_precision":   round(float(weighted_precision), 4),
    "weighted_recall":      round(float(weighted_recall),    4),
    "weighted_f1":          round(float(weighted_f1),        4),
    "vs_v2_accuracy_delta": round(float(accuracy - v2_accuracy), 4),
    "vs_v2_macro_f1_delta": round(float(macro_f1 - v2_macro_f1), 4),
}

metrics_file = RESULTS_DIR / "disease_test_metrics_v3_phase1.json"
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print(f"\nResults saved to {RESULTS_DIR}/")
print(f"  disease_classification_report_v3_phase1.csv")
print(f"  disease_confusion_matrix_v3_phase1.csv")
print(f"  disease_test_metrics_v3_phase1.json")
print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
