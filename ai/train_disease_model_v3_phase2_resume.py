"""
DISEASE MODEL V3 — PHASE 2 RESUME SCRIPT
Resumes fine-tuning directly from disease_v3_phase1_best.keras.
Skips Phase 1 completely and fine-tunes the top 50% of EfficientNetV2-S.
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR    = Path(__file__).resolve().parent.parent

TRAIN_CSV   = BASE_DIR / "datasets" / "disease_train_clean.csv"
VAL_CSV     = BASE_DIR / "datasets" / "disease_val_clean.csv"
MODEL_DIR   = BASE_DIR / "ai" / "models"
CLASS_FILE  = MODEL_DIR / "disease_classes.json"

PHASE1_MODEL = MODEL_DIR / "disease_v3_phase1_best.keras"
FINAL_MODEL  = MODEL_DIR / "disease_efficientnetv2s_v3.keras"

IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32

PHASE2_EPOCHS = 15      # 15 epochs is plenty with early stopping
INITIAL_LR    = 3e-5    # Gentle learning rate for fine-tuning
LABEL_SMOOTHING = 0.10


print("=" * 70)
print("DISEASE MODEL V3  —  PHASE 2 FINE-TUNING RESUME")
print("=" * 70)


# =========================================================
# 1. CHECK REQUIRED FILES
# =========================================================

print("\nChecking files...")

if not PHASE1_MODEL.exists():
    raise FileNotFoundError(f"Phase 1 model not found: {PHASE1_MODEL}")

for f in [TRAIN_CSV, VAL_CSV, CLASS_FILE]:
    if not f.exists():
        raise FileNotFoundError(f"Required file not found: {f}")

print("Phase 1 model found.")
print("Datasets and class mapping found.")


# =========================================================
# 2. LOAD DATASETS & CLASS MAPPING
# =========================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

print(f"Training images  : {len(train_df):,}")
print(f"Validation images: {len(val_df):,}")

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    raw_classes = json.load(f)

index_to_class = {int(k): v for k, v in raw_classes.items()}
class_to_index = {v: k for k, v in index_to_class.items()}
num_classes    = len(index_to_class)

print(f"Total classes    : {num_classes}")

train_df["label"] = train_df["condition"].map(class_to_index).astype(int)
val_df["label"]   = val_df["condition"].map(class_to_index).astype(int)


# =========================================================
# 3. CLASS WEIGHTS
# =========================================================

print("\nCalculating class weights...")
y_train = train_df["label"].values
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_train,
)
class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}


# =========================================================
# 4. DATA PIPELINE WITH AUGMENTATION
# =========================================================

def load_image_raw(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label

augmentation_pipeline = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.10, 0.10),
        tf.keras.layers.RandomBrightness(0.20),
        tf.keras.layers.RandomContrast(0.15),
    ],
    name="augmentation",
)

def augment_image(image, label):
    image = augmentation_pipeline(image, training=True)
    image = tf.image.random_hue(image, max_delta=0.04)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label

print("\nBuilding tf.data pipelines...")

train_ds = (
    tf.data.Dataset
    .from_tensor_slices((train_df["image_path"].values, train_df["label"].values))
    .shuffle(len(train_df), reshuffle_each_iteration=True)
    .map(load_image_raw, num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment_image,   num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset
    .from_tensor_slices((val_df["image_path"].values, val_df["label"].values))
    .map(load_image_raw, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("Pipelines ready.")


# =========================================================
# 5. LOAD PHASE 1 MODEL & UNFREEZE TOP 50%
# =========================================================

print("\nLoading Phase 1 trained model...")

model = tf.keras.models.load_model(PHASE1_MODEL, compile=False)
print("Phase 1 model loaded.")

# Find efficientnet backbone
base_model = None
for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise ValueError("Could not find EfficientNet backbone inside loaded model.")

print(f"Backbone identified: {base_model.name}")

# Unfreeze backbone
base_model.trainable = True

total_backbone_layers = len(base_model.layers)
freeze_until          = int(total_backbone_layers * 0.50)

for i, layer in enumerate(base_model.layers):
    # Keep lower 50% frozen; keep ALL BatchNormalization layers frozen
    if i < freeze_until or isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Total backbone layers     : {total_backbone_layers}")
print(f"Trainable backbone layers : {trainable_count} (top 50%)")
print(f"Frozen backbone layers    : {total_backbone_layers - trainable_count}")


# =========================================================
# 6. COMPILE FOR PHASE 2
# =========================================================

def sparse_label_smooth_loss(y_true, y_pred):
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    return tf.keras.losses.categorical_crossentropy(
        y_true_oh, y_pred, label_smoothing=LABEL_SMOOTHING
    )

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss=sparse_label_smooth_loss,
    metrics=["accuracy"],
)

print(f"\nModel re-compiled with Adam (learning_rate={INITIAL_LR}) and Label Smoothing ({LABEL_SMOOTHING}).")


# =========================================================
# 7. CALLBACKS (Auto-save best checkpoints)
# =========================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    FINAL_MODEL,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.4,
    patience=2,
    min_lr=1e-7,
    verbose=1,
)


# =========================================================
# 8. START PHASE 2 FINE-TUNING
# =========================================================

print("\n" + "=" * 70)
print(f"STARTING PHASE 2 FINE-TUNING (Up to {PHASE2_EPOCHS} Epochs)")
print("Every epoch that improves validation accuracy will be auto-saved!")
print("=" * 70 + "\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1,
)

# Ensure final best model is saved
model.save(FINAL_MODEL)

best_acc = max(history.history["val_accuracy"])
best_ep  = np.argmax(history.history["val_accuracy"]) + 1

print("\n" + "=" * 70)
print("PHASE 2 FINE-TUNING COMPLETE!")
print(f"Best Validation Accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%) at Epoch {best_ep}")
print(f"Saved final model to: {FINAL_MODEL}")
print("=" * 70)
