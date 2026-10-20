"""
DISEASE MODEL V3 — EfficientNetV2-S with advanced training techniques.

Improvements over v2 (MobileNetV3Small fine-tune):
  1. Backbone  : EfficientNetV2-S (ImageNet pretrained) — much stronger
                 feature extractor than MobileNetV3Small.
  2. Augment   : Stronger pipeline (flip H+V, rotation ±15°, zoom ±15%,
                 brightness ±20%, contrast ±15%, hue ±5%, translation ±10%)
  3. Loss      : Label-smoothing cross-entropy (ε=0.1) — reduces
                 overconfidence on majority classes and improves macro F1.
  4. LR sched  : Cosine-decay with linear warmup (5 epochs). Better than
                 ReduceLROnPlateau for large datasets.
  5. Fine-tune : Two-phase — (a) train head only, (b) unfreeze top 50%.
  6. Metrics   : Tracks val accuracy + saves best checkpoint per phase.
  7. Class wts : sklearn balanced weights (same as v2).

Expected gains: +4 to +8% accuracy, +0.05 to +0.12 macro F1 vs v2.
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

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV   = BASE_DIR / "datasets" / "disease_train_clean.csv"
VAL_CSV     = BASE_DIR / "datasets" / "disease_val_clean.csv"
MODEL_DIR   = BASE_DIR / "ai" / "models"
CLASS_FILE  = MODEL_DIR / "disease_classes.json"

V3_MODEL    = MODEL_DIR / "disease_efficientnetv2s_v3.keras"
BEST_HEAD   = MODEL_DIR / "disease_v3_phase1_best.keras"

IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32

# Phase 1 — head-only training
PHASE1_EPOCHS   = 10
PHASE1_LR       = 1e-3

# Phase 2 — fine-tune top 50% of backbone
PHASE2_EPOCHS   = 20
PHASE2_LR       = 5e-5

# Label smoothing (reduces majority-class overconfidence)
LABEL_SMOOTHING = 0.10

# LR warmup (epochs)
WARMUP_EPOCHS   = 5


print("=" * 70)
print("DISEASE MODEL V3  —  EfficientNetV2-S  +  Advanced Training")
print("=" * 70)


# =========================================================
# 1. CHECK FILES
# =========================================================

print("\nChecking required files...")

for f in [TRAIN_CSV, VAL_CSV]:
    if not f.exists():
        raise FileNotFoundError(f"Required file not found: {f}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
print("All required files found.")


# =========================================================
# 2. LOAD DATA
# =========================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

print(f"Training images  : {len(train_df):,}")
print(f"Validation images: {len(val_df):,}")


# =========================================================
# 3. CLASS MAPPING
# =========================================================

classes        = sorted(train_df["condition"].unique())
num_classes    = len(classes)
class_to_index = {name: idx for idx, name in enumerate(classes)}
index_to_class = {idx: name for name, idx in class_to_index.items()}

print(f"\nNumber of classes: {num_classes}")
for idx in range(num_classes):
    print(f"  {idx:3d} -> {index_to_class[idx]}")

with open(CLASS_FILE, "w", encoding="utf-8") as f:
    json.dump(index_to_class, f, indent=4, ensure_ascii=False)

print(f"\nClass mapping saved to:\n  {CLASS_FILE}")


# =========================================================
# 4. LABEL ENCODING
# =========================================================

train_df["label"] = train_df["condition"].map(class_to_index).astype(int)
val_df["label"]   = val_df["condition"].map(class_to_index).astype(int)

if train_df["label"].isna().any():
    raise ValueError("Unknown class in training dataset.")
if val_df["label"].isna().any():
    raise ValueError("Unknown class in validation dataset.")


# =========================================================
# 5. CLASS WEIGHTS
# =========================================================

print("\nCalculating class weights...")

y_train = train_df["label"].values
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_train,
)
class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}

print("Class weights:")
for idx in range(num_classes):
    print(f"  {idx:3d} {index_to_class[idx]:40s} {class_weights[idx]:.3f}")


# =========================================================
# 6. IMAGE LOADING (no preprocessing — EfficientNetV2 does it internally)
# =========================================================

def load_image_raw(path, label):
    """Read, decode, resize — NO manual normalization (EfficientNetV2 handles it)."""
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


# =========================================================
# 7. STRONG DATA AUGMENTATION
# =========================================================

augmentation_pipeline = tf.keras.Sequential(
    [
        # Geometric
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.10, 0.10),

        # Photometric
        tf.keras.layers.RandomBrightness(0.20),
        tf.keras.layers.RandomContrast(0.15),
    ],
    name="strong_augmentation",
)

# Hue jitter (not in Sequential — applied manually in map)
def augment_with_hue(image, label):
    image = augmentation_pipeline(image, training=True)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


# =========================================================
# 8. tf.data PIPELINES
# =========================================================

print("\nBuilding tf.data pipelines...")

# --- Training ---
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_df["image_path"].values, train_df["label"].values)
)
train_ds = train_ds.shuffle(len(train_df), reshuffle_each_iteration=True)
train_ds = train_ds.map(load_image_raw,        num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(augment_with_hue,      num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# --- Validation ---
val_ds = tf.data.Dataset.from_tensor_slices(
    (val_df["image_path"].values, val_df["label"].values)
)
val_ds = val_ds.map(load_image_raw, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

print("Pipelines ready.")


# =========================================================
# 9. LR SCHEDULE — Cosine decay with linear warmup
# =========================================================

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup for `warmup_steps` then cosine decay to `min_lr`."""

    def __init__(self, initial_lr, total_steps, warmup_steps, min_lr=1e-7):
        super().__init__()
        self.initial_lr   = float(initial_lr)
        self.total_steps  = float(total_steps)
        self.warmup_steps = float(warmup_steps)
        self.min_lr       = float(min_lr)

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        progress  = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1.0 + tf.cos(np.pi * tf.minimum(progress, 1.0))
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr":   self.initial_lr,
            "total_steps":  self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr":       self.min_lr,
        }


# =========================================================
# 10. BUILD MODEL — EfficientNetV2-S backbone
# =========================================================

print("\nBuilding EfficientNetV2-S model...")

# Base: EfficientNetV2-S pretrained on ImageNet
# include_preprocessing=True means raw [0,255] input works directly
base_model = tf.keras.applications.EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, 3),
    include_preprocessing=True,   # handles normalization internally
)

# Freeze backbone for Phase 1
base_model.trainable = False

# Build classification head
inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
x      = base_model(inputs, training=False)
x      = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
x      = tf.keras.layers.BatchNormalization(name="head_bn")(x)
x      = tf.keras.layers.Dense(512, activation="relu", name="head_dense1")(x)
x      = tf.keras.layers.Dropout(0.40, name="dropout1")(x)
x      = tf.keras.layers.Dense(256, activation="relu", name="head_dense2")(x)
x      = tf.keras.layers.Dropout(0.30, name="dropout2")(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

model = tf.keras.Model(inputs, outputs, name="disease_v3_efficientnetv2s")

total_params = model.count_params()
print(f"Model: {model.name}")
print(f"Total parameters: {total_params:,}")
print(f"Backbone: {base_model.name} (frozen for Phase 1)")


# =========================================================
# 11. PHASE 1 — Train head only
# =========================================================

print("\n" + "=" * 70)
print("PHASE 1: Head-only training (backbone frozen)")
print("=" * 70)

steps_per_epoch_p1 = len(train_df) // BATCH_SIZE
total_steps_p1     = PHASE1_EPOCHS * steps_per_epoch_p1
warmup_steps_p1    = WARMUP_EPOCHS * steps_per_epoch_p1

lr_schedule_p1 = WarmupCosineDecay(
    initial_lr=PHASE1_LR,
    total_steps=total_steps_p1,
    warmup_steps=warmup_steps_p1,
    min_lr=1e-6,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_p1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        # Label smoothing via a custom wrapper — SparseCategoricalCrossentropy
        # doesn't accept label_smoothing directly; use reduction trick below.
    ),
    metrics=["accuracy"],
)

# Recompile with label smoothing using one-hot conversion
def sparse_label_smooth_loss(y_true, y_pred):
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    return tf.keras.losses.categorical_crossentropy(
        y_true_oh, y_pred, label_smoothing=LABEL_SMOOTHING
    )

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_p1),
    loss=sparse_label_smooth_loss,
    metrics=["accuracy"],
)

print(f"  Epochs          : {PHASE1_EPOCHS}")
print(f"  Learning rate   : {PHASE1_LR} (cosine decay, {WARMUP_EPOCHS}-epoch warmup)")
print(f"  Label smoothing : {LABEL_SMOOTHING}")

checkpoint_p1 = tf.keras.callbacks.ModelCheckpoint(
    BEST_HEAD,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

early_stop_p1 = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=4,
    restore_best_weights=True,
    verbose=1,
)

history_p1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint_p1, early_stop_p1],
    verbose=1,
)

best_p1_acc = max(history_p1.history["val_accuracy"])
print(f"\n✅ Phase 1 complete — best val accuracy: {best_p1_acc * 100:.2f}%")


# =========================================================
# 12. PHASE 2 — Fine-tune top 50% of EfficientNetV2-S
# =========================================================

print("\n" + "=" * 70)
print("PHASE 2: Fine-tuning top 50% of EfficientNetV2-S backbone")
print("=" * 70)

# Unfreeze backbone
base_model.trainable = True

total_backbone_layers = len(base_model.layers)
freeze_until          = int(total_backbone_layers * 0.50)

for i, layer in enumerate(base_model.layers):
    # Keep lower 50% frozen; keep ALL BatchNorm frozen (critical for fine-tuning)
    if i < freeze_until or isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Backbone layers total    : {total_backbone_layers}")
print(f"Backbone layers trainable: {trainable_count}")
print(f"Backbone layers frozen   : {total_backbone_layers - trainable_count}")

steps_per_epoch_p2 = len(train_df) // BATCH_SIZE
total_steps_p2     = PHASE2_EPOCHS * steps_per_epoch_p2
warmup_steps_p2    = 2 * steps_per_epoch_p2  # 2-epoch warmup for phase 2

lr_schedule_p2 = WarmupCosineDecay(
    initial_lr=PHASE2_LR,
    total_steps=total_steps_p2,
    warmup_steps=warmup_steps_p2,
    min_lr=1e-8,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_p2),
    loss=sparse_label_smooth_loss,
    metrics=["accuracy"],
)

print(f"  Epochs          : {PHASE2_EPOCHS}")
print(f"  Learning rate   : {PHASE2_LR} (cosine decay)")

checkpoint_p2 = tf.keras.callbacks.ModelCheckpoint(
    V3_MODEL,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

early_stop_p2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=6,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr_p2 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-8,
    verbose=1,
)

history_p2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint_p2, early_stop_p2, reduce_lr_p2],
    verbose=1,
)

best_p2_acc = max(history_p2.history["val_accuracy"])
print(f"\n✅ Phase 2 complete — best val accuracy: {best_p2_acc * 100:.2f}%")


# =========================================================
# 13. SAVE FINAL MODEL
# =========================================================

model.save(V3_MODEL)
print(f"\n✅ Final V3 model saved to:\n  {V3_MODEL}")
print(f"   Class mapping saved to:\n  {CLASS_FILE}")


# =========================================================
# 14. SUMMARY
# =========================================================

all_val_acc = history_p1.history["val_accuracy"] + history_p2.history["val_accuracy"]
best_overall = max(all_val_acc)

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"  Phase 1 best val accuracy : {best_p1_acc * 100:.2f}%")
print(f"  Phase 2 best val accuracy : {best_p2_acc * 100:.2f}%")
print(f"  Overall best val accuracy : {best_overall * 100:.2f}%")
print(f"  Model saved to            : {V3_MODEL}")
print("\nNext step → run evaluate_disease_model_v3.py to get full metrics.")
print("=" * 70)
a