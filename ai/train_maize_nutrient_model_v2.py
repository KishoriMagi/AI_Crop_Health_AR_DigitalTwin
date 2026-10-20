import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
MODEL_DIR = os.path.join(PROJECT_ROOT, "ai", "models")

TRAIN_CSV = os.path.join(DATA_DIR, "nutrient_train.csv")
VAL_CSV = os.path.join(DATA_DIR, "nutrient_val.csv")

MODEL_OUTPUT = os.path.join(
    MODEL_DIR,
    "maize_nutrient_mobilenetv3_v2.keras"
)

CLASS_OUTPUT = os.path.join(
    MODEL_DIR,
    "maize_nutrient_classes_v2.json"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

SEED = 42

STAGE1_EPOCHS = 20
STAGE2_EPOCHS = 30

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("IMPROVED MAIZE NUTRIENT MODEL TRAINING")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Train CSV:")
print(TRAIN_CSV)

print()
print("Validation CSV:")
print(VAL_CSV)

print()
print("Output model:")
print(MODEL_OUTPUT)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(
        f"Training CSV not found:\n{TRAIN_CSV}"
    )

if not os.path.exists(VAL_CSV):
    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

# Keep only maize
train_df = train_df[
    train_df["crop"].astype(str).str.lower() == "maize"
].copy()

val_df = val_df[
    val_df["crop"].astype(str).str.lower() == "maize"
].copy()

print()
print("Training images:", len(train_df))
print("Validation images:", len(val_df))


# ============================================================
# REMOVE INVALID PATHS
# ============================================================

print()
print("Checking image paths...")

train_exists = train_df["image_path"].apply(os.path.exists)
val_exists = val_df["image_path"].apply(os.path.exists)

print(
    "Missing training images:",
    int((~train_exists).sum())
)

print(
    "Missing validation images:",
    int((~val_exists).sum())
)

train_df = train_df[train_exists].reset_index(drop=True)
val_df = val_df[val_exists].reset_index(drop=True)

print("Valid training images:", len(train_df))
print("Valid validation images:", len(val_df))


# ============================================================
# CLASS DEFINITIONS
# ============================================================

classes = sorted(
    train_df["label"].unique().tolist()
)

print()
print("=" * 70)
print("CLASSES")
print("=" * 70)

for i, cls in enumerate(classes):
    print(f"{i}: {cls}")

NUM_CLASSES = len(classes)

if NUM_CLASSES != 5:
    raise ValueError(
        f"Expected 5 maize nutrient classes, found {NUM_CLASSES}"
    )


class_to_index = {
    cls: i
    for i, cls in enumerate(classes)
}

train_df["class_index"] = train_df["label"].map(
    class_to_index
)

val_df["class_index"] = val_df["label"].map(
    class_to_index
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("TRAINING DISTRIBUTION")
print("=" * 70)

print(train_df["label"].value_counts())

print()
print("=" * 70)
print("VALIDATION DISTRIBUTION")
print("=" * 70)

print(val_df["label"].value_counts())


# ============================================================
# CLASS WEIGHTS
# ============================================================

print()
print("=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["class_index"].values
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

for i, weight in class_weights.items():
    print(
        f"{classes[i]:25s} -> {weight:.4f}"
    )


# ============================================================
# IMAGE PREPROCESSING
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
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    # MobileNetV3 preprocessing
    image = image / 127.5 - 1.0

    return image, label


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.12
        ),

        layers.RandomZoom(
            0.15
        ),

        layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08
        ),

        layers.RandomContrast(
            0.15
        ),
    ],
    name="maize_augmentation"
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_paths = train_df["image_path"].values
train_labels = train_df["class_index"].values

val_paths = val_df["image_path"].values
val_labels = val_df["class_index"].values


train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)


def train_preprocess(path, label):

    image, label = load_image(
        path,
        label
    )

    image = augmentation(
        image,
        training=True
    )

    return image, label


train_ds = (
    train_ds
    .shuffle(
        len(train_df),
        seed=SEED,
        reshuffle_each_iteration=True
    )
    .map(
        train_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


val_ds = (
    val_ds
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# BUILD MODEL
# ============================================================

print()
print("=" * 70)
print("BUILDING MOBILENETV3 MODEL")
print("=" * 70)

base_model = MobileNetV3Small(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)

# Stage 1: freeze pretrained backbone
base_model.trainable = False


inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(
    0.35
)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.25
)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ============================================================
# STAGE 1 COMPILATION
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)

print()
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_OUTPUT,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# STAGE 1 TRAINING
# ============================================================

print()
print("=" * 70)
print("STAGE 1: TRANSFER LEARNING")
print("=" * 70)

history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ============================================================
# STAGE 2: FINE TUNING
# ============================================================

print()
print("=" * 70)
print("STAGE 2: FINE TUNING")
print("=" * 70)

base_model.trainable = True

# Freeze early layers.
# Fine-tune only the later feature layers.
fine_tune_from = int(
    len(base_model.layers) * 0.70
)

for layer in base_model.layers[
    :fine_tune_from
]:
    layer.trainable = False

for layer in base_model.layers[
    fine_tune_from:
]:
    layer.trainable = True

print(
    "Fine-tuning from layer:",
    fine_tune_from
)

trainable_count = sum(
    1
    for layer in model.layers
    if layer.trainable
)

print(
    "Trainable layers:",
    trainable_count
)


# ============================================================
# RECOMPILE WITH LOW LR
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# FINE-TUNING CALLBACKS
# ============================================================

fine_checkpoint = ModelCheckpoint(
    MODEL_OUTPUT,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

fine_early_stopping = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

fine_reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-8,
    verbose=1
)


# ============================================================
# STAGE 2 TRAINING
# ============================================================

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE2_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        fine_checkpoint,
        fine_early_stopping,
        fine_reduce_lr
    ],
    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

best_model = tf.keras.models.load_model(
    MODEL_OUTPUT
)

print(
    "Best model loaded successfully."
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL VALIDATION EVALUATION")
print("=" * 70)

loss, accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print()
print(
    f"Validation Accuracy: {accuracy:.4f}"
)

print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

class_mapping = {
    str(i): classes[i]
    for i in range(NUM_CLASSES)
}

with open(
    CLASS_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_mapping,
        f,
        indent=4
    )


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print()
print("Best model:")
print(MODEL_OUTPUT)

print()
print("Class mapping:")
print(CLASS_OUTPUT)

print()
print(
    f"Final validation accuracy: "
    f"{accuracy * 100:.2f}%"
)

print()
print("=" * 70)