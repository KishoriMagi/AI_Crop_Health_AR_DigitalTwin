import json
import pandas as pd
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "ai" / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = DATASET_DIR / "disease_train_clean.csv"
VAL_FILE = DATASET_DIR / "disease_val_clean.csv"

MODEL_FILE = MODEL_DIR / "disease_mobilenetv3.keras"
CLASS_FILE = MODEL_DIR / "disease_classes.json"


# =========================================================
# CONFIGURATION
# =========================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 15

RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


print("=" * 70)
print("MOBILENETV3 DISEASE CLASSIFICATION")
print("=" * 70)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

print(
    f"Training images:   {len(train_df):,}"
)

print(
    f"Validation images: {len(val_df):,}"
)


# =========================================================
# CREATE CLASS MAPPING
# =========================================================

classes = sorted(
    train_df["condition"].unique()
)

class_to_index = {
    name: index
    for index, name in enumerate(classes)
}

index_to_class = {
    index: name
    for name, index in class_to_index.items()
}

NUM_CLASSES = len(classes)


print(
    f"\nNumber of classes: {NUM_CLASSES}"
)

print("\nClasses:")

for index, name in index_to_class.items():
    print(
        f"{index:2} -> {name}"
    )


# Save class names for later prediction
with open(
    CLASS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        index_to_class,
        f,
        indent=4,
        ensure_ascii=False
    )


# =========================================================
# CONVERT LABELS
# =========================================================

train_df["class_id"] = train_df[
    "condition"
].map(class_to_index)

val_df["class_id"] = val_df[
    "condition"
].map(class_to_index)


# =========================================================
# IMAGE PATH
# =========================================================

def resolve_path(path):

    return path

# =========================================================
# LOAD IMAGE
# =========================================================

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


# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        )
    ],
    name="data_augmentation"
)


def augment(image, label):

    image = data_augmentation(
        image,
        training=True
    )

    return image, label


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess(image, label):

    image = tf.keras.applications.mobilenet_v3.preprocess_input(
        image
    )

    label = tf.one_hot(
        label,
        NUM_CLASSES
    )

    return image, label


# =========================================================
# CREATE TRAIN DATASET
# =========================================================

print("\nPreparing training pipeline...")

train_paths = train_df[
    "image_path"
].values

train_labels = train_df[
    "class_id"
].values


train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)


train_dataset = (
    train_dataset
    .shuffle(
        3000,
        seed=RANDOM_SEED
    )
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .map(
        augment,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(
        BATCH_SIZE
    )
    .prefetch(
        tf.data.AUTOTUNE
    )
)


# =========================================================
# CREATE VALIDATION DATASET
# =========================================================

print("Preparing validation pipeline...")

val_paths = val_df[
    "image_path"
].values

val_labels = val_df[
    "class_id"
].values


val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)


val_dataset = (
    val_dataset
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(
        BATCH_SIZE
    )
    .prefetch(
        tf.data.AUTOTUNE
    )
)


# =========================================================
# BUILD MOBILENETV3
# =========================================================

print("\nBuilding MobileNetV3Small...")

base_model = MobileNetV3Small(
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),

    include_top=False,

    weights="imagenet"
)


# Freeze pretrained layers
base_model.trainable = False


# =========================================================
# CLASSIFICATION HEAD
# =========================================================

inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


x = base_model(
    inputs,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dropout(
    0.30
)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print("\nModel created successfully.")

print(
    f"Total parameters: {model.count_params():,}"
)


# =========================================================
# CALLBACKS
# =========================================================

checkpoint = ModelCheckpoint(
    MODEL_FILE,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(
    monitor="val_loss",

    patience=4,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-6,

    verbose=1
)


# =========================================================
# TRAIN
# =========================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(
    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# =========================================================
# SAVE MODEL
# =========================================================

model.save(
    MODEL_FILE
)


print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nModel saved:"
)

print(
    MODEL_FILE
)

print(
    "\nClass mapping saved:"
)

print(
    CLASS_FILE
)


best_val_accuracy = max(
    history.history["val_accuracy"]
)

print(
    f"\nBest validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print("=" * 70)