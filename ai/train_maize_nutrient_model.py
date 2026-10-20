import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_CSV = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "nutrient_train.csv"
)

VAL_CSV = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "nutrient_val.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ai",
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "maize_nutrient_mobilenetv3.keras"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "maize_nutrient_classes.json"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

HEAD_EPOCHS = 20
FINE_TUNE_EPOCHS = 35

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

os.makedirs(MODEL_DIR, exist_ok=True)


print("=" * 70)
print("MAIZE NUTRIENT MODEL - IMPROVED TRAINING")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(
        f"Training CSV not found:\n{TRAIN_CSV}"
    )

if not os.path.exists(VAL_CSV):
    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )

print("Training dataset found.")
print("Validation dataset found.")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)


# Keep only maize
train_df = train_df[
    train_df["crop"].astype(str).str.lower() == "maize"
].copy()

val_df = val_df[
    val_df["crop"].astype(str).str.lower() == "maize"
].copy()

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)


print(f"\nTraining images   : {len(train_df)}")
print(f"Validation images : {len(val_df)}")


if len(train_df) == 0:
    raise ValueError("No maize training images found.")

if len(val_df) == 0:
    raise ValueError("No maize validation images found.")


# ============================================================
# CLASS MAPPING
# ============================================================

classes = sorted(
    train_df["label"].unique().tolist()
)

print(f"\nNumber of classes: {len(classes)}")

for i, cls in enumerate(classes):
    print(f"{i}: {cls}")


class_to_index = {
    cls: i
    for i, cls in enumerate(classes)
}

index_to_class = {
    str(i): cls
    for cls, i in class_to_index.items()
}


with open(
    CLASS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        index_to_class,
        f,
        indent=4
    )


# ============================================================
# LABEL ENCODING
# ============================================================

train_df["class_id"] = train_df["label"].map(
    class_to_index
)

val_df["class_id"] = val_df["label"].map(
    class_to_index
)


if train_df["class_id"].isna().any():
    raise ValueError(
        "Unknown class found in training data."
    )

if val_df["class_id"].isna().any():
    raise ValueError(
        "Unknown class found in validation data."
    )


# ============================================================
# DATA DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING DISTRIBUTION")
print("=" * 70)

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 70)
print("VALIDATION DISTRIBUTION")
print("=" * 70)

print(
    val_df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# CHECK PATHS
# ============================================================

print("\nChecking image paths...")

missing_train = [
    p for p in train_df["image_path"]
    if not os.path.exists(p)
]

missing_val = [
    p for p in val_df["image_path"]
    if not os.path.exists(p)
]


print(
    f"Missing training images   : {len(missing_train)}"
)

print(
    f"Missing validation images : {len(missing_val)}"
)


if missing_train:
    print("\nFirst missing training image:")
    print(missing_train[0])
    raise FileNotFoundError(
        "Training image paths are invalid."
    )

if missing_val:
    print("\nFirst missing validation image:")
    print(missing_val[0])
    raise FileNotFoundError(
        "Validation image paths are invalid."
    )

print("All image paths are valid.")


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

y_train = (
    train_df["class_id"]
    .astype(int)
    .values
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(classes)),
    y=y_train
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}


print("\nClass weights:")

for i, cls in enumerate(classes):

    print(
        f"{cls:<25} : "
        f"{class_weights[i]:.3f}"
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
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = preprocess_input(
        image
    )

    return image, label


# ============================================================
# STRONGER LEAF AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.15
        ),

        layers.RandomZoom(
            height_factor=(-0.10, 0.15),
            width_factor=(-0.10, 0.15)
        ),

        layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08
        ),

        layers.RandomContrast(
            0.15
        ),

        layers.RandomBrightness(
            0.10
        ),

    ],
    name="maize_leaf_augmentation"
)


def augment_image(image, label):

    image = augmentation(
        image,
        training=True
    )

    return image, label


# ============================================================
# CREATE DATASETS
# ============================================================

print("\nCreating TensorFlow datasets...")


train_paths = train_df["image_path"].values
train_labels = (
    train_df["class_id"]
    .astype(int)
    .values
)

val_paths = val_df["image_path"].values
val_labels = (
    val_df["class_id"]
    .astype(int)
    .values
)


train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)


train_ds = train_ds.shuffle(
    buffer_size=len(train_df),
    seed=SEED,
    reshuffle_each_iteration=True
)


train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


train_ds = train_ds.map(
    augment_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)


val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)


val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(
    BATCH_SIZE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding MobileNetV3Small...")

base_model = MobileNetV3Small(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)


# ============================================================
# STAGE 1
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1 - CLASSIFICATION HEAD")
print("=" * 70)


base_model.trainable = False


inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    name="image"
)


x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(
    0.40
)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    len(classes),
    activation="softmax",
    name="nutrient_prediction"
)(x)


model = models.Model(
    inputs,
    outputs,
    name="Maize_Nutrient_MobileNetV3Small"
)


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=5e-4
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print("Model created successfully.")


# ============================================================
# STAGE 1 CALLBACKS
# ============================================================

stage1_checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


stage1_early = EarlyStopping(

    monitor="val_accuracy",

    mode="max",

    patience=6,

    restore_best_weights=True,

    verbose=1
)


stage1_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=3,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# STAGE 1 TRAINING
# ============================================================

history1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=HEAD_EPOCHS,

    class_weight=class_weights,

    callbacks=[
        stage1_checkpoint,
        stage1_early,
        stage1_lr
    ],

    verbose=1
)


best_stage1 = max(
    history1.history["val_accuracy"]
)


print("\nBest Stage 1 validation accuracy:")
print(
    f"{best_stage1 * 100:.2f}%"
)


# ============================================================
# LOAD BEST STAGE 1 MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)

base_model = model.layers[1]


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2 - FINE TUNING")
print("=" * 70)


base_model.trainable = True


# MobileNetV3Small has relatively few layers.
# Fine-tune only the upper portion.

fine_tune_from = int(
    len(base_model.layers) * 0.65
)


for layer in base_model.layers:

    layer.trainable = False


for layer in base_model.layers[
    fine_tune_from:
]:

    layer.trainable = True


# Keep BatchNormalization frozen.
for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


trainable_count = sum(
    layer.trainable
    for layer in base_model.layers
)


print(
    f"Trainable backbone layers: "
    f"{trainable_count}"
)


# ============================================================
# RECOMPILE
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
# STAGE 2 CALLBACKS
# ============================================================

stage2_checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


stage2_early = EarlyStopping(

    monitor="val_accuracy",

    mode="max",

    patience=8,

    restore_best_weights=True,

    verbose=1
)


stage2_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=3,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# STAGE 2 TRAINING
# ============================================================

history2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=[
        stage2_checkpoint,
        stage2_early,
        stage2_lr
    ],

    verbose=1
)


best_stage2 = max(
    history2.history["val_accuracy"]
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MAIZE NUTRIENT TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nBest Stage 1 validation accuracy : "
    f"{best_stage1 * 100:.2f}%"
)

print(
    f"Best Stage 2 validation accuracy : "
    f"{best_stage2 * 100:.2f}%"
)

print("\nModel:")
print(MODEL_PATH)

print("\nClasses:")
print(CLASS_PATH)

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "\nThe test dataset was NOT used during training."
)

print(
    "\nNow evaluate the new model with:"
)

print(
    "python ai\\evaluate_maize_nutrient_model.py"
)

print(
    "\nPay particular attention to:"
)

print("1. Test Accuracy")
print("2. Macro F1")
print("3. Potassium Deficiency F1")
print("4. Confusion Matrix")

print("\n" + "=" * 70)