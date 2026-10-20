import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)


# ==============================================================
# CONFIGURATION
# ==============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "ai",
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "banana_nutrient_mobilenetv3_v3.keras"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "banana_nutrient_classes_v3.json"
)

LOG_PATH = os.path.join(
    RESULT_DIR,
    "banana_nutrient_training_v3.csv"
)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 30

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


print("=" * 70)
print("BANANA NUTRIENT MODEL V3")
print("=" * 70)


# ==============================================================
# CHECK FILES
# ==============================================================

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


# ==============================================================
# LOAD DATA
# ==============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

# IMPORTANT:
# Only Banana records are used.
train_df = train_df[
    train_df["crop"].astype(str).str.lower() == "banana"
].copy()

val_df = val_df[
    val_df["crop"].astype(str).str.lower() == "banana"
].copy()

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Training images:   {len(train_df)}")
print(f"Validation images: {len(val_df)}")


# ==============================================================
# CLASS MAPPING
# ==============================================================

classes = sorted(train_df["label"].unique().tolist())

NUM_CLASSES = len(classes)

class_to_index = {
    name: i
    for i, name in enumerate(classes)
}

index_to_class = {
    str(i): name
    for i, name in enumerate(classes)
}

print(f"\nNumber of classes: {NUM_CLASSES}")
print("\nClasses:")

for i, name in enumerate(classes):
    print(f"{i}: {name}")


with open(CLASS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "classes": classes,
            "class_to_index": class_to_index,
            "index_to_class": index_to_class
        },
        f,
        indent=4
    )

print("\nClass mapping saved to:")
print(CLASS_PATH)


# ==============================================================
# CONVERT LABELS
# ==============================================================

train_df["class_id"] = train_df["label"].map(class_to_index)
val_df["class_id"] = val_df["label"].map(class_to_index)


# ==============================================================
# CHECK IMAGE PATHS
# ==============================================================

print("\nChecking image paths...")

missing_train = [
    p for p in train_df["image_path"]
    if not os.path.exists(p)
]

missing_val = [
    p for p in val_df["image_path"]
    if not os.path.exists(p)
]

print(f"Missing training images:   {len(missing_train)}")
print(f"Missing validation images: {len(missing_val)}")

if missing_train:
    print("\nFirst missing training image:")
    print(missing_train[0])
    raise FileNotFoundError("Missing training images detected.")

if missing_val:
    print("\nFirst missing validation image:")
    print(missing_val[0])
    raise FileNotFoundError("Missing validation images detected.")

print("All image paths are valid.")


# ==============================================================
# CLASS DISTRIBUTION
# ==============================================================

print("\nTraining class distribution:")

print(
    train_df["label"].value_counts()
    .sort_index()
)


# ==============================================================
# CLASS WEIGHTS
# ==============================================================

print("\nCalculating class weights...")

y_train = train_df["class_id"].values

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_train
)

class_weights = {
    i: float(weights[i])
    for i in range(NUM_CLASSES)
}

print("\nClass weights:")

for i, class_name in enumerate(classes):
    print(
        f"{i}: {class_name:<25} "
        f"{class_weights[i]:.3f}"
    )


# ==============================================================
# DATA PIPELINE
# ==============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    label = tf.cast(
        label,
        tf.int32
    )

    return image, label


print("\nCreating TensorFlow datasets...")


train_paths = train_df["image_path"].values
train_labels = train_df["class_id"].values

val_paths = val_df["image_path"].values
val_labels = val_df["class_id"].values


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
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

train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)


val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
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


# ==============================================================
# DATA AUGMENTATION
# ==============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            height_factor=(-0.15, 0.15),
            width_factor=(-0.15, 0.15)
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
        )
    ],
    name="banana_augmentation"
)


# ==============================================================
# MODEL
# ==============================================================

print("\nBuilding MobileNetV3Large...")

base_model = MobileNetV3Large(
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)

# Stage 1:
# Freeze entire pretrained backbone.
base_model.trainable = False


inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    name="banana_image"
)


x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v3.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(
    0.35
)(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="nutrient_prediction"
)(x)


model = models.Model(
    inputs,
    outputs,
    name="Banana_Nutrient_MobileNetV3Large_V3"
)


model.summary()


# ==============================================================
# LOSS
# ==============================================================

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False
)


# ==============================================================
# CALLBACKS
# ==============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
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

csv_logger = CSVLogger(
    LOG_PATH
)


# ==============================================================
# STAGE 1
# ==============================================================

print("\n")
print("=" * 70)
print("STAGE 1 - TRAINING CLASSIFICATION HEAD")
print("=" * 70)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss=loss_fn,
    metrics=[
        "accuracy"
    ]
)


history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        reduce_lr,
        csv_logger
    ],
    verbose=1
)


# ==============================================================
# STAGE 2 - FINE TUNING
# ==============================================================

print("\n")
print("=" * 70)
print("STAGE 2 - FINE-TUNING MOBILENETV3LARGE")
print("=" * 70)


# Unfreeze backbone
base_model.trainable = True


# Freeze lower layers and fine-tune upper layers.
fine_tune_from = 100

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

for layer in base_model.layers[fine_tune_from:]:
    layer.trainable = True


# Keep BatchNormalization layers frozen.
for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False


trainable_count = sum(
    1 for layer in model.layers
    if layer.trainable
)

print(
    f"\nTrainable top-level layers: "
    f"{trainable_count}"
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss=loss_fn,
    metrics=[
        "accuracy"
    ]
)


history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE2_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ],
    verbose=1
)


# ==============================================================
# LOAD BEST MODEL
# ==============================================================

print("\nLoading best V3 model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Best V3 model loaded.")


# ==============================================================
# FINAL VALIDATION EVALUATION
# ==============================================================

print("\n")
print("=" * 70)
print("FINAL VALIDATION RESULTS")
print("=" * 70)


val_loss, val_accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print(
    f"\nBest validation accuracy: "
    f"{val_accuracy:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


# ==============================================================
# SAVE FINAL MODEL
# ==============================================================

best_model.save(
    MODEL_PATH
)


print("\n")
print("=" * 70)
print("BANANA NUTRIENT V3 TRAINING COMPLETED")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nClass mapping:")
print(CLASS_PATH)

print("\nTraining log:")
print(LOG_PATH)

print("\nIMPORTANT:")
print(
    "The test dataset was NOT used during training."
)

print(
    "Run the separate V3 evaluation script on the "
    "untouched test dataset."
)

print("=" * 70)