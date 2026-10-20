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

TRAIN_CSV = BASE_DIR / "datasets" / "disease_train_clean.csv"
VAL_CSV = BASE_DIR / "datasets" / "disease_val_clean.csv"

V1_MODEL = BASE_DIR / "ai" / "models" / "disease_mobilenetv3.keras"
V2_MODEL = BASE_DIR / "ai" / "models" / "disease_mobilenetv3_v2.keras"

CLASS_FILE = BASE_DIR / "ai" / "models" / "disease_classes.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

FINE_TUNE_EPOCHS = 12


print("=" * 70)
print("MOBILENETV3 DISEASE CLASSIFICATION - MODEL V2")
print("=" * 70)


# =========================================================
# 1. CHECK FILES
# =========================================================

print("\nChecking required files...")

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"Training CSV not found: {TRAIN_CSV}"
    )

if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"Validation CSV not found: {VAL_CSV}"
    )

if not V1_MODEL.exists():
    raise FileNotFoundError(
        f"V1 model not found: {V1_MODEL}"
    )

print("Training dataset found.")
print("Validation dataset found.")
print("V1 model found.")


# =========================================================
# 2. LOAD DATA
# =========================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print(f"Training images:   {len(train_df):,}")
print(f"Validation images: {len(val_df):,}")


# =========================================================
# 3. CREATE CLASS MAPPING
# =========================================================

classes = sorted(
    train_df["condition"].unique()
)

num_classes = len(classes)

class_to_index = {
    name: index
    for index, name in enumerate(classes)
}

index_to_class = {
    index: name
    for name, index in class_to_index.items()
}

print(f"\nNumber of classes: {num_classes}")

for index in range(num_classes):
    print(
        f"{index:2d} -> {index_to_class[index]}"
    )


# =========================================================
# 4. SAVE CLASS MAPPING
# =========================================================

CLASS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

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

print(
    f"\nClass mapping saved to:\n{CLASS_FILE}"
)


# =========================================================
# 5. CONVERT LABELS TO INTEGER
# =========================================================

train_df["label"] = train_df["condition"].map(
    class_to_index
)

val_df["label"] = val_df["condition"].map(
    class_to_index
)

if train_df["label"].isna().any():
    raise ValueError(
        "Unknown class found in training dataset."
    )

if val_df["label"].isna().any():
    raise ValueError(
        "Unknown class found in validation dataset."
    )

train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)


# =========================================================
# 6. CALCULATE CLASS WEIGHTS
# =========================================================

print("\nCalculating class weights...")

y_train = train_df["label"].values

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_train
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(class_weights_array)
}

print("\nClass weights:")

for index in range(num_classes):

    print(
        f"{index:2d} -> "
        f"{index_to_class[index]:35s} "
        f"{class_weights[index]:.3f}"
    )


# =========================================================
# 7. IMAGE LOADING
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
# 8. DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),
    ],
    name="data_augmentation"
)


# =========================================================
# 9. CREATE TRAINING PIPELINE
# =========================================================

print("\nPreparing training pipeline...")

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_df["image_path"].values,
        train_df["label"].values
    )
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(train_df),
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.map(
    lambda image, label: (
        data_augmentation(
            image,
            training=True
        ),
        label
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


# =========================================================
# 10. VALIDATION PIPELINE
# =========================================================

print("Preparing validation pipeline...")

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_df["image_path"].values,
        val_df["label"].values
    )
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# =========================================================
# 11. LOAD V1 MODEL
# =========================================================

print("\nLoading V1 model...")

model = tf.keras.models.load_model(
    V1_MODEL
)

print("V1 model loaded successfully.")


# =========================================================
# 12. FIND MOBILENET BACKBONE
# =========================================================

print("\nFinding MobileNetV3Small backbone...")

base_model = None

for layer in model.layers:

    if "mobilenetv3small" in layer.name.lower():

        base_model = layer
        break


if base_model is None:

    raise ValueError(
        "MobileNetV3Small backbone could not be found."
    )


print(
    f"Backbone found: {base_model.name}"
)


# =========================================================
# 13. UNFREEZE LAST 30% OF BACKBONE
# =========================================================

print("\nConfiguring fine-tuning...")

base_model.trainable = True

total_layers = len(
    base_model.layers
)

freeze_until = int(
    total_layers * 0.70
)

for i, layer in enumerate(
    base_model.layers
):

    if i < freeze_until:

        layer.trainable = False

    else:

        layer.trainable = True


# Keep BatchNormalization layers frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


trainable_layers = sum(
    layer.trainable
    for layer in base_model.layers
)


print(
    f"Total backbone layers: "
    f"{total_layers}"
)

print(
    f"Frozen backbone layers: "
    f"{total_layers - trainable_layers}"
)

print(
    f"Trainable backbone layers: "
    f"{trainable_layers}"
)


# =========================================================
# 14. COMPILE MODEL
# =========================================================

print("\nCompiling V2 model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)

print("Model compiled successfully.")


# =========================================================
# 15. CALLBACKS
# =========================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    V2_MODEL,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# =========================================================
# 16. TRAIN MODEL V2
# =========================================================

print("\n" + "=" * 70)
print("STARTING MODEL V2 FINE-TUNING")
print("=" * 70)

history = model.fit(
    train_dataset,

    validation_data=val_dataset,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# =========================================================
# 17. SAVE MODEL
# =========================================================

model.save(
    V2_MODEL
)


# =========================================================
# 18. FIND BEST VALIDATION RESULT
# =========================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_epoch = (
    np.argmax(
        history.history["val_accuracy"]
    ) + 1
)


# =========================================================
# 19. COMPLETION
# =========================================================

print("\n" + "=" * 70)
print("MODEL V2 TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nBest validation accuracy: "
    f"{best_val_accuracy:.4f} "
    f"({best_val_accuracy * 100:.2f}%)"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"\nV2 model saved to:\n{V2_MODEL}"
)

print(
    f"\nClass mapping saved to:\n{CLASS_FILE}"
)

print("\n" + "=" * 70)