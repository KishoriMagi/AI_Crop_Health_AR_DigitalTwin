"""
train_banana_nutrient_model.py

Banana Nutrient Deficiency Classification
Architecture: MobileNetV3Small

Classes:
    Healthy
    Boron Deficiency
    Calcium Deficiency
    Iron Deficiency
    Magnesium Deficiency
    Manganese Deficiency
    Potassium Deficiency
    Sulphur Deficiency
    Zinc Deficiency
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. PROJECT PATHS
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

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


TRAIN_CSV = os.path.join(
    DATASET_DIR,
    "nutrient_train.csv"
)

VAL_CSV = os.path.join(
    DATASET_DIR,
    "nutrient_val.csv"
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
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 15

RANDOM_SEED = 42


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 4. LOAD CSV FILES
# ============================================================

print("\n" + "=" * 70)
print("BANANA NUTRIENT MODEL TRAINING")
print("=" * 70)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_CSV)

val_df = pd.read_csv(VAL_CSV)


# ------------------------------------------------------------
# Keep Banana only
# ------------------------------------------------------------

train_df = train_df[
    train_df["crop"] == "Banana"
].copy()

val_df = val_df[
    val_df["crop"] == "Banana"
].copy()


print(
    f"\nTraining images:   {len(train_df)}"
)

print(
    f"Validation images: {len(val_df)}"
)


# ============================================================
# 5. CREATE CLASS MAPPING
# ============================================================

classes = sorted(
    train_df["label"].unique()
)

NUM_CLASSES = len(classes)

class_to_index = {
    class_name: index
    for index, class_name in enumerate(classes)
}

index_to_class = {
    index: class_name
    for class_name, index in class_to_index.items()
}


print(
    f"\nNumber of classes: {NUM_CLASSES}"
)

print("\nClasses:")

for index in range(NUM_CLASSES):

    print(
        f"{index}: {index_to_class[index]}"
    )


# Save class mapping

with open(
    CLASS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        index_to_class,
        file,
        indent=4
    )

print(
    f"\nClass mapping saved to:\n{CLASS_PATH}"
)


# ============================================================
# 6. CONVERT LABELS TO INTEGER
# ============================================================

train_df["class_id"] = train_df[
    "label"
].map(class_to_index)

val_df["class_id"] = val_df[
    "label"
].map(class_to_index)


# ============================================================
# 7. CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

print(
    train_df["label"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 8. CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["class_id"].values
)

class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(
        class_weights_array
    )
}


print("\nClass weights:")

for index in range(NUM_CLASSES):

    print(
        f"{index_to_class[index]:25s} "
        f"{class_weights[index]:.3f}"
    )


# ============================================================
# 9. IMAGE LOADING FUNCTION
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
# 10. DATA AUGMENTATION
# ============================================================

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


# ============================================================
# 11. CREATE TF.DATA DATASET
# ============================================================

def create_dataset(
    dataframe,
    training=False
):

    paths = dataframe[
        "image_path"
    ].values

    labels = dataframe[
        "class_id"
    ].values.astype(
        np.int32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:

        dataset = dataset.map(
            lambda image, label: (
                data_augmentation(
                    image,
                    training=True
                ),
                label
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# 12. CREATE DATASETS
# ============================================================

print("\nCreating TensorFlow datasets...")

train_dataset = create_dataset(
    train_df,
    training=True
)

val_dataset = create_dataset(
    val_df,
    training=False
)


# ============================================================
# 13. BUILD MOBILE NET V3 SMALL
# ============================================================

print("\nBuilding MobileNetV3Small model...")

base_model = tf.keras.applications.MobileNetV3Small(

    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),

    include_top=False,

    weights="imagenet",

    include_preprocessing=True
)


# ------------------------------------------------------------
# Freeze backbone initially
# ------------------------------------------------------------

base_model.trainable = False


# ============================================================
# 14. CLASSIFICATION HEAD
# ============================================================

inputs = tf.keras.Input(

    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),

    name="image"
)


x = base_model(
    inputs,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(
    x
)

x = tf.keras.layers.Dropout(
    0.30
)(
    x
)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="nutrient_prediction"
)(
    x
)


model = tf.keras.Model(
    inputs,
    outputs,
    name="Banana_Nutrient_MobileNetV3Small"
)


# ============================================================
# 15. COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# 16. MODEL SUMMARY
# ============================================================

print("\nModel summary:")

model.summary()


# ============================================================
# 17. CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=4,

    mode="max",

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# 18. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]

)


# ============================================================
# 19. TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)


best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_epoch = (
    np.argmax(
        history.history["val_accuracy"]
    ) + 1
)


print(
    f"\nBest epoch: {best_epoch}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


# ============================================================
# 20. SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print(
    f"\nModel saved to:\n{MODEL_PATH}"
)

print(
    f"\nClass mapping saved to:\n{CLASS_PATH}"
)


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("BANANA NUTRIENT TRAINING FINISHED")
print("=" * 70)

print(
    "\nNext step:"
)

print(
    "Run the Banana nutrient evaluation script "
    "on the untouched test dataset."
)