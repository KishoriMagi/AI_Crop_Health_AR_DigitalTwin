"""
train_banana_nutrient_model_v2.py

Banana Nutrient Deficiency Classification - V2

V2 improvements:
- Fine-tunes the later MobileNetV3Small layers
- Keeps BatchNormalization layers frozen
- Uses class weighting
- Uses a small learning rate
- Uses data augmentation
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PATHS
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

os.makedirs(MODEL_DIR, exist_ok=True)


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
    "banana_nutrient_mobilenetv3_v2.keras"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "banana_nutrient_classes.json"
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 15

RANDOM_SEED = 42


tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("BANANA NUTRIENT MODEL V2 TRAINING")
print("=" * 70)

train_df = pd.read_csv(TRAIN_CSV)

val_df = pd.read_csv(VAL_CSV)


train_df = train_df[
    train_df["crop"] == "Banana"
].copy()

val_df = val_df[
    val_df["crop"] == "Banana"
].copy()


print(
    f"\nTraining images: {len(train_df)}"
)

print(
    f"Validation images: {len(val_df)}"
)


# ============================================================
# CLASS MAPPING
# ============================================================

classes = sorted(
    train_df["label"].unique()
)

NUM_CLASSES = len(classes)

class_to_index = {
    name: i
    for i, name in enumerate(classes)
}

index_to_class = {
    i: name
    for name, i in class_to_index.items()
}


print(
    f"\nNumber of classes: {NUM_CLASSES}"
)

for i in range(NUM_CLASSES):

    print(
        f"{i}: {index_to_class[i]}"
    )


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


# ============================================================
# LABELS
# ============================================================

train_df["class_id"] = train_df[
    "label"
].map(class_to_index)

val_df["class_id"] = val_df[
    "label"
].map(class_to_index)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["class_id"].values
)

class_weights = {
    int(i): float(weight)
    for i, weight in enumerate(
        class_weights_array
    )
}


print("\nClass weights:")

for i in range(NUM_CLASSES):

    print(
        f"{index_to_class[i]:25s} "
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
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(

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
# DATASET
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
    ].values.astype(np.int32)

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
                augmentation(
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


train_dataset = create_dataset(
    train_df,
    training=True
)

val_dataset = create_dataset(
    val_df,
    training=False
)


# ============================================================
# MOBILE NET V3 SMALL
# ============================================================

print("\nBuilding MobileNetV3Small...")

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


# ============================================================
# FINE-TUNING
# ============================================================

base_model.trainable = True


# Count layers

total_layers = len(
    base_model.layers
)


# Freeze approximately first 70%

freeze_until = int(
    total_layers * 0.70
)


print(
    f"\nTotal backbone layers: "
    f"{total_layers}"
)

print(
    f"Freezing first: "
    f"{freeze_until} layers"
)


for i, layer in enumerate(
    base_model.layers
):

    if i < freeze_until:

        layer.trainable = False

    else:

        # Keep BatchNormalization frozen
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

        else:

            layer.trainable = True


trainable_layers = sum(
    layer.trainable
    for layer in base_model.layers
)

frozen_layers = total_layers - trainable_layers


print(
    f"Trainable backbone layers: "
    f"{trainable_layers}"
)

print(
    f"Frozen backbone layers: "
    f"{frozen_layers}"
)


# ============================================================
# MODEL HEAD
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
    name="Banana_Nutrient_MobileNetV3Small_V2"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel summary:")

model.summary()


# ============================================================
# CALLBACKS
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

    patience=5,

    mode="max",

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


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING V2 TRAINING")
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
# RESULTS
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_epoch = (
    np.argmax(
        history.history["val_accuracy"]
    ) + 1
)


print("\n" + "=" * 70)
print("V2 TRAINING COMPLETED")
print("=" * 70)

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
# SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print(
    f"\nV2 model saved to:\n{MODEL_PATH}"
)

print(
    f"\nClass mapping saved to:\n{CLASS_PATH}"
)


print("\nDONE.")