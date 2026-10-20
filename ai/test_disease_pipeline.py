import pandas as pd
import tensorflow as tf

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = BASE_DIR / "datasets" / "disease_train_clean.csv"

IMAGE_SIZE = (224, 224)


df = pd.read_csv(CSV_FILE)

print("=" * 60)
print("TESTING IMAGE PIPELINE")
print("=" * 60)

print("Images:", len(df))

paths = df["image_path"].values[:5]
labels = df["condition"].values[:5]

for path, label in zip(paths, labels):

    print("\nLabel:", label)
    print("Path:", path)

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    print(
        "Image shape:",
        image.shape
    )

print("\n" + "=" * 60)
print("PIPELINE TEST PASSED")
print("=" * 60)