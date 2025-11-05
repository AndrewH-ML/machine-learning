#!/usr/bin/env python3
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
from matplotlib import pyplot as plt 
matplotlib.use("AGG")

import tensorflow as tf
import tensorflow_datasets as tfds
import tf_keras as keras
from model import unet_model

OUT_DIR = '/workspace/outputs'
os.makedirs(OUT_DIR, exist_ok= True)

IMG_H, IMG_W = 96, 128
N_CLASSES = 3

def print_gpu_info():
    """Print useful GPU info so we know the container sees the device."""
    print("\n=== TensorFlow / GPU info ===")
    print("TF version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs discovered:", gpus)
    for i, d in enumerate(gpus):
        try:
            det = tf.config.experimental.get_device_details(d)
            print(f"GPU[{i}] name:", det.get("device_name"))
            print(f"GPU[{i}] compute_capability:", det.get("compute_capability"))
        except Exception:
            pass
    print("=============================\n")


def preprocess_one(example):
    image  = tf.image.resize(example["image"], (IMG_H, IMG_W), method="bilinear")
    mask   = tf.image.resize(example["segmentation_mask"], (IMG_H, IMG_W), method="nearest")

    image  = tf.cast(image, tf.float32) / 255.0
    mask   = tf.cast(mask, tf.int32) - 1
    mask   = tf.where(mask < 0, 0, mask)
    mask = tf.squeeze(mask, axis = -1)
    
    return image, mask 

def to_numpy_subset(ds, count):
    images, masks = [], []
    for ex in ds.take(count):
        img, msk = preprocess_one(ex)
        images.append(img)
        masks.append(msk)
    X = tf.stack(images, axis=0)
    Y = tf.stack(masks,  axis=0)
    return X, Y

def create_mask(logits):
    """(B,H,W,C) logits -> (B,H,W) integer class IDs via argmax."""
    return tf.argmax(logits, axis=-1)

def save_triplet(image, true_mask, pred_mask, path):
    """Save (image, true, pred) side-by-side to a PNG at `path`."""
    plt.figure(figsize=(10, 3))
    for i, data in enumerate([image, true_mask, pred_mask]):
        plt.subplot(1, 3, i + 1)
        if i == 0:
            plt.imshow(data)
            plt.title("Image")
        else:
            plt.imshow(data, vmin=0, vmax=N_CLASSES - 1)
            plt.title("Mask" if i == 1 else "Pred")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_learning_curves(history, out_prefix):
    """Save loss/accuracy curves from Keras History to PNG files."""
    hist = history.history

    # Loss curve
    plt.figure()
    plt.plot(hist["loss"], label="train")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss")
    plt.legend()
    plt.savefig(f"{out_prefix}_loss.png", dpi=150)
    plt.close()

    # Accuracy curve (only if present)
    if "accuracy" in hist:
        plt.figure()
        plt.plot(hist["accuracy"], label="train")
        if "val_accuracy" in hist:
            plt.plot(hist["val_accuracy"], label="val")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy")
        plt.legend()
        plt.savefig(f"{out_prefix}_accuracy.png", dpi=150)
        plt.close()

def main():

    print_gpu_info()

    # ---- Load using your exact TFDS structure ----
    (ds_train, ds_test), ds_info = tfds.load(
        "oxford_iiit_pet",
        split=["train", "test"],
        with_info=True,
        as_supervised=False,
    )

    # ---- Print what tfds.load returned FIRST ----
    print("Types returned by tfds.load:", type(ds_train), type(ds_test))
    print("Element spec (train):", ds_train.element_spec)
    print("Split sizes from ds_info -> train:",
          ds_info.splits["train"].num_examples, " test:",
          ds_info.splits["test"].num_examples)

    # ---- Materialize small subsets to dense tensors ----
    TRAIN_COUNT = 200
    VAL_COUNT   = 40

    X_train, Y_train = to_numpy_subset(ds_train, count=TRAIN_COUNT)
    X_val,   Y_val   = to_numpy_subset(ds_test,  count=VAL_COUNT)

    # ---- Print tensor shapes for each split ----
    print("X_train shape:", X_train.shape, " Y_train shape:", Y_train.shape)
    print("X_val   shape:", X_val.shape,   " Y_val   shape:", Y_val.shape)

    # ---- Build / compile / train ----
    model = unet_model(input_size=(IMG_H, IMG_W, 3), n_filters=32, n_classes=N_CLASSES)
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    csv_path = os.path.join(OUT_DIR, "train_log.csv")
    callbacks = [
        keras.callbacks.CSVLogger(csv_path, separator=',', append=False)
    ]
    
    history = model.fit(
        X_train, Y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=2
        )

    # ---- Predict a few and visualize ----
    logits = model.predict(X_val[:6], verbose = 0)
    preds  = create_mask(logits)
    for i in range(min(6, X_val.shape[0])):
        out_file = os.path.join(OUT_DIR, f"pred_{i:02d}.png")
        save_triplet(X_val[i].numpy(), Y_val[i].numpy(), preds[i].numpy(), out_file)
        print(f"Saved CSV metrics to: {csv_path}")
        print(f"Saved plots & preds under: {OUT_DIR}")

if __name__ == "__main__":
    main()