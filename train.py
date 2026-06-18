import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_FROZEN = 10
EPOCHS_FINETUNE = 10
MODEL_PATH = "alzheimer_model.h5"
CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# Locate dataset root — handles nested zip extractions
def find_dataset_root(base="dataset"):
    for root, dirs, _ in os.walk(base):
        if any(c in dirs for c in CLASSES):
            return root
    raise FileNotFoundError(
        f"Could not find class folders under '{base}'. "
        "Run download_data.py first."
    )

# ── Data ──────────────────────────────────────────────────────────────────────
dataset_root = find_dataset_root()
print(f"Dataset root: {dataset_root}")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    dataset_root,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

val_gen = val_datagen.flow_from_directory(
    dataset_root,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
)

print("Class indices:", train_gen.class_indices)
num_classes = len(train_gen.class_indices)

# ── Class weights ─────────────────────────────────────────────────────────────
labels = train_gen.classes
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels,
)
class_weights = dict(enumerate(class_weights_array))
print("Class weights:", class_weights)

# ── Model ─────────────────────────────────────────────────────────────────────
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Phase 1: train head ───────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6),
]

print("\n=== Phase 1: Training classification head ===")
history1 = model.fit(
    train_gen,
    epochs=EPOCHS_FROZEN,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ── Phase 2: fine-tune top layers ─────────────────────────────────────────────
base_model.trainable = True
# Freeze all but the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n=== Phase 2: Fine-tuning top layers ===")
history2 = model.fit(
    train_gen,
    epochs=EPOCHS_FINETUNE,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n=== Evaluation on validation set ===")
val_gen.reset()
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes

class_names = list(train_gen.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved to confusion_matrix.png")

# Training curves
def plot_history(h1, h2):
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="Train Acc")
    axes[0].plot(val_acc, label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(loss, label="Train Loss")
    axes[1].plot(val_loss, label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Training curves saved to training_curves.png")

plot_history(history1, history2)
print("\nTraining complete.")
