import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

# Load preprocessed data
data = np.load("preprocessed_data.npz")
X_train, Y_train = data["X_train"], data["Y_train"]
X_val, Y_val = data["X_val"], data["Y_val"]
X_test, Y_test = data["X_test"], data["Y_test"]

print(f"Loaded data - Training: {len(X_train)}, Validation: {len(X_val)}, Testing: {len(X_test)}")

# Build classifier model
model = keras.Sequential([
    layers.Input(shape=(1024,)),  # YAMNet embedding size
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),  # Dropout to prevent overfitting
    layers.Dense(128, activation="relu"),
    layers.Dense(2, activation="softmax")  # Two classes: background, vehicles
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the classifier
history = model.fit(
    X_train, Y_train,
    epochs=10,
    validation_data=(X_val, Y_val),  # Use validation set
    batch_size=16
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the trained model
model.save("yamnet_finetuned.h5")
print("Model saved successfully.")
