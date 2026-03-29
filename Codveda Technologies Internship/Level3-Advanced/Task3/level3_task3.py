import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. LOAD AND PREPROCESS DATA
print("Loading MNIST dataset...")
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data (Scale pixels from 0-255 down to 0-1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# 2. DESIGN THE NEURAL NETWORK (Objective 2)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # Flattens the 2D image into a 1D line
    layers.Dense(128, activation='relu'), # Hidden layer with 128 "neurons"
    layers.Dropout(0.2),                  # Prevents the model from "cheating" (overfitting)
    layers.Dense(10, activation='softmax') # Output layer (10 digits: 0-9)
])

# 3. COMPILE THE MODEL
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. TRAIN THE MODEL (Objective 3)
print("\nStarting Training (The Brain is Learning)...")
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 5. EVALUATE AND VISUALIZE (Objective 4)
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.legend()

plt.show()

# Final Test
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nFinal Accuracy on Test Data: {test_acc*100:.2f}%')