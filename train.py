import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# a single color channel (the 1 at the end (28,28,1)),
# chat helped here as we were no familiar with the color channels and greyscale and that we need to specify it like this
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32") / 255

# one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# compiling
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# training
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)

# evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# saving the model
model.save("mnist_cnn_model.keras")
print("Model saved to mnist_cnn_model.keras")
