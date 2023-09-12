from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import SparseCategoricalCrossentropy 
from tensorflow.keras.models import Sequential


(x, y), (x_test, y_test) = mnist.load_data()

x = x / 255.
x_test = x_test / 255.

data_augmentation = Sequential([
    layers.RandomZoom(.2, input_shape=(28, 28, 1)),
    layers.RandomRotation(.2)
])

model = Sequential([
    layers.Conv2D(5, (5, 5), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(50, (5, 5), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dropout(1/3),
    layers.Dense(100, activation="relu"),

    layers.Dense(10),
    layers.Softmax()
])

model_wa = Sequential([
    data_augmentation,
    model
])

model.compile(
    optimizer = 'adam',
    loss = SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)

model_wa.compile(
    optimizer = 'adam',
    loss = SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)

model_wa.fit(x, y, validation_split=0.2, epochs=200)

model.evaluate(x_test, y_test)

model.save("./model.h5")
