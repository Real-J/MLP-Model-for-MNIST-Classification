import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Define the MLP model
class MLP(models.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = layers.Flatten()
        self.input_layer = layers.Dense(512, activation='relu')
        self.hidden_layer = layers.Dense(512, activation='relu')
        self.output_layer = layers.Dense(10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        return self.output_layer(x)

# Hyperparameters
batch_size = 128
epoch_count = 12
learning_rate = 0.1

# Prepare MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Normalize the dataset
mean = 0.1307
std = 0.3081
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Initialize model, optimizer, and loss function
model = MLP()
optimizer = optimizers.SGD(learning_rate=learning_rate)
loss_fn = losses.CategoricalCrossentropy(from_logits=True)

# Training loop
print("Starting training...")
for epoch in range(epoch_count):
    # Training
    for step, (images, labels) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Evaluation on the test set
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for images, labels in test_dataset:
        predictions = model(images, training=False)
        loss = loss_fn(labels, predictions)
        test_loss.update_state(loss)
        test_accuracy.update_state(labels, predictions)

    print(f'Epoch {epoch + 1}, Loss: {test_loss.result():.4f}, Accuracy: {test_accuracy.result() * 100:.2f}%')
