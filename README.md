# MLP Model for MNIST Classification

This repository contains a Python implementation of a Multi-Layer Perceptron (MLP) using TensorFlow and Keras. The model is designed to classify handwritten digits from the MNIST dataset.

## Features
- Implements a simple MLP architecture.
- Uses TensorFlow's `tf.GradientTape` for custom training loops.
- Includes data preprocessing (normalization and one-hot encoding).
- Provides training and evaluation on the MNIST dataset.

## Model Architecture
The MLP consists of the following layers:
1. **Flatten Layer**: Converts 28x28 images into a 1D vector.
2. **Dense Layer 1**: Fully connected layer with 512 units and ReLU activation.
3. **Dense Layer 2**: Fully connected layer with 512 units and ReLU activation.
4. **Output Layer**: Fully connected layer with 10 units (no activation, as logits are used).

## Hyperparameters
- **Batch Size**: 128
- **Epochs**: 12
- **Learning Rate**: 0.1

## Requirements
Ensure the following dependencies are installed:
- Python 3.7+
- TensorFlow 2.x

Install TensorFlow using:
```bash
pip install tensorflow
```

## Dataset
The code uses the MNIST dataset, which is automatically downloaded using TensorFlow's `keras.datasets.mnist` module. MNIST consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

## Preprocessing
1. The images are normalized to have a mean of `0.1307` and a standard deviation of `0.3081`.
2. Labels are one-hot encoded to match the output layer format.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/mlp-mnist.git
   cd mlp-mnist
   ```
2. Run the script:
   ```bash
   python mlp_mnist.py
   ```

## Training and Evaluation
During training:
- The model uses Stochastic Gradient Descent (SGD) with a learning rate of 0.1.
- The loss function is Categorical Crossentropy with logits.

After each epoch, the model's performance on the test dataset is evaluated and printed.

### Example Output
```
Starting training...
Epoch 1, Loss: 0.3302, Accuracy: 89.25%
Epoch 2, Loss: 0.2158, Accuracy: 93.12%
...
Epoch 12, Loss: 0.1104, Accuracy: 96.35%
```

## Customization
You can modify the hyperparameters (e.g., batch size, learning rate, epochs) in the script. Additionally, the model architecture can be extended by adding more layers or changing activation functions.

## Contributions
Contributions are welcome! Feel free to submit a pull request or open an issue for improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

