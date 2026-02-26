import numpy as np


class MLP:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))

        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros((1, output_size))

        self.hidden_sum = np.empty(0, dtype=np.float64)
        self.hidden_activation = np.empty(0, dtype=np.float64)
        self.output_sum = np.empty(0, dtype=np.float64)
        self.output_activation = np.empty(0, dtype=np.float64)

    def sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray):
        return x * (1 - x)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.hidden_sum = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_activation = self.sigmoid(self.hidden_sum)

        self.output_sum = (
            np.dot(self.hidden_activation, self.weights_hidden_output)
            + self.bias_output
        )
        self.output_activation = self.sigmoid(self.output_sum)

        return self.output_activation

    def train(
        self, inputs: np.ndarray, targets: np.ndarray, learning_rate: float = 0.1
    ):
        predictions = self.forward(inputs)

        error_output = targets - predictions
        delta_output = error_output * self.sigmoid_derivative(predictions)

        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_activation)

        self.weights_hidden_output += (
            self.hidden_activation.T.dot(delta_output) * learning_rate
        )
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden += inputs.T.dot(delta_hidden) * learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

        return float(np.mean(np.square(error_output)))
