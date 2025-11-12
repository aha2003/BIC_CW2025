#  import numpy as np

# class ANN:
#     """
#     A feedforward multi-layer ANN designed for optimization via PSO.
#     This version dynamically decodes activation functions for hidden layers
#     from the solution vector itself, alongside the weights and biases.
#     """

#     def __init__(self, layer_sizes):
#         """
#         Initializes the ANN with a given architecture.
#         Args:
#             layer_sizes (list of int): Number of neurons in each layer, e.g., [8, 10, 10, 1].
#         """
#         self.layer_sizes = layer_sizes
        
#         # --- Available activation functions for PSO to choose from ---
#         self.activation_options = [
#             self.relu,    # Index 0
#             self.tanh,    # Index 1
#             self.logistic # Index 2
#         ]
        
#         # Placeholders to be filled dynamically
#         self.activations = [None] * (len(layer_sizes) - 1)
#         self.weights = []
#         self.biases = []

#         # Initialize weights and biases with placeholder shapes
#         for i in range(len(layer_sizes) - 1):
#             w_shape = (layer_sizes[i], layer_sizes[i+1])
#             b_shape = (layer_sizes[i+1],)
#             self.weights.append(np.empty(w_shape))
#             self.biases.append(np.empty(b_shape))

#     # --- Static methods for activation functions ---
#     @staticmethod
#     def logistic(x): return 1 / (1 + np.exp(-x))
#     @staticmethod
#     def relu(x): return np.maximum(0, x)
#     @staticmethod
#     def tanh(x): return np.tanh(x)

#     def get_total_params(self, include_activations=False):
#         """
#         Calculates the total number of parameters to be optimized.
#         Args:
#             include_activations (bool): If True, adds dimensions for the
#                                         activation function choices.
#         Returns:
#             int: The total number of parameters.
#         """
#         total_params = 0
#         if include_activations:
#             # Add one parameter for each hidden layer's activation choice
#             num_hidden_layers = len(self.layer_sizes) - 2
#             total_params += num_hidden_layers
            
#         # Add weights and biases
#         for i in range(len(self.layer_sizes) - 1):
#             total_params += self.layer_sizes[i] * self.layer_sizes[i+1]
#             total_params += self.layer_sizes[i+1]
            
#         return total_params

#     def set_params_from_vector(self, params_vector):
#         """
#         Decodes a full solution vector to set activation functions, weights, and biases.
#         Args:
#             params_vector (np.ndarray): The flat vector from a PSO particle.
#         """
#         num_hidden_layers = len(self.layer_sizes) - 2
        
#         # --- 1. Decode and set activation functions for hidden layers ---
#         activation_choices = params_vector[:num_hidden_layers]
#         for i in range(num_hidden_layers):
#             # Convert the float value from PSO into a valid integer index
#             choice_index = int(abs(activation_choices[i])) % len(self.activation_options)
#             self.activations[i] = self.activation_options[choice_index]
        
#         # The output layer has a linear activation (None) for regression
#         self.activations[-1] = None
        
#         # --- 2. Set weights and biases from the rest of the vector ---
#         weights_biases_vector = params_vector[num_hidden_layers:]
        
#         pointer = 0
#         for i in range(len(self.layer_sizes) - 1):
#             # Unpack weights
#             w_size = self.layer_sizes[i] * self.layer_sizes[i+1]
#             self.weights[i] = weights_biases_vector[pointer : pointer + w_size].reshape((self.layer_sizes[i], self.layer_sizes[i+1]))
#             pointer += w_size
            
#             # Unpack biases
#             b_size = self.layer_sizes[i+1]
#             self.biases[i] = weights_biases_vector[pointer : pointer + b_size]
#             pointer += b_size

#     def forward(self, inputs):
#         """Performs a forward pass through the network."""
#         x = inputs
#         for i in range(len(self.weights)):
#             z = np.dot(x, self.weights[i]) + self.biases[i]
#             if self.activations[i] is not None:
#                 x = self.activations[i](z)
#             else:
#                 x = z
#         return x
import numpy as np

class ANN:
    """
    A simple feedforward multi-layer artificial neural network (ANN).
    This version is configured with a fixed architecture and activation functions
    at initialization, suitable for the separated experiments.
    """

    def __init__(self, layer_sizes, activations):
        """
        Initializes the ANN.
        Args:
            layer_sizes (list of int): A list specifying the number of neurons in each layer.
            activations (list of callable): A list of activation functions for each layer
                                          after the input layer.
        """
        if len(layer_sizes) - 1 != len(activations):
            raise ValueError("Number of activation functions must match number of hidden/output layers.")
            
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights = []
        self.biases = []

        # Initialize weights and biases with placeholder shapes
        for i in range(len(layer_sizes) - 1):
            w_shape = (layer_sizes[i], layer_sizes[i+1])
            b_shape = (layer_sizes[i+1],)
            self.weights.append(np.empty(w_shape))
            self.biases.append(np.empty(b_shape))

    @staticmethod
    def logistic(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    def get_total_params(self):
        """Calculates the total number of weights and biases in the network."""
        total_params = 0
        for i in range(len(self.layer_sizes) - 1):
            total_params += self.layer_sizes[i] * self.layer_sizes[i+1]
            total_params += self.layer_sizes[i+1]
        return total_params

    def set_params_from_vector(self, params_vector):
        """
        Sets the network's weights and biases from a flat 1D vector.
        Args:
            params_vector (np.ndarray): A flat vector containing all weights and biases.
        """
        if len(params_vector) != self.get_total_params():
            raise ValueError("The length of the params vector does not match the total number of network parameters.")
            
        pointer = 0
        for i in range(len(self.layer_sizes) - 1):
            w_size = self.layer_sizes[i] * self.layer_sizes[i+1]
            self.weights[i] = params_vector[pointer : pointer + w_size].reshape((self.layer_sizes[i], self.layer_sizes[i+1]))
            pointer += w_size
            
            b_size = self.layer_sizes[i+1]
            self.biases[i] = params_vector[pointer : pointer + b_size]
            pointer += b_size

    def forward(self, inputs):
        """Performs a forward pass through the network."""
        x = inputs
        for i in range(len(self.weights)):
            z = np.dot(x, self.weights[i]) + self.biases[i]
            if self.activations[i] is not None:
                x = self.activations[i](z)
            else:
                x = z
        return x

