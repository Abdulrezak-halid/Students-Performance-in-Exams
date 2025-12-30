"""
İleri Beslemeli Sinir Ağı Uygulaması
Temel sinir ağı kavramlarını uygular: aktivasyon fonksiyonları, ileri yayılım,
zincir kuralı ile geri yayılım ve gradyan iniş optimizasyonu.
"""
import numpy as np
import pickle
from typing import List, Tuple, Dict


class ActivationFunctions:
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        # Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
        
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        # Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))
     
        return a * (1 - a)
    
    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        # ReLU activation function: f(z) = max(0, z)
       
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        # Derivative of ReLU: f'(z) = 1 if z > 0 else 0
        
        return (z > 0).astype(float)
    
    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        # Tanh activation function: f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
        
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(a: np.ndarray) -> np.ndarray:
        # Derivative of tanh: f'(z) = 1 - tanh²(z)
        
        return 1 - a**2
    
    @staticmethod
    def linear(z: np.ndarray) -> np.ndarray:
        # Linear activation function: f(z) = z
        # Used for regression output layer.
       
        return z
    
    @staticmethod
    def linear_derivative(z: np.ndarray) -> np.ndarray:
        # Derivative of linear: f'(z) = 1
        return np.ones_like(z)


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], activation: str = 'relu', 
                 learning_rate: float = 0.01, random_state: int = 42):

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.activation_name = activation
        
        self.activation_functions = ActivationFunctions()
        if activation == 'relu':
            self.activation = self.activation_functions.relu
            self.activation_derivative = self.activation_functions.relu_derivative
        elif activation == 'sigmoid':
            self.activation = self.activation_functions.sigmoid
            self.activation_derivative = self.activation_functions.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = self.activation_functions.tanh
            self.activation_derivative = self.activation_functions.tanh_derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        np.random.seed(random_state)
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            if activation == 'relu':
                # std = sqrt(2/n_in)
                w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                # std = sqrt(1/n_in)
                w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(1.0 / layer_sizes[i])
            
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
        
        self.train_loss_history = []
        self.val_loss_history = []
        
        print(f"\nNeural Network Architecture:")
        print(f"  Layers: {' -> '.join(map(str, layer_sizes))}")
        print(f"  Activation: {activation}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Total parameters: {self.count_parameters()}")
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total
    
    def forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        pre_activations = []
        
        a = X
        for i in range(self.n_layers - 1):
            # linear transformation: z = W @ a + b
            z = self.weights[i] @ a + self.biases[i]
            pre_activations.append(z)
            
            if i == self.n_layers - 2:  # Output layer
                a = self.activation_functions.linear(z)
            else:  # Hidden layers
                a = self.activation(z)
            
            activations.append(a)
        
        return activations, pre_activations
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error loss.
        Loss: L = (1/2m) * Σ(y_pred - y_true)²
        """
        m = y_true.shape[1]
        loss = (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)
        return loss
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray, 
                           activations: List[np.ndarray], 
                           pre_activations: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        m = X.shape[1]  
        
        weight_grads = [None] * (self.n_layers - 1)
        bias_grads = [None] * (self.n_layers - 1)
        
        delta = activations[-1] - y
        
        for i in reversed(range(self.n_layers - 1)):
            weight_grads[i] = (1 / m) * (delta @ activations[i].T)
            bias_grads[i] = (1 / m) * np.sum(delta, axis=1, keepdims=True)
            
            if i > 0:
                delta = (self.weights[i].T @ delta) * self.activation_derivative(pre_activations[i-1])
        
        return weight_grads, bias_grads
    
    def update_parameters(self, weight_grads: List[np.ndarray], bias_grads: List[np.ndarray]):
        for i in range(self.n_layers - 1):
            self.weights[i] -= self.learning_rate * weight_grads[i]
            self.biases[i] -= self.learning_rate * bias_grads[i]
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        activations, pre_activations = self.forward_propagation(X)
        
        loss = self.compute_loss(y, activations[-1])
        
        weight_grads, bias_grads = self.backward_propagation(X, y, activations, pre_activations)
        
        self.update_parameters(weight_grads, bias_grads)
        
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward_propagation(X)
        return activations[-1]
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 1000, batch_size: int = 32, verbose: int = 100):
        
        # Train the neural network.
        X_train = X_train.T
        y_train = y_train.T
        
        if X_val is not None:
            X_val = X_val.T
            y_val = y_val.T
        
        n_samples = X_train.shape[1]
        
        print(f"\nTraining started...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_train[:, batch_indices]
                y_batch = y_train[:, batch_indices]
                
                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            self.train_loss_history.append(avg_train_loss)
            
            if X_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = self.compute_loss(y_val, val_predictions)
                self.val_loss_history.append(val_loss)
            
            if (epoch + 1) % verbose == 0 or epoch == 0:
                if X_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        print(f"\nTraining completed!")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        
        #Evaluate model performance.
        X = X.T
        y = y.T
        
        predictions = self.predict(X)
        
        # MSE
        mse = np.mean((predictions - y) ** 2, axis=1)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAE
        mae = np.mean(np.abs(predictions - y), axis=1)
        
        # R² score
        ss_res = np.sum((y - predictions) ** 2, axis=1)
        ss_tot = np.sum((y - np.mean(y, axis=1, keepdims=True)) ** 2, axis=1)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save model parameters."""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'activation': self.activation_name,
            'learning_rate': self.learning_rate,
            'weights': self.weights,
            'biases': self.biases,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model parameters."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.layer_sizes = model_data['layer_sizes']
        self.activation_name = model_data['activation']
        self.learning_rate = model_data['learning_rate']
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.train_loss_history = model_data['train_loss_history']
        self.val_loss_history = model_data['val_loss_history']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Testing Neural Network Implementation...")
    
    # create simple dataset
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100, 3)
    
    # create network
    nn = NeuralNetwork(layer_sizes=[5, 64, 3], activation='relu', learning_rate=0.01)
    
    # Train
    nn.fit(X, y, epochs=100, batch_size=16, verbose=50)
    
    # Evaluate
    metrics = nn.evaluate(X, y)
    print(f"\nMetrics: {metrics}")
