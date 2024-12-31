import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
import pennylane as qml
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Dataset Configuration
# ------------------------------
dataset_path = r"D:\Diabetic retinopathy\archive"
image_dir = os.path.join(dataset_path, 'colored_images')
image_size = (128, 128)  # EfficientNet requires 128x128 input
folder_to_label = {'No_DR': 0, 'Mild': 1, 'Moderate': 2, 'Proliferate_DR': 3, 'Severe': 4}

# ------------------------------
# 2. Helper Function: Load Data
# ------------------------------
def load_data(image_dir, image_size):
    images, labels = [], []
    for folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                if img_path.lower().endswith(('.jpeg', '.jpg', '.png')):
                    img = cv2.imread(img_path)
                    if img is not None:
                        label = folder_to_label.get(folder, -1)
                        if label != -1:
                            img = cv2.resize(img, image_size)
                            images.append(img)
                            labels.append(label)
                        else:
                            print(f"Warning: Folder '{folder}' not mapped to label.")
                    else:
                        print(f"Warning: Failed to load {img_path}.")
    return np.array(images), np.array(labels)

# ------------------------------
# 3. Load and Preprocess Data
# ------------------------------
print("Loading data...")
images, labels = load_data(image_dir, image_size)
print(f"Total images loaded: {images.shape[0]}")

# Normalize the images
images = images / 255.0

# Encode labels using One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
labels_encoded = encoder.fit_transform(labels.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ------------------------------
# 4. Feature Extraction with EfficientNetB0
# ------------------------------
# Load pre-trained EfficientNetB0 model + higher level layers
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the base model

# Create a feature extractor model
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Extract features
print("Extracting features from training data...")
X_train_features = feature_extractor.predict(X_train, batch_size=32, verbose=1)
print("Extracting features from testing data...")
X_test_features = feature_extractor.predict(X_test, batch_size=32, verbose=1)

# Flatten the features
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)
print(f"Shape of training features: {X_train_features.shape}")
print(f"Shape of testing features: {X_test_features.shape}")

# ------------------------------
# 5. Dimensionality Reduction with PCA
# ------------------------------
num_qubits = 4  # Number of qubits for the quantum layer

# pca = PCA(n_components=num_qubits, random_state=42)
# print("Applying PCA to reduce feature dimensions...")
# X_train_reduced = pca.fit_transform(X_train_features)
# X_test_reduced = pca.transform(X_test_features)
# print(f"Shape of reduced training features: {X_train_reduced.shape}")
# print(f"Shape of reduced testing features: {X_test_reduced.shape}")

# Normalize the reduced features
def normalize_features(features):
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return np.where(norm > 0, features / norm, features)

# X_train_normalized = normalize_features(X_train_reduced)
# X_test_normalized = normalize_features(X_test_reduced)

# ------------------------------
# 6. Define the Quantum Layer
# ------------------------------
class QuantumLayer(Layer):
    def __init__(self, num_qubits, weight_shapes, **kwargs):
        super().__init__(**kwargs)
        self.num_qubits = num_qubits
        self.weight_shapes = weight_shapes

        self.quantum_weights = self.add_weight(
            name="quantum_weights",
            shape=self.weight_shapes["weights"],
            initializer="random_normal",
            trainable=True,
        )

        # Define the quantum device and quantum circuit
        dev = qml.device("default.qubit.tf", wires=self.num_qubits)

        @qml.qnode(dev, interface="tf")
        def quantum_circuit(inputs, weights):
            # Apply rotation gates
            for i in range(self.num_qubits):
                qml.RX(inputs[i], wires=i)
            # Apply strongly entangling layers
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            # Return the expectation values (as a list of float64 tensors)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.qnode = quantum_circuit

    def call(self, inputs):
        # Ensure inputs are tensors with consistent float32 types
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        quantum_weights = tf.convert_to_tensor(self.quantum_weights, dtype=tf.float32)

        # Use tf.vectorized_map to handle the batch processing
        def apply_quantum_circuit(x):
            return self.qnode(x, quantum_weights)

        # Apply the quantum circuit to the entire batch
        output = tf.vectorized_map(apply_quantum_circuit, inputs)

        # Stack the output to create a single tensor (flatten the list of tensors)
        output = tf.stack(output, axis=-1)  # Stack along the last axis (num_qubits)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_qubits)  # This matches the output shape from qnode


def create_quantum_model(input_dim, num_qubits, num_classes):
    weight_shapes = {"weights": (3, num_qubits, 3)}  # Adjust the weight shape accordingly

    quantum_layer = QuantumLayer(
        num_qubits=num_qubits,
        weight_shapes=weight_shapes
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        quantum_layer,
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    return model
    
if X_train_features.shape[1] < num_qubits:
    raise ValueError(
        f"Number of PCA components ({X_train_features.shape[1]}) "
        f"must be at least equal to the number of qubits ({num_qubits})."
    )

pca = PCA(n_components=num_qubits, random_state=42)

# Apply PCA and normalize features
print("Applying PCA to reduce feature dimensions...")
X_train_reduced = pca.fit_transform(X_train_features)
X_test_reduced = pca.transform(X_test_features)
print(f"Shape of reduced training features: {X_train_reduced.shape}")
print(f"Shape of reduced testing features: {X_test_reduced.shape}")

X_train_normalized = normalize_features(X_train_reduced)
X_test_normalized = normalize_features(X_test_reduced)

num_classes = len(folder_to_label)
model = create_quantum_model(input_dim=num_qubits, num_qubits=num_qubits, num_classes=num_classes)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

# ------------------------------
# 9. Compute Class Weights
# ------------------------------
# Convert one-hot labels back to single integers for class_weight computation
y_train_integers = np.argmax(y_train, axis=1)
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_integers),
    y=y_train_integers
)
class_weights = dict(zip(np.unique(y_train_integers), class_weights_array))
print(f"Class weights: {class_weights}")

# ------------------------------
# 10. Train the Model (without callbacks)
# ------------------------------
history = model.fit(
    X_train_normalized,
    y_train,
    validation_data=(X_test_normalized, y_test),
    epochs=50,
    batch_size=16,
    class_weight=class_weights,
    verbose=1
)

# ------------------------------
# 11. Evaluate the Model
# ------------------------------
test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")

# ------------------------------
# 12. Predictions and Evaluation Metrics
# ------------------------------
# Predictions for Confusion Matrix and Classification Report
y_pred = model.predict(X_test_normalized)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=folder_to_label.keys(),
            yticklabels=folder_to_label.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=folder_to_label.keys()))

# Save the model to a file
model_save_path = "quantum_model.keras"  # Specify the file path
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
