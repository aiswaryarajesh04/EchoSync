import os
import numpy as np
import librosa
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

# Load YAMNet model
yamnet_model = hub.load("https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1")

# Dataset directories
data_dir = "dataset"  # Change this to your dataset path
classes = ["background", "vehicles"]

# Function to extract YAMNet embeddings
def extract_embedding(file_path):
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)  # Convert to 16kHz mono
    waveform = waveform.astype(np.float32)

    # Get YAMNet embeddings
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    # Average embeddings over time
    return np.mean(embeddings.numpy(), axis=0)

# Prepare dataset
X, Y = [], []
for label, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        try:
            embedding = extract_embedding(file_path)
            X.append(embedding)
            Y.append(label)
        except Exception as e:
            print(f"Skipping {file_name}: {e}")

X = np.array(X)
Y = np.array(Y)

# Split dataset: 70% train, 15% validation, 15% test
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Save preprocessed data
np.savez("preprocessed_data.npz", X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples: {len(X_test)}")
print("Data saved successfully.")
