import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import os
import time
from tensorflow.keras.models import load_model

# Directory to monitor
RECORDINGS_DIR = "D:\REC"
LAST_PROCESSED = {}  # Dictionary to track last modification times of files

# Load YAMNet model
yamnet_model = hub.load("https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1")

# Load trained classifier model
classifier_model = load_model("yamnet_finetuned.h5")

# Class labels
classes = ["background", "Anomaly"]

# Function to extract YAMNet embeddings
def extract_embedding(file_path):
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)  # Convert to 16kHz mono
    waveform = waveform.astype(np.float32)

    # Get YAMNet embeddings
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    # Average embeddings over time
    return np.mean(embeddings.numpy(), axis=0)

# Function to classify an audio file
def classify_audio(file_path):
    try:
        embedding = extract_embedding(file_path)
        embedding = np.expand_dims(embedding, axis=0)  # Reshape for model input

        # Predict class
        predictions = classifier_model.predict(embedding)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class] * 100

        print(f"Predicted class: {classes[predicted_class]}")
        print(f"Confidence: {confidence:.2f}%")

        # Alert if anomaly detected
        if predicted_class == 1:
            folder_name = os.path.basename(os.path.dirname(file_path))  # Extract folder name
            file_name = os.path.basename(file_path)  # Extract file name
            
            if "erecordings2" in folder_name:
                print("🚨 ALERT: Anomaly detected in esp2")
            elif "erecordings" in folder_name:
                print("🚨 ALERT: Anomaly detected in esp1")
            else:
                print(f"🚨 ALERT: Anomaly detected in {folder_name}/{file_name}!")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Function to monitor directory for updates
def monitor_directory():
    global LAST_PROCESSED
    
    try:
        while True:
            # Get all WAV files in directory and subdirectories
            wav_files = []
            for root, _, files in os.walk(RECORDINGS_DIR):
                for f in files:
                    if f.endswith(".wav"):
                        full_path = os.path.join(root, f)
                        wav_files.append(full_path)
            
            if not wav_files:
                time.sleep(1)
                continue  # No files found

            for file_path in wav_files:
                mod_time = os.path.getmtime(file_path)
                
                # Process only if file is new or modified
                if file_path not in LAST_PROCESSED or mod_time > LAST_PROCESSED[file_path]:
                    print(f"🔍 Processing: {file_path}")
                    classify_audio(file_path)
                    LAST_PROCESSED[file_path] = mod_time  # Update tracking time

            time.sleep(1)  # Check every second
    
    except KeyboardInterrupt:
        print("🛑 Monitoring stopped by user.")

# Start monitoring
if __name__ == "__main__":
    print(f"Monitoring {RECORDINGS_DIR} and its subdirectories for new recordings...")
    monitor_directory()
