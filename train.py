import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
import time
import h5py

class PlayerDetectorModel:
    def __init__(self):
        self.input_shape = (768, 1024, 3)  # Match CSGO resolution
        self.batch_size = 32
        self.learning_rate = 0.0001
        
    def build_model(self):
        """Create model architecture using EfficientNet as base"""
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Create new model on top
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layers
        detection = Dense(4, name='detection')(x)  # [x, y, width, height]
        confidence = Dense(1, activation='sigmoid', name='confidence')(x)
        
        model = Model(inputs=inputs, outputs=[detection, confidence])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                'detection': 'mse',
                'confidence': 'binary_crossentropy'
            },
            metrics={
                'detection': 'mse',
                'confidence': 'accuracy'
            }
        )
        
        return model

def load_training_data(folder_path):
    """Load training data from folder containing labeled images"""
    images = []
    labels = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load image
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load corresponding label
            label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    bbox = list(map(float, f.read().strip().split()))
                images.append(img)
                labels.append(bbox)
    
    return np.array(images), np.array(labels)

def main():
    # Initialize paths
    model_save_dir = 'models'
    data_dir = 'training_data'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Create and train model
    detector = PlayerDetectorModel()
    model = detector.build_model()
    
    print("Loading training data...")
    X_train, y_train = load_training_data(data_dir)
    
    # Split into train/validation
    split = int(0.8 * len(X_train))
    X_val = X_train[split:]
    y_val = y_train[split:]
    X_train = X_train[:split]
    y_train = y_train[:split]
    
    print("Training model...")
    start_time = time.time()
    
    # Train model
    history = model.fit(
        X_train,
        {
            'detection': y_train,
            'confidence': np.ones(len(y_train))  # All training examples are positive
        },
        validation_data=(
            X_val,
            {
                'detection': y_val,
                'confidence': np.ones(len(y_val))
            }
        ),
        batch_size=detector.batch_size,
        epochs=50,
        verbose=1
    )
    
    print(f"Training took {time.time() - start_time:.2f} seconds")
    
    # Save model
    model_path = os.path.join(model_save_dir, 'player_detector.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
