"""
Healthiligence - Image Model Training Script
Trains CNN models for Malaria and Pneumonia detection from images
"""

import os
import json
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

class ImageModelTrainer:
    def __init__(self):
        self.model_info_path = 'models/model_info.json'
        self.img_height = 128
        self.img_width = 128
        self.load_model_info()
    
    def load_model_info(self):
        """Load model information"""
        if os.path.exists(self.model_info_path):
            with open(self.model_info_path, 'r') as f:
                self.model_info = json.load(f)
        else:
            self.model_info = {
                "malaria": {"accuracy": 0.0, "last_trained": None},
                "pneumonia": {"accuracy": 0.0, "last_trained": None}
            }
    
    def save_model_info(self):
        """Save model information"""
        with open(self.model_info_path, 'w') as f:
            json.dump(self.model_info, f, indent=2)
    
    def create_cnn_model(self):
        """Create CNN model architecture"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_malaria_model(self):
        """Train malaria detection model"""
        print("\n" + "="*50)
        print("Training Malaria Detection Model...")
        print("="*50)
        
        try:
            data_dir = 'datasets/cell_images'
            
            if not os.path.exists(data_dir):
                print(f"\n⚠ Malaria dataset not found at {data_dir}")
                return None
            
            # Data augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )
            
            # Training generator
            train_generator = train_datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=32,
                class_mode='binary',
                subset='training'
            )
            
            # Validation generator
            validation_generator = train_datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=32,
                class_mode='binary',
                subset='validation'
            )
            
            print(f"Training samples: {train_generator.samples}")
            print(f"Validation samples: {validation_generator.samples}")
            
            # Create model
            model = self.create_cnn_model()
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ModelCheckpoint('models/malaria_model.h5', save_best_only=True, monitor='val_accuracy')
            ]
            
            # Train model
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=20,
                callbacks=callbacks,
                verbose=1
            )
            
            # Get best accuracy
            accuracy = max(history.history['val_accuracy'])
            
            # Save model
            model.save('models/malaria_model.h5')
            
            # Update model info
            self.model_info['malaria']['accuracy'] = float(accuracy)
            self.model_info['malaria']['last_trained'] = datetime.now().isoformat()
            
            print(f"\n✓ Model trained successfully!")
            print(f"✓ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"✓ Model saved to: models/malaria_model.h5")
            
            return accuracy
            
        except Exception as e:
            print(f"\n✗ Error training malaria model: {str(e)}")
            return None
    
    def train_pneumonia_model(self):
        """Train pneumonia detection model"""
        print("\n" + "="*50)
        print("Training Pneumonia Detection Model...")
        print("="*50)
        
        try:
            data_dir = 'datasets/chest_xray/train'
            
            if not os.path.exists(data_dir):
                print(f"\n⚠ Pneumonia dataset not found at {data_dir}")
                return None
            
            # Data augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )
            
            # Training generator
            train_generator = train_datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=32,
                class_mode='binary',
                subset='training'
            )
            
            # Validation generator
            validation_generator = train_datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=32,
                class_mode='binary',
                subset='validation'
            )
            
            print(f"Training samples: {train_generator.samples}")
            print(f"Validation samples: {validation_generator.samples}")
            
            # Create model
            model = self.create_cnn_model()
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ModelCheckpoint('models/pneumonia_model.h5', save_best_only=True, monitor='val_accuracy')
            ]
            
            # Train model
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=20,
                callbacks=callbacks,
                verbose=1
            )
            
            # Get best accuracy
            accuracy = max(history.history['val_accuracy'])
            
            # Save model
            model.save('models/pneumonia_model.h5')
            
            # Update model info
            self.model_info['pneumonia']['accuracy'] = float(accuracy)
            self.model_info['pneumonia']['last_trained'] = datetime.now().isoformat()
            
            print(f"\n✓ Model trained successfully!")
            print(f"✓ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"✓ Model saved to: models/pneumonia_model.h5")
            
            return accuracy
            
        except Exception as e:
            print(f"\n✗ Error training pneumonia model: {str(e)}")
            return None

def main():
    """Main training function"""
    trainer = ImageModelTrainer()
    
    print("\n" + "="*60)
    print("HEALTHILIGENCE - IMAGE MODEL TRAINING")
    print("="*60)
    
    results = {}
    
    # Train malaria model
    results['malaria'] = trainer.train_malaria_model()
    
    # Train pneumonia model
    results['pneumonia'] = trainer.train_pneumonia_model()
    
    # Save model info
    trainer.save_model_info()
    
    # Display summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for model_name, accuracy in results.items():
        if accuracy is not None:
            print(f"✓ {model_name.capitalize()}: {accuracy*100:.2f}%")
        else:
            print(f"✗ {model_name.capitalize()}: Failed")
    print("="*60)

if __name__ == "__main__":
    main()
