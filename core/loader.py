import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class DualOutputDataGenerator(Sequence):
    """
    Custom Data Generator for Dual-Head CNN.
    Yields (images, [plant_labels, disease_labels])
    """
    def __init__(self, df, plant_lb, disease_lb, batch_size=32, target_size=(224, 224), augment=None):
        self.df = df
        self.plant_lb = plant_lb
        self.disease_lb = disease_lb
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Select data
        batch_df = self.df.iloc[indexes]
        
        X = []
        y_plant = []
        y_disease = []

        for _, row in batch_df.iterrows():
            # Load and preprocess image
            img = load_img(row['image_path'], target_size=self.target_size)
            img = img_to_array(img) / 255.0  # Normalize
            
            X.append(img)
            
            # Encode labels using the pre-fitted LabelBinarizers
            plant_enc = self.plant_lb.transform([row['plant_label']])[0]
            disease_enc = self.disease_lb.transform([row['disease_label']])[0]
            
            y_plant.append(plant_enc)
            y_disease.append(disease_enc)

        return np.array(X), {
            "plant_output": np.array(y_plant),
            "disease_output": np.array(y_disease)
        }

if __name__ == "__main__":
    import pickle
    # Test loading
    df = pd.read_csv('data/training_manifest.csv')
    with open('models/encoders/plant_lb.pkl', 'rb') as f:
        p_lb = pickle.load(f)
    with open('models/encoders/disease_lb.pkl', 'rb') as f:
        d_lb = pickle.load(f)
        
    gen = DualOutputDataGenerator(df, p_lb, d_lb, batch_size=2)
    X, y = gen[0]
    print(f"Batch X Shape: {X.shape}")
    print(f"Plant Output Shape: {y['plant_output'].shape}")
    print(f"Disease Output Shape: {y['disease_output'].shape}")
