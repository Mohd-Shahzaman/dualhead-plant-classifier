import os
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import build_dual_head_model
from loader import DualOutputDataGenerator

def train_model():
    # 1. Load data and encoders
    print("[*] Loading Manifest and Encoders...")
    df = pd.read_csv('data/training_manifest.csv')
    
    with open('models/encoders/plant_lb.pkl', 'rb') as f:
        plant_lb = pickle.load(f)
    with open('models/encoders/disease_lb.pkl', 'rb') as f:
        disease_lb = pickle.load(f)
        
    num_plants = len(plant_lb.classes_)
    num_diseases = len(disease_lb.classes_)
    
    print(f"[+] Training on {len(df)} images.")
    print(f"[+] Plant species: {num_plants}, Diseases: {num_diseases}")

    # 2. Build and Compile Model
    print("[*] Building Dual-Head Architecture...")
    model = build_dual_head_model(num_plants, num_diseases)
    
    # LOSS WEIGHTS: We prioritize disease detection accuracy
    losses = {
        "plant_output": "categorical_crossentropy",
        "disease_output": "categorical_crossentropy"
    }
    loss_weights = {"plant_output": 0.5, "disease_output": 1.0}
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=losses,
        loss_weights=loss_weights,
        metrics=["accuracy"]
    )

    # 3. Setup Data Generators
    # In a real scenario, we'd split the CSV here. For now, we use the same for demo.
    train_gen = DualOutputDataGenerator(df, plant_lb, disease_lb, batch_size=8)

    # 4. Start Training
    epochs = 10
    print(f"[*] Starting training for {epochs} epochs...")
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        verbose=1
    )

    # 5. Save the trained masterpiece
    os.makedirs('models/saved', exist_ok=True)
    model.save('models/saved/plant_disease_dual_head.h5')
    print("[SUCCESS] Model saved as models/saved/plant_disease_dual_head.h5")

if __name__ == "__main__":
    train_model()
