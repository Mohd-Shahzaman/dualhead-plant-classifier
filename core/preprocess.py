import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle

def prepare_dual_labels(base_path):
    """
    Parses the directory structure for a Dual-Head CNN.
    Expected folder format: 'PlantName___DiseaseName'
    Example: 'Apple___Black_rot'
    """
    data = []
    
    if not os.path.exists(base_path):
        print(f"[ERROR] Source directory not found: {base_path}")
        return None

    # Iterate through folder names
    all_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    for folder in all_folders:
        folder_path = os.path.join(base_path, folder)
        
        # Split logic: Plant species vs Disease name
        if "___" in folder:
            plant, disease = folder.split("___")
        else:
            # Fallback/Default for legacy structures
            plant, disease = "Unknown_Plant", folder 
            
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.abspath(os.path.join(folder_path, img_file))
                data.append({
                    "image_path": img_path,
                    "plant_label": plant.lower(),
                    "disease_label": disease.lower()
                })
                
    return pd.DataFrame(data)

def setup_pipeline(dataset_root='Dataset/Train'):
    print(f"--- Step 1: Initializing Data Pipeline ---")
    
    df = prepare_dual_labels(dataset_root)
    if df is None or df.empty:
        print("[!] No images found. Please ensure Dataset/Train exists and contains subfolders.")
        return
    
    print(f"[+] Total Images Found: {len(df)}")
    print(f"[+] Plant Species Detected: {df['plant_label'].unique()}")
    print(f"[+] Disease Categories Detected: {df['disease_label'].unique()}")

    # Initialize and fit Label Binarizers
    plant_lb = LabelBinarizer()
    disease_lb = LabelBinarizer()

    # Create encoded labels
    plant_encoded = plant_lb.fit_transform(df['plant_label'])
    disease_encoded = disease_lb.fit_transform(df['disease_label'])

    # Save encoders for later use in app.py
    os.makedirs('models/encoders', exist_ok=True)
    with open('models/encoders/plant_lb.pkl', 'wb') as f:
        pickle.dump(plant_lb, f)
    with open('models/encoders/disease_lb.pkl', 'wb') as f:
        pickle.dump(disease_lb, f)
    
    print("[+] Encoders saved to models/encoders/")
    
    # Save the dataframe for training step
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/training_manifest.csv', index=False)
    print("[+] Training manifest saved to data/training_manifest.csv")
    print("--- Step 1 Completed Successfully ---")

if __name__ == "__main__":
    # Adjust this path to wherever your main Dataset folder is
    setup_pipeline()
