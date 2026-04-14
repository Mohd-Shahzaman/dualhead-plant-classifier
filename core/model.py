from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2

def build_dual_head_model(num_plants, num_diseases, input_shape=(224, 224, 3)):
    """
    Builds a Multi-Output CNN for Plant and Disease Classification.
    Provides two distinct heads: one for Plant Species and one for Disease Status.
    """
    
    # 1. Base Backbone: Transfer Learning with MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze initial weights for stability
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    
    # 2. Shared Path (Universal leaf features)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # 3. Head A: Plant Species Classification
    x_plant = Dense(128, activation='relu', name='plant_feat')(x)
    plant_output = Dense(num_plants, activation='softmax', name='plant_output')(x_plant)

    # 4. Head B: Disease Category classification
    x_disease = Dense(128, activation='relu', name='disease_feat')(x)
    disease_output = Dense(num_diseases, activation='softmax', name='disease_output')(x_disease)

    # Final Multi-Output Model
    model = Model(inputs=inputs, outputs=[plant_output, disease_output])
    
    return model
    
    return model

if __name__ == "__main__":
    # Test compilation
    # Based on your previous data: 1 plant type, 3 disease types
    m = build_dual_head_model(num_plants=1, num_diseases=3)
    m.summary()
