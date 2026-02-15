import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import splitfolders
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# CONFIGURACIÓN GLOBAL Y CONSTANTES
# ==========================================
# Usamos rutas relativas para que funcione en cualquier ordenador
BASE_PATH = os.getcwd()
INPUT_FOLDER = os.path.join(BASE_PATH, 'data_raw') # Carpeta donde pones tus imágenes originales
OUTPUT_FOLDER = os.path.join(BASE_PATH, 'dataset_final')
MODEL_SAVE_PATH = os.path.join(BASE_PATH, 'models', 'best_model.keras')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 1337
EPOCHS = 30
CLASSES = ['Bacterial_dermatosis', 'Fungal_infections', 'Healthy', 'Hypersensitivity_allergic_dermatosis']

# Crear carpetas si no existen
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ==========================================
# DEFINICIÓN DE FUNCIONES (Módulos)
# ==========================================

def prepare_dataset(input_dir, output_dir):
    """
    Divide el dataset original en train/val/test si no se ha hecho previamente.
    """
    if not os.path.exists(output_dir):
        print(f"--- Generando división de datos en {output_dir} ---")
        splitfolders.ratio(input_dir,
                           output=output_dir,
                           seed=RANDOM_SEED,
                           ratio=(.8, .1, .1),
                           group_prefix=None)
    else:
        print("--- Dataset ya dividido encontrado. Saltando splitfolders. ---")

def get_data_generators(data_dir):
    """
    Configura y retorna los generadores de imágenes con Data Augmentation para Train.
    """
    # Configuración de Augmentation solo para Train
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES
    )

    val_gen = test_val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES
    )

    # Importante: shuffle=False para evaluación correcta
    test_gen = test_val_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False 
    )

    return train_gen, val_gen, test_gen

def calculate_class_weights(train_gen):
    """
    Calcula pesos para balancear clases minoritarias durante el entrenamiento.
    """
    train_classes = train_gen.classes
    class_weights_list = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_classes),
        y=train_classes
    )
    class_weights_dict = dict(enumerate(class_weights_list))
    
    print("\n--- Pesos de Clases Calculados ---")
    for i, peso in class_weights_dict.items():
        print(f"Clase {CLASSES[i]}: {peso:.2f}")
    
    return class_weights_dict

def build_model(num_classes):
    """
    Construye el modelo basado en MobileNetV2 con Fine-Tuning.
    """
    base_model = MobileNetV2(weights='imagenet',
                             include_top=False,
                             input_shape=(224, 224, 3))
    
    # Descongelamos las últimas capas para fine-tuning
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_history(history):
    """
    Genera gráficas de precisión y pérdida.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.show()

def evaluate_model(model, test_gen):
    """
    Genera matriz de confusión y reporte de clasificación.
    """
    print("\n--- Evaluando modelo en Test Set ---")
    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadera')
    plt.xlabel('Predicción')
    plt.show()

    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

# ==========================================
# ENTRY POINT 
# ==========================================

if __name__ == "__main__":
    # 1. Preparación de datos
    
    # Si ya tienes la carpeta dataset_final generada, usamos esa:
    train_gen, val_gen, test_gen = get_data_generators(OUTPUT_FOLDER)
    
    # 2. Cálculo de pesos
    weights = calculate_class_weights(train_gen)
    
    # 3. Construcción y compilación
    print("\n--- Construyendo Modelo ---")
    model = build_model(num_classes=len(CLASSES))
    
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # 4. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')
    ]
    
    # 5. Entrenamiento
    print("\n--- Iniciando Entrenamiento ---")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=weights
    )
    
    # 6. Evaluación
    plot_history(history)
    evaluate_model(model, test_gen)