🐾 PetHealthCare — Clasificador de enfermedades dermatológicas caninas
Desarrollo de un sistema de visión artificial para asistir en el triaje veterinario, clasificando 4 patologías dermatológicas comunes en perros. El proyecto explora la eficacia de MobileNetV2 con Fine-Tuning en un entorno de datos escasos (Small Data).

DATASET KAGGLE: https://www.kaggle.com/datasets/yashmotiani/dogs-skin-disease-dataset

🛠️ Tech Stack: Python, TensorFlow/Keras, Pandas, Seaborn, Scikit-Learn.

🧪 Metodología:

ETL & Cleaning: Pipeline automatizado para limpiar datos corruptos y evitar Data Leakage.

Baseline: CNN personalizada (4 capas) construida desde cero.

Transfer Learning: Implementación de MobileNetV2 (ImageNet weights).

Optimization: Estrategias de Fine-Tuning, Class Weights para desbalanceo y Early Stopping.

📊 Resultados y Análisis de Error:

Accuracy Final: ~50% (Test Set).

Hallazgos Clave: El modelo demuestra una alta sensibilidad para Alergias y Casos Sanos, pero la Matriz de Confusión revela una dificultad sistémica para distinguir visualmente entre infecciones Fúngicas y Bacterianas en baja resolución.

--- Informe Detallado por Enfermedad ---
                                      precision    recall  f1-score   support

                Bacterial_dermatosis       0.18      0.20      0.19        10
                   Fungal_infections       0.27      0.20      0.23        15
                             Healthy       0.12      0.08      0.10        13
Hypersensitivity_allergic_dermatosis       0.18      0.33      0.23         9

                            accuracy                           0.19        47
                           macro avg       0.19      0.20      0.19        47
                        weighted avg       0.19      0.19      0.18        47

                        

🚀 Próximos Pasos:
Implementación de preprocesado CLAHE para resaltar texturas y migración a arquitectura EfficientNetB0.
