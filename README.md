# Hacking the Human Body - Dashboard de Segmentación de FTUs

Aplicación web interactiva para visualizar y comparar modelos de segmentación de Functional Tissue Units (FTUs) en imágenes histológicas.

## 📋 Descripción

Este proyecto implementa dos modelos de deep learning para la segmentación de FTUs:
- **U-Net con ResNet34**: Red neuronal convolucional especializada en segmentación semántica
- **DeepLab V3+ con ResNet50**: Arquitectura avanzada con módulo ASPP para mejor resolución espacial

## 🚀 Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Entrenar y guardar los modelos:
   - Ejecuta el notebook `model_training.ipynb` para entrenar los modelos
   - Después del entrenamiento, ejecuta el código en `save_models_from_notebook.py` en una celda del notebook para guardar los modelos
   - Los modelos se guardarán en el directorio `models/`

## 🎯 Uso

### Iniciar la aplicación

```bash
python app.py
```

La aplicación se abrirá en `http://localhost:7860` (o la URL que se muestre en la terminal).

### Funcionalidades

La aplicación tiene 4 pestañas principales:

#### 1. 🔮 Predicción de Imágenes
- Sube una imagen para obtener predicciones de segmentación
- Selecciona el modelo a usar (U-Net, DeepLab V3+, o ambos)
- Ajusta el threshold para binarización
- Visualiza la predicción, overlay y métricas del modelo

#### 2. ⚖️ Comparación de Modelos
- Compara ambos modelos en imágenes del dataset de validación
- Visualiza predicciones lado a lado con ground truth
- Muestra métricas comparativas (Dice, IoU)

#### 3. 📊 Análisis de Rendimiento
- Visualiza métricas globales de ambos modelos
- Tabla comparativa de rendimiento
- Gráficos comparativos

#### 4. ℹ️ Información del Proyecto
- Descripción del proyecto
- Información sobre los modelos
- Métricas de entrenamiento

## 📁 Estructura del Proyecto

```
Hacking-the-Human-Body/
├── app.py                          # Aplicación principal Gradio
├── model_utils.py                  # Utilidades para cargar modelos y predicciones
├── visualization_utils.py          # Funciones de visualización
├── save_models.py                  # Script de ayuda para guardar modelos
├── save_models_from_notebook.py    # Código para copiar en el notebook
├── model_training.ipynb            # Notebook de entrenamiento
├── analysis.ipynb                  # Notebook de análisis
├── models/                         # Directorio para modelos guardados
│   ├── unet_resnet34.pth
│   ├── unet_resnet34_metrics.json
│   ├── deeplabv3plus_resnet50.pth
│   └── deeplabv3plus_resnet50_metrics.json
├── data/                           # Datos del proyecto
│   ├── train.csv
│   ├── test.csv
│   └── processed/                  # Imágenes procesadas
└── requirements.txt               # Dependencias
```

## 🔧 Guardar Modelos desde el Notebook

Después de entrenar los modelos en `model_training.ipynb`, copia y pega el código de `save_models_from_notebook.py` en una nueva celda del notebook y ejecútalo. Esto guardará:

- Los pesos de los modelos (`.pth`)
- Las métricas de evaluación (`.json`)

## 📊 Métricas

Los modelos se evalúan usando:
- **Dice Score**: Mide la superposición entre predicción y ground truth
- **IoU (Intersection over Union)**: Mide la intersección sobre la unión
- **Accuracy, Precision, Recall, F1-Score**: Métricas adicionales para DeepLab V3+

## 🛠️ Requisitos Técnicos

- Python 3.8+
- PyTorch (con soporte CUDA recomendado)
- Gradio >= 4.0.0
- Ver `requirements.txt` para la lista completa

## 📝 Notas

- Los modelos se cargan de forma lazy (solo cuando se necesitan)
- El preprocesamiento es consistente con el entrenamiento (normalización ImageNet, resize a 512x512)
- La aplicación soporta imágenes en formato TIFF, PNG, JPEG

## 👥 Autores

Proyecto de Data Science - Hacking the Human Body
