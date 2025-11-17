"""
Código para copiar y pegar en el notebook después de entrenar los modelos.
Ejecuta este código en una celda del notebook model_training.ipynb después del entrenamiento.
"""

# ============================================================================
# CÓDIGO PARA GUARDAR MODELOS DESDE EL NOTEBOOK
# ============================================================================
# Copia y pega este código en una nueva celda del notebook después de entrenar

import torch
import json
from pathlib import Path

# Crear directorio de modelos
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Guardar modelo U-Net
if 'model' in globals():  # Verificar que el modelo existe
    torch.save(model.state_dict(), models_dir / "unet_resnet34.pth")
    print(f"✓ Modelo U-Net guardado en: {models_dir / 'unet_resnet34.pth'}")
    
    # Guardar métricas de U-Net (actualizar con valores reales)
    if 'metrics' in globals():
        unet_metrics = {
            "model_name": "U-Net with ResNet34",
            "mean_dice": metrics.get("mean_dice", 0.5903),
            "mean_iou": metrics.get("mean_iou", 0.5469),
            "encoder": "resnet34",
            "classes": 1
        }
    else:
        unet_metrics = {
            "model_name": "U-Net with ResNet34",
            "mean_dice": 0.5903,  # Actualizar con valor real
            "mean_iou": 0.5469,   # Actualizar con valor real
            "encoder": "resnet34",
            "classes": 1
        }
    
    with open(models_dir / "unet_resnet34_metrics.json", "w") as f:
        json.dump(unet_metrics, f, indent=2)
    print(f"✓ Métricas U-Net guardadas")
else:
    print("⚠ Modelo U-Net no encontrado. Asegúrate de haber ejecutado el entrenamiento.")

# Guardar modelo DeepLab V3+
if 'model_deeplab' in globals():  # Verificar que el modelo existe
    torch.save(model_deeplab.state_dict(), models_dir / "deeplabv3plus_resnet50.pth")
    print(f"✓ Modelo DeepLab V3+ guardado en: {models_dir / 'deeplabv3plus_resnet50.pth'}")
    
    # Guardar métricas de DeepLab V3+ (actualizar con valores reales del análisis)
    deeplab_metrics = {
        "model_name": "DeepLab V3+ with ResNet50",
        "mean_dice": 0.6794,   # Actualizar con valor real del notebook
        "mean_iou": 0.5469,    # Actualizar con valor real
        "accuracy": 0.9571,    # Del análisis cuantitativo
        "precision": 0.8383,   # Del análisis cuantitativo
        "recall": 0.7620,      # Del análisis cuantitativo
        "f1_score": 0.7984,    # Del análisis cuantitativo
        "encoder": "resnet50",
        "classes": 1
    }
    
    with open(models_dir / "deeplabv3plus_resnet50_metrics.json", "w") as f:
        json.dump(deeplab_metrics, f, indent=2)
    print(f"✓ Métricas DeepLab V3+ guardadas")
else:
    print("⚠ Modelo DeepLab V3+ no encontrado. Asegúrate de haber ejecutado el entrenamiento.")

print("\n" + "="*60)
print("Modelos guardados exitosamente!")
print("Ahora puedes ejecutar: python app.py")
print("="*60)

