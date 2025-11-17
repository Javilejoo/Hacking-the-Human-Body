"""
Script para guardar los modelos entrenados desde el notebook.
Ejecutar este script después de entrenar los modelos en model_training.ipynb
"""

import torch
import json
from pathlib import Path
import segmentation_models_pytorch as smp

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print(f"Usando dispositivo: {device}")
print("=" * 60)

# ============================================================================
# 1. Guardar modelo ResNet + U-Net
# ============================================================================
print("\n1. Guardando modelo ResNet + U-Net...")

# Inicializar modelo (misma arquitectura que en el notebook)
model_unet = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(device)

# NOTA: Este script asume que los modelos ya fueron entrenados en el notebook.
# Si ejecutas este script desde el notebook, puedes usar directamente:
# torch.save(model.state_dict(), models_dir / "unet_resnet34.pth")
# torch.save(model_deeplab.state_dict(), models_dir / "deeplabv3plus_resnet50.pth")

# Para uso desde notebook, descomenta estas líneas y ejecuta después del entrenamiento:
"""
# Guardar U-Net
torch.save(model.state_dict(), models_dir / "unet_resnet34.pth")
print(f"✓ Modelo U-Net guardado en: {models_dir / 'unet_resnet34.pth'}")

# Guardar métricas de U-Net
unet_metrics = {
    "model_name": "U-Net with ResNet34",
    "mean_dice": 0.5903,  # Actualizar con valores reales del notebook
    "mean_iou": 0.5469,
    "encoder": "resnet34",
    "classes": 1
}
with open(models_dir / "unet_resnet34_metrics.json", "w") as f:
    json.dump(unet_metrics, f, indent=2)
print(f"✓ Métricas U-Net guardadas en: {models_dir / 'unet_resnet34_metrics.json'}")
"""

# ============================================================================
# 2. Guardar modelo DeepLab V3+
# ============================================================================
print("\n2. Guardando modelo DeepLab V3+...")

model_deeplab = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    encoder_output_stride=16,
    decoder_atrous_rates=(12, 24, 36),
    decoder_channels=256
).to(device)

# Para uso desde notebook, descomenta estas líneas:
"""
# Guardar DeepLab V3+
torch.save(model_deeplab.state_dict(), models_dir / "deeplabv3plus_resnet50.pth")
print(f"✓ Modelo DeepLab V3+ guardado en: {models_dir / 'deeplabv3plus_resnet50.pth'}")

# Guardar métricas de DeepLab V3+
deeplab_metrics = {
    "model_name": "DeepLab V3+ with ResNet50",
    "mean_dice": 0.6794,  # Actualizar con valores reales del notebook
    "mean_iou": 0.5469,   # Actualizar con valores reales
    "accuracy": 0.9571,
    "precision": 0.8383,
    "recall": 0.7620,
    "f1_score": 0.7984,
    "encoder": "resnet50",
    "classes": 1
}
with open(models_dir / "deeplabv3plus_resnet50_metrics.json", "w") as f:
    json.dump(deeplab_metrics, f, indent=2)
print(f"✓ Métricas DeepLab V3+ guardadas en: {models_dir / 'deeplabv3plus_resnet50_metrics.json'}")
"""

print("\n" + "=" * 60)
print("INSTRUCCIONES:")
print("=" * 60)
print("1. Ejecuta este script desde el notebook DESPUÉS de entrenar los modelos")
print("2. O copia y pega las líneas comentadas en el notebook después del entrenamiento")
print("3. Asegúrate de actualizar los valores de métricas con los valores reales")
print("=" * 60)

