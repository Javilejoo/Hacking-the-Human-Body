"""
Utilidades para cargar modelos y hacer predicciones de segmentación.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import segmentation_models_pytorch as smp
from torchvision.transforms.functional import to_tensor
import json

# Configuración
TARGET_SIZE = (512, 512)  # (width, height)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_type="unet", model_path=None):
    """
    Carga un modelo entrenado.
    
    Args:
        model_type: "unet" o "deeplab"
        model_path: Ruta al archivo .pth del modelo. Si es None, usa la ruta por defecto.
    
    Returns:
        Modelo cargado y configurado en modo evaluación.
    """
    models_dir = Path("models")
    
    if model_type == "unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,  # No cargar pesos de ImageNet, usaremos los entrenados
            in_channels=3,
            classes=1
        )
        if model_path is None:
            model_path = models_dir / "unet_resnet34.pth"
    elif model_type == "deeplab":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            encoder_output_stride=16,
            decoder_atrous_rates=(12, 24, 36),
            decoder_channels=256
        )
        if model_path is None:
            model_path = models_dir / "deeplabv3plus_resnet50.pth"
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}. Use 'unet' o 'deeplab'")
    
    # Cargar pesos
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {model_path}. "
            f"Asegúrate de haber guardado el modelo usando save_models.py"
        )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_input):
    """
    Preprocesa una imagen para la predicción.
    
    Args:
        image_input: Puede ser:
            - Ruta a archivo (str o Path)
            - PIL Image
            - numpy array (H, W, 3) en RGB
    
    Returns:
        Tensor preprocesado (1, 3, H, W) listo para el modelo
    """
    # Cargar imagen
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    else:
        raise ValueError(f"Tipo de entrada no soportado: {type(image_input)}")
    
    # Convertir a numpy y redimensionar
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    img_resized = cv2.resize(img_array, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    # Convertir a tensor y normalizar
    img_tensor = to_tensor(img_resized)  # (3, H, W) en [0, 1]
    img_tensor = (img_tensor - MEAN) / STD
    
    # Añadir dimensión de batch
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)
    
    return img_tensor.to(device), (h, w)  # También devolver tamaño original


def predict_image(model, image_input, threshold=THRESHOLD, return_prob=False):
    """
    Hace una predicción de segmentación en una imagen.
    
    Args:
        model: Modelo cargado
        image_input: Imagen de entrada (ver preprocess_image)
        threshold: Umbral para binarización (default 0.5)
        return_prob: Si True, también devuelve las probabilidades
    
    Returns:
        Máscara binaria como numpy array (H, W) o dict con máscara y probabilidades
    """
    img_tensor, original_size = preprocess_image(image_input)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
    
    # Convertir a numpy y redimensionar al tamaño original si es necesario
    mask = preds[0, 0].cpu().numpy()  # (H, W)
    prob_map = probs[0, 0].cpu().numpy()  # (H, W)
    
    # Redimensionar al tamaño original
    if original_size != TARGET_SIZE[::-1]:  # TARGET_SIZE es (W, H), original_size es (H, W)
        mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        prob_map = cv2.resize(prob_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    
    if return_prob:
        return {"mask": mask, "prob": prob_map}
    return mask


def calculate_metrics(pred_mask, gt_mask, eps=1e-7):
    """
    Calcula métricas de evaluación (Dice, IoU) comparando predicción con ground truth.
    
    Args:
        pred_mask: Máscara predicha (numpy array binario)
        gt_mask: Máscara ground truth (numpy array binario)
        eps: Valor pequeño para evitar división por cero
    
    Returns:
        Dict con métricas: dice, iou
    """
    pred_mask = pred_mask.astype(np.float32)
    gt_mask = gt_mask.astype(np.float32)
    
    # Asegurar que tienen el mismo tamaño
    if pred_mask.shape != gt_mask.shape:
        h, w = gt_mask.shape
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Binarizar
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    # Calcular intersección y unión
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    
    # Dice coefficient
    dice = (2 * intersection + eps) / (pred_mask.sum() + gt_mask.sum() + eps)
    
    # IoU (Intersection over Union)
    iou = (intersection + eps) / (union + eps)
    
    return {
        "dice": float(dice),
        "iou": float(iou),
        "intersection": float(intersection),
        "union": float(union)
    }


def load_model_metrics(model_type="unet"):
    """
    Carga las métricas guardadas de un modelo.
    
    Args:
        model_type: "unet" o "deeplab"
    
    Returns:
        Dict con métricas o None si no existe el archivo
    """
    models_dir = Path("models")
    
    if model_type == "unet":
        metrics_path = models_dir / "unet_resnet34_metrics.json"
    elif model_type == "deeplab":
        metrics_path = models_dir / "deeplabv3plus_resnet50_metrics.json"
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")
    
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, "r") as f:
        return json.load(f)


def get_model_info(model_type="unet"):
    """
    Obtiene información sobre un modelo.
    
    Args:
        model_type: "unet" o "deeplab"
    
    Returns:
        Dict con información del modelo
    """
    if model_type == "unet":
        return {
            "name": "U-Net with ResNet34",
            "encoder": "resnet34",
            "architecture": "U-Net",
            "description": "Red neuronal convolucional U-Net con encoder ResNet34 para segmentación semántica de FTUs"
        }
    elif model_type == "deeplab":
        return {
            "name": "DeepLab V3+ with ResNet50",
            "encoder": "resnet50",
            "architecture": "DeepLab V3+",
            "description": "DeepLab V3+ con encoder ResNet50 y módulo ASPP para segmentación de FTUs con mejor resolución espacial"
        }
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")

