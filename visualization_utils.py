"""
Utilidades para generar visualizaciones y gráficos comparativos.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para Gradio
import seaborn as sns
from PIL import Image
import io
import base64

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


def create_overlay_visualization(image, mask, alpha=0.5, colormap='Reds'):
    """
    Crea una visualización superponiendo la máscara sobre la imagen.
    
    Args:
        image: Imagen original (numpy array o PIL Image)
        mask: Máscara binaria (numpy array)
        alpha: Transparencia de la máscara (0-1)
        colormap: Colormap para la máscara
    
    Returns:
        PIL Image con overlay
    """
    # Convertir imagen a numpy si es necesario
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Asegurar que la imagen está en RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Normalizar imagen a [0, 1] si está en [0, 255]
    if img_array.max() > 1.0:
        img_array = img_array.astype(np.float32) / 255.0
    
    # Normalizar máscara
    if mask.max() > 1.0:
        mask = mask.astype(np.float32) / 255.0
    
    # Asegurar mismo tamaño
    if img_array.shape[:2] != mask.shape[:2]:
        from pathlib import Path
        import cv2
        h, w = img_array.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Crear overlay
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_array)
    ax.imshow(mask, cmap=colormap, alpha=alpha)
    ax.axis('off')
    
    # Convertir a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    overlay_img = Image.open(buf)
    plt.close()
    
    return overlay_img


def create_comparison_plot(image, mask1, mask2, label1="Modelo 1", label2="Modelo 2", 
                           gt_mask=None, show_gt=False):
    """
    Crea una visualización comparando dos predicciones lado a lado.
    
    Args:
        image: Imagen original
        mask1: Primera máscara predicha
        mask2: Segunda máscara predicha
        label1: Etiqueta para primera máscara
        label2: Etiqueta para segunda máscara
        gt_mask: Máscara ground truth (opcional)
        show_gt: Si True, muestra también la máscara ground truth
    
    Returns:
        PIL Image con comparación
    """
    # Convertir imagen
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    if img_array.max() > 1.0:
        img_array = img_array.astype(np.float32) / 255.0
    
    # Determinar número de columnas
    n_cols = 4 if show_gt and gt_mask is not None else 3
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    # Imagen original
    axes[0].imshow(img_array)
    axes[0].set_title("Imagen Original", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Máscara 1
    axes[1].imshow(img_array)
    axes[1].imshow(mask1, cmap='Reds', alpha=0.5)
    axes[1].set_title(f"{label1}", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Máscara 2
    axes[2].imshow(img_array)
    axes[2].imshow(mask2, cmap='Blues', alpha=0.5)
    axes[2].set_title(f"{label2}", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Ground truth (si está disponible)
    if show_gt and gt_mask is not None:
        axes[3].imshow(img_array)
        axes[3].imshow(gt_mask, cmap='Greens', alpha=0.5)
        axes[3].set_title("Ground Truth", fontsize=12, fontweight='bold')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    # Convertir a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    comparison_img = Image.open(buf)
    plt.close()
    
    return comparison_img


def create_metrics_comparison(metrics_dict, title="Comparación de Métricas"):
    """
    Crea un gráfico de barras comparando métricas entre modelos.
    
    Args:
        metrics_dict: Dict con estructura {"Modelo 1": {"dice": 0.6, "iou": 0.5}, ...}
        title: Título del gráfico
    
    Returns:
        PIL Image con gráfico
    """
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    
    # Preparar datos
    x = np.arange(len(metrics))
    width = 0.35
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model in enumerate(models):
        values = [metrics_dict[model].get(metric, 0) for metric in metrics]
        offset = (i - n_models/2 + 0.5) * width / n_models
        ax.bar(x + offset, values, width/n_models, label=model, alpha=0.8)
    
    ax.set_xlabel('Métricas', fontsize=11, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Convertir a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_img = Image.open(buf)
    plt.close()
    
    return plot_img


def create_metrics_distribution(dice_scores, iou_scores, model_name="Modelo", bins=30):
    """
    Crea histogramas de distribución de métricas.
    
    Args:
        dice_scores: Lista o array de Dice scores
        iou_scores: Lista o array de IoU scores
        model_name: Nombre del modelo
        bins: Número de bins para histogramas
    
    Returns:
        PIL Image con histogramas
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma Dice
    ax1.hist(dice_scores, bins=bins, alpha=0.7, color='green', edgecolor='black')
    mean_dice = np.mean(dice_scores)
    ax1.axvline(mean_dice, color='red', linestyle='--', linewidth=2, 
               label=f'Media: {mean_dice:.4f}')
    ax1.set_xlabel('Dice Score', fontsize=10)
    ax1.set_ylabel('Frecuencia', fontsize=10)
    ax1.set_title(f'Distribución de Dice Score - {model_name}', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Histograma IoU
    ax2.hist(iou_scores, bins=bins, alpha=0.7, color='orange', edgecolor='black')
    mean_iou = np.mean(iou_scores)
    ax2.axvline(mean_iou, color='red', linestyle='--', linewidth=2, 
               label=f'Media: {mean_iou:.4f}')
    ax2.set_xlabel('IoU Score', fontsize=10)
    ax2.set_ylabel('Frecuencia', fontsize=10)
    ax2.set_title(f'Distribución de IoU - {model_name}', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Convertir a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_img = Image.open(buf)
    plt.close()
    
    return plot_img


def create_confusion_matrix_plot(cm, normalize=True, title="Matriz de Confusión"):
    """
    Crea una visualización de matriz de confusión.
    
    Args:
        cm: Matriz de confusión 2x2 (numpy array)
        normalize: Si True, normaliza la matriz
        title: Título del gráfico
    
    Returns:
        PIL Image con matriz de confusión
    """
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_to_plot = cm_normalized
    else:
        cm_to_plot = cm
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_to_plot, interpolation='nearest', cmap='Blues')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicho', fontsize=11)
    ax.set_ylabel('Real', fontsize=11)
    
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Background', 'FTU'])
    ax.set_yticklabels(['Background', 'FTU'])
    
    # Añadir valores en las celdas
    for i in range(2):
        for j in range(2):
            if normalize:
                text = ax.text(j, i, f'{cm[i, j]:,}\n({cm_to_plot[i, j]:.2%})',
                             ha="center", va="center", 
                             color="white" if cm_to_plot[i, j] > 0.5 else "black",
                             fontsize=10)
            else:
                text = ax.text(j, i, f'{cm[i, j]:,}',
                             ha="center", va="center", 
                             color="white" if cm_to_plot[i, j] > cm_to_plot.max()/2 else "black",
                             fontsize=10)
    
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    
    # Convertir a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_img = Image.open(buf)
    plt.close()
    
    return plot_img


def create_scatter_plot(gt_pixels, pred_pixels, dice_scores=None, title="Píxeles Real vs Predicho"):
    """
    Crea un scatter plot comparando píxeles reales vs predichos.
    
    Args:
        gt_pixels: Array de píxeles ground truth
        pred_pixels: Array de píxeles predichos
        dice_scores: Array opcional de Dice scores para colorear puntos
        title: Título del gráfico
    
    Returns:
        PIL Image con scatter plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if dice_scores is not None:
        scatter = ax.scatter(gt_pixels, pred_pixels, c=dice_scores, 
                           cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter, ax=ax, label='Dice Score')
    else:
        ax.scatter(gt_pixels, pred_pixels, alpha=0.6, s=30)
    
    max_pixels = max(max(gt_pixels), max(pred_pixels))
    ax.plot([0, max_pixels], [0, max_pixels], 'r--', linewidth=2, 
           label='Predicción Perfecta')
    ax.set_xlabel('Píxeles FTU Reales (GT)', fontsize=11)
    ax.set_ylabel('Píxeles FTU Predichos', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Convertir a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_img = Image.open(buf)
    plt.close()
    
    return plot_img


def create_triple_view(image, pred_mask, gt_mask=None, title="Segmentación"):
    """
    Crea una visualización triple: imagen original, máscara predicha, y overlay.
    
    Args:
        image: Imagen original
        pred_mask: Máscara predicha
        gt_mask: Máscara ground truth (opcional)
        title: Título general
    
    Returns:
        PIL Image con triple vista
    """
    # Convertir imagen
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    if img_array.max() > 1.0:
        img_array = img_array.astype(np.float32) / 255.0
    
    n_cols = 3 if gt_mask is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    # Imagen original
    axes[0].imshow(img_array)
    axes[0].set_title("Imagen Original", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Máscara predicha
    axes[1].imshow(pred_mask, cmap='Reds')
    axes[1].set_title("Máscara Predicha", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_array)
    axes[2].imshow(pred_mask, cmap='Reds', alpha=0.5)
    axes[2].set_title("Overlay", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    # Ground truth si está disponible
    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap='Greens')
        axes[3].set_title("Ground Truth", fontsize=11, fontweight='bold')
        axes[3].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Convertir a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plot_img = Image.open(buf)
    plt.close()
    
    return plot_img

