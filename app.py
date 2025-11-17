"""
Aplicación web interactiva con Gradio para visualización y comparación de modelos de segmentación.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
import pandas as pd
import cv2

from model_utils import (
    load_model, predict_image, calculate_metrics, 
    load_model_metrics, get_model_info, preprocess_image
)
from visualization_utils import (
    create_overlay_visualization, create_comparison_plot,
    create_metrics_comparison, create_triple_view
)

# Configuración global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_dir = Path("models")

# Variables globales para modelos (carga lazy)
_model_unet = None
_model_deeplab = None
_val_dataset = None


def get_unet_model():
    """Carga el modelo U-Net de forma lazy."""
    global _model_unet
    if _model_unet is None:
        try:
            _model_unet = load_model("unet")
            print("✓ Modelo U-Net cargado")
        except Exception as e:
            print(f"Error cargando U-Net: {e}")
            return None
    return _model_unet


def get_deeplab_model():
    """Carga el modelo DeepLab V3+ de forma lazy."""
    global _model_deeplab
    if _model_deeplab is None:
        try:
            _model_deeplab = load_model("deeplab")
            print("✓ Modelo DeepLab V3+ cargado")
        except Exception as e:
            print(f"Error cargando DeepLab V3+: {e}")
            return None
    return _model_deeplab


# ============================================================================
# PESTAÑA 1: PREDICCIÓN DE IMÁGENES
# ============================================================================

def predict_single_image(image, model_choice, threshold):
    """
    Hace predicción en una imagen subida.
    
    Args:
        image: Imagen subida (PIL Image o numpy array)
        model_choice: "U-Net", "DeepLab V3+", o "Ambos"
        threshold: Umbral para binarización
    
    Returns:
        Tupla con visualizaciones y métricas
    """
    if image is None:
        return None, None, "Por favor, sube una imagen."
    
    try:
        results = []
        metrics_text = ""
        pred_mask_unet = None
        pred_mask_deeplab = None
        
        if model_choice in ["U-Net", "Ambos"]:
            model = get_unet_model()
            if model is None:
                return None, None, "Error: No se pudo cargar el modelo U-Net. Asegúrate de haber guardado el modelo."
            
            pred_result = predict_image(model, image, threshold=threshold, return_prob=True)
            pred_mask_unet = pred_result["mask"]
            
            # Crear visualización
            overlay = create_overlay_visualization(image, pred_mask_unet, alpha=0.5)
            triple_view = create_triple_view(image, pred_mask_unet)
            
            results.append(("U-Net", overlay, triple_view))
            
            # Métricas del modelo
            unet_metrics = load_model_metrics("unet")
            if unet_metrics:
                metrics_text += f"**U-Net (ResNet34):**\n"
                metrics_text += f"- Dice Score: {unet_metrics.get('mean_dice', 'N/A'):.4f}\n"
                metrics_text += f"- IoU: {unet_metrics.get('mean_iou', 'N/A'):.4f}\n\n"
        
        if model_choice in ["DeepLab V3+", "Ambos"]:
            model = get_deeplab_model()
            if model is None:
                return None, None, "Error: No se pudo cargar el modelo DeepLab V3+. Asegúrate de haber guardado el modelo."
            
            pred_result = predict_image(model, image, threshold=threshold, return_prob=True)
            pred_mask_deeplab = pred_result["mask"]
            
            # Crear visualización
            overlay = create_overlay_visualization(image, pred_mask_deeplab, alpha=0.5, colormap='Blues')
            triple_view = create_triple_view(image, pred_mask_deeplab)
            
            results.append(("DeepLab V3+", overlay, triple_view))
            
            # Métricas del modelo
            deeplab_metrics = load_model_metrics("deeplab")
            if deeplab_metrics:
                metrics_text += f"**DeepLab V3+ (ResNet50):**\n"
                metrics_text += f"- Dice Score: {deeplab_metrics.get('mean_dice', 'N/A'):.4f}\n"
                metrics_text += f"- IoU: {deeplab_metrics.get('mean_iou', 'N/A'):.4f}\n"
                if 'accuracy' in deeplab_metrics:
                    metrics_text += f"- Accuracy: {deeplab_metrics['accuracy']:.4f}\n"
                    metrics_text += f"- Precision: {deeplab_metrics['precision']:.4f}\n"
                    metrics_text += f"- Recall: {deeplab_metrics['recall']:.4f}\n"
                    metrics_text += f"- F1-Score: {deeplab_metrics['f1_score']:.4f}\n"
        
        # Si ambos modelos, crear comparación
        if model_choice == "Ambos" and pred_mask_unet is not None and pred_mask_deeplab is not None:
            comparison = create_comparison_plot(
                image, 
                pred_mask_unet,
                pred_mask_deeplab,
                label1="U-Net",
                label2="DeepLab V3+"
            )
            return comparison, results[0][1], metrics_text
        elif len(results) == 1:
            return results[0][2], results[0][1], metrics_text
        else:
            return None, None, "Error procesando predicciones."
            
    except Exception as e:
        return None, None, f"Error: {str(e)}"


# ============================================================================
# PESTAÑA 2: COMPARACIÓN DE MODELOS
# ============================================================================

def compare_models_on_validation(image_idx):
    """
    Compara ambos modelos en una imagen del dataset de validación.
    """
    try:
        # Intentar cargar dataset de validación desde archivos procesados
        processed_img_dir = Path("data/processed/images")
        processed_mask_dir = Path("data/processed/masks")
        
        if not processed_img_dir.exists() or not processed_mask_dir.exists():
            return None, None, "Dataset de validación no disponible. Ejecuta primero el notebook de entrenamiento para procesar las imágenes."
        
        # Obtener lista de imágenes
        image_files = sorted(list(processed_img_dir.glob("*.png")))
        if len(image_files) == 0:
            return None, None, "No se encontraron imágenes procesadas."
        
        if image_idx >= len(image_files):
            image_idx = len(image_files) - 1
        
        # Cargar imagen y máscara
        img_path = image_files[image_idx]
        mask_path = processed_mask_dir / f"{img_path.stem}_mask.png"
        
        if not mask_path.exists():
            return None, None, f"Máscara no encontrada para {img_path.name}"
        
        # Cargar imágenes
        img_pil = Image.open(img_path).convert("RGB")
        gt_mask = Image.open(mask_path).convert("L")
        gt_mask_np = np.array(gt_mask) / 255.0  # Normalizar a [0, 1]
        
        # Predicciones
        model_unet = get_unet_model()
        model_deeplab = get_deeplab_model()
        
        if model_unet is None or model_deeplab is None:
            return None, None, "Error: No se pudieron cargar los modelos."
        
        # Hacer predicciones
        pred_result_unet = predict_image(model_unet, img_pil, threshold=0.5, return_prob=True)
        pred_result_deeplab = predict_image(model_deeplab, img_pil, threshold=0.5, return_prob=True)
        
        pred_unet = pred_result_unet["mask"]
        pred_deeplab = pred_result_deeplab["mask"]
        
        # Asegurar mismo tamaño para métricas
        if pred_unet.shape != gt_mask_np.shape:
            h, w = gt_mask_np.shape
            pred_unet = cv2.resize(pred_unet, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_deeplab = cv2.resize(pred_deeplab, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Crear comparación
        comparison = create_comparison_plot(
            img_pil, pred_unet, pred_deeplab,
            label1="U-Net", label2="DeepLab V3+",
            gt_mask=gt_mask_np, show_gt=True
        )
        
        # Calcular métricas
        metrics_unet = calculate_metrics(pred_unet, gt_mask_np)
        metrics_deeplab = calculate_metrics(pred_deeplab, gt_mask_np)
        
        # Crear tabla comparativa
        metrics_dict = {
            "U-Net": metrics_unet,
            "DeepLab V3+": metrics_deeplab
        }
        metrics_plot = create_metrics_comparison(metrics_dict, "Comparación de Métricas")
        
        # Texto de métricas
        metrics_text = f"""
**Métricas para Imagen {image_idx + 1} ({img_path.name}):**

**U-Net:**
- Dice Score: {metrics_unet['dice']:.4f}
- IoU: {metrics_unet['iou']:.4f}

**DeepLab V3+:**
- Dice Score: {metrics_deeplab['dice']:.4f}
- IoU: {metrics_deeplab['iou']:.4f}
        """
        
        return comparison, metrics_plot, metrics_text
        
    except Exception as e:
        import traceback
        return None, None, f"Error: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# PESTAÑA 3: ANÁLISIS DE RENDIMIENTO
# ============================================================================

def load_performance_analysis():
    """
    Carga y muestra análisis de rendimiento global.
    """
    try:
        unet_metrics = load_model_metrics("unet")
        deeplab_metrics = load_model_metrics("deeplab")
        
        if unet_metrics is None and deeplab_metrics is None:
            return "No hay métricas guardadas. Ejecuta el notebook de entrenamiento y guarda los modelos."
        
        # Crear tabla comparativa
        metrics_data = []
        if unet_metrics:
            metrics_data.append({
                "Modelo": "U-Net (ResNet34)",
                "Dice Score": unet_metrics.get("mean_dice", "N/A"),
                "IoU": unet_metrics.get("mean_iou", "N/A"),
                "Accuracy": "N/A",
                "Precision": "N/A",
                "Recall": "N/A",
                "F1-Score": "N/A"
            })
        
        if deeplab_metrics:
            metrics_data.append({
                "Modelo": "DeepLab V3+ (ResNet50)",
                "Dice Score": deeplab_metrics.get("mean_dice", "N/A"),
                "IoU": deeplab_metrics.get("mean_iou", "N/A"),
                "Accuracy": deeplab_metrics.get("accuracy", "N/A"),
                "Precision": deeplab_metrics.get("precision", "N/A"),
                "Recall": deeplab_metrics.get("recall", "N/A"),
                "F1-Score": deeplab_metrics.get("f1_score", "N/A")
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Crear gráfico comparativo
        if len(metrics_data) > 0:
            plot_dict = {}
            for row in metrics_data:
                model_name = row["Modelo"]
                plot_dict[model_name] = {
                    "dice": row["Dice Score"] if row["Dice Score"] != "N/A" else 0,
                    "iou": row["IoU"] if row["IoU"] != "N/A" else 0
                }
            
            metrics_plot = create_metrics_comparison(plot_dict, "Comparación Global de Métricas")
        else:
            metrics_plot = None
        
        return df, metrics_plot
        
    except Exception as e:
        return f"Error cargando análisis: {str(e)}", None


# ============================================================================
# PESTAÑA 4: INFORMACIÓN DEL PROYECTO
# ============================================================================

def get_project_info():
    """
    Retorna información sobre el proyecto y los modelos.
    """
    info_text = """
# Hacking the Human Body - Dashboard de Segmentación de FTUs

## Descripción del Proyecto

Este proyecto tiene como objetivo identificar y segmentar **Functional Tissue Units (FTUs)** 
en imágenes histológicas de tejidos humanos utilizando técnicas de deep learning.

## Modelos Implementados

### 1. U-Net con ResNet34
- **Arquitectura**: U-Net con encoder ResNet34
- **Descripción**: Red neuronal convolucional especializada en segmentación semántica
- **Encoder**: ResNet34 pre-entrenado en ImageNet
- **Clases**: Segmentación binaria (FTU vs Background)

### 2. DeepLab V3+ con ResNet50
- **Arquitectura**: DeepLab V3+ con encoder ResNet50
- **Descripción**: Arquitectura avanzada con módulo ASPP (Atrous Spatial Pyramid Pooling)
- **Encoder**: ResNet50 pre-entrenado en ImageNet
- **Características**: Mejor resolución espacial y manejo de objetos a múltiples escalas

## Métricas de Evaluación

- **Dice Score**: Mide la superposición entre predicción y ground truth
- **IoU (Intersection over Union)**: Mide la intersección sobre la unión
- **Accuracy**: Precisión general a nivel de píxel
- **Precision**: Proporción de píxeles predichos como FTU que son realmente FTU
- **Recall**: Proporción de píxeles FTU que fueron correctamente identificados
- **F1-Score**: Media armónica de Precision y Recall

## Uso de la Aplicación

1. **Predicción de Imágenes**: Sube una imagen y obtén predicciones de segmentación
2. **Comparación de Modelos**: Compara ambos modelos en imágenes del dataset de validación
3. **Análisis de Rendimiento**: Visualiza métricas globales y comparativas
4. **Información**: Consulta detalles sobre el proyecto y modelos

## Notas Técnicas

- **Tamaño de entrada**: 512x512 píxeles
- **Normalización**: Media y desviación estándar de ImageNet
- **Threshold**: 0.5 para binarización de máscaras
- **Dispositivo**: GPU (CUDA) si está disponible, CPU en caso contrario
    """
    
    return info_text


# ============================================================================
# INTERFAZ GRADIO
# ============================================================================

def create_interface():
    """Crea la interfaz de Gradio."""
    
    with gr.Blocks(title="Hacking the Human Body - Dashboard", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🧬 Hacking the Human Body - Dashboard de Segmentación de FTUs
            
            Aplicación interactiva para visualizar y comparar modelos de segmentación de Functional Tissue Units (FTUs)
            """
        )
        
        with gr.Tabs():
            # Pestaña 1: Predicción
            with gr.Tab("🔮 Predicción de Imágenes"):
                gr.Markdown("### Sube una imagen para obtener predicciones de segmentación")
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Imagen de Entrada")
                        model_choice = gr.Radio(
                            choices=["U-Net", "DeepLab V3+", "Ambos"],
                            value="Ambos",
                            label="Modelo a Usar"
                        )
                        threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Threshold para Binarización"
                        )
                        predict_btn = gr.Button("Predecir", variant="primary")
                    
                    with gr.Column():
                        prediction_output = gr.Image(label="Visualización de Predicción")
                        overlay_output = gr.Image(label="Overlay sobre Imagen Original")
                        metrics_output = gr.Markdown(label="Métricas del Modelo")
                
                predict_btn.click(
                    fn=predict_single_image,
                    inputs=[image_input, model_choice, threshold],
                    outputs=[prediction_output, overlay_output, metrics_output]
                )
            
            # Pestaña 2: Comparación
            with gr.Tab("⚖️ Comparación de Modelos"):
                gr.Markdown("### Compara ambos modelos en imágenes del dataset de validación")
                
                with gr.Row():
                    with gr.Column():
                        image_idx = gr.Number(
                            value=0,
                            label="Índice de Imagen del Dataset (0-N)",
                            precision=0
                        )
                        compare_btn = gr.Button("Comparar", variant="primary")
                    
                    with gr.Column():
                        comparison_output = gr.Image(label="Comparación Visual")
                        metrics_comparison_plot = gr.Image(label="Gráfico Comparativo de Métricas")
                        metrics_text_output = gr.Markdown(label="Métricas Detalladas")
                
                compare_btn.click(
                    fn=compare_models_on_validation,
                    inputs=[image_idx],
                    outputs=[comparison_output, metrics_comparison_plot, metrics_text_output]
                )
            
            # Pestaña 3: Análisis de Rendimiento
            with gr.Tab("📊 Análisis de Rendimiento"):
                gr.Markdown("### Métricas globales y análisis comparativo")
                
                with gr.Row():
                    with gr.Column():
                        load_analysis_btn = gr.Button("Cargar Análisis", variant="primary")
                    
                    with gr.Column():
                        metrics_table = gr.Dataframe(label="Tabla de Métricas")
                        metrics_plot = gr.Image(label="Gráfico Comparativo")
                
                load_analysis_btn.click(
                    fn=load_performance_analysis,
                    inputs=[],
                    outputs=[metrics_table, metrics_plot]
                )
            
            # Pestaña 4: Información
            with gr.Tab("ℹ️ Información del Proyecto"):
                info_output = gr.Markdown(get_project_info())
    
    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Iniciando aplicación Gradio...")
    print(f"Dispositivo: {device}")
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

