import streamlit as st
import torch
import os
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from helpers import U_NET
import numpy as np
import io

# Configuración de la página
st.set_page_config(
    page_title="U-NET Segmentación de Coches",
    page_icon="🚗",
    layout="wide",
)

# Título principal
st.title("Segmentación de Imágenes con U-NET y bbdd Carvana")

# Función para listar los modelos disponibles en la carpeta models
def get_available_models(models_dir="models"):
    if not os.path.exists(models_dir):
        return []
    
    models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    return models

# Función para listar las imágenes disponibles en la carpeta test
def get_available_images(images_dir="data/test"):
    if not os.path.exists(images_dir):
        return []
    
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return images

# Función para cargar una imagen y prepararla para el modelo
def load_and_process_image(image_path):
    img = Image.open(image_path)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # Añadir dimensión de batch
    return img, img_tensor

# Función para realizar la predicción
def predict_mask(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        mask = torch.argmax(output, dim=1).float()
    return mask

# Función para mostrar imagen y máscara similar a plot_mini_batch
# Reemplaza la función plot_result con esta versión
def plot_result(img, mask):
    # Asegurarse de que la imagen esté en formato numpy
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy()
    
    # Asegurarse de que la máscara esté en formato numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()
    
    # Normalizar la máscara para visualización
    if mask.max() > 0:
        mask = mask / mask.max()
    
    # Crear una copia de la imagen para no modificar la original
    img_with_mask = img.copy()
    
    # Si la imagen está normalizada entre 0-1, convertir a 0-255
    if img_with_mask.max() <= 1.0:
        img_with_mask = (img_with_mask * 255).astype(np.uint8)
    
    # Crear una máscara de color (en este caso, usamos un tono azul semitransparente)
    mask_color = np.zeros((*mask.shape, 4), dtype=np.uint8)  # RGBA        
    mask_color[mask > 0] = [255, 255, 0, 128]  # Amarillo semi-transparente
    
    # Convertir a imagen PIL para facilitar la combinación
    img_pil = Image.fromarray(img_with_mask.astype('uint8'), 'RGB')
    mask_pil = Image.fromarray(mask_color, 'RGBA')
    
    # Superponer la máscara en la imagen
    img_pil.paste(mask_pil, (0, 0), mask_pil)
    
    return img_pil

# Detectar dispositivo disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.sidebar.info(f"Dispositivo: {device}")

# Crear columnas para los selectores
col1, col2 = st.columns(2)

# 1. Selector de modelo
with col1:
    st.subheader("Seleccione el modelo U-Net a evaluar")
    models = get_available_models()
    
    if not models:
        st.error("⚠️ No se encontraron modelos en la carpeta /models/")
        selected_model = None
    else:
        selected_model = st.selectbox("Modelos disponibles", models)
        st.success(f"✅ Modelo seleccionado: {selected_model}")

# 2. Selector de imagen
with col2:
    st.subheader("Seleccione la imagen a segmentar")
    images = get_available_images()
    
    if not images:
        st.error("⚠️ No se encontraron imágenes en la carpeta /data/test/")
        selected_image = None
    else:
        selected_image = st.selectbox("Imágenes disponibles", images)
        st.success(f"✅ Imagen seleccionada: {selected_image}")

# 3. Procesar y mostrar resultados si se seleccionaron modelo e imagen
if selected_model and selected_image:
    try:
        # Cargar el modelo
        model_path = os.path.join("models", selected_model)
        
        try:
            # Intentar cargar el modelo completo primero
            model = torch.load(model_path, weights_only=False, map_location=device)
            st.sidebar.success("✅ Modelo cargado correctamente")
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar modelo completo: {e}")
            st.sidebar.info("Intentando crear el modelo y cargar solo los parámetros...")
            
            # Si falla, intentar crear el modelo e importar los parámetros
            try:
                model = U_NET(in_channels=3, channels=64, num_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=device))
                st.sidebar.success("✅ Parámetros del modelo cargados correctamente")
            except Exception as e:
                st.error(f"❌ Error al cargar los parámetros del modelo: {e}")
                st.stop()
        
        # Mover el modelo al dispositivo adecuado
        model = model.to(device)
        
        # Cargar y procesar la imagen
        image_path = os.path.join("data/test", selected_image)
        original_img, img_tensor = load_and_process_image(image_path)
        
        # Realizar la predicción
        mask = predict_mask(model, img_tensor, device)
        
        # Mostrar el resultado
        st.header("Resultado de la segmentación")
        
        # Columnas para mostrar la imagen original y la predicción
        col_img, col_result = st.columns(2)
        
        # Mostrar la imagen original
        with col_img:
            st.subheader("Imagen Original")
            st.image(original_img, use_container_width=True)  # Corregido aquí
        
        # Mostrar la imagen con la máscara superpuesta
        with col_result:
            st.subheader("Segmentación Predicha")
            
            # Obtener dimensiones originales
            original_width, original_height = original_img.size
            
            # Redimensionar la máscara al tamaño original para visualización
            mask_resized = mask.clone()
            mask_resized = mask_resized.cpu().numpy().squeeze()
            mask_resized = Image.fromarray(mask_resized).resize((original_width, original_height), Image.NEAREST)
            mask_resized = np.array(mask_resized)
            
            # Usar la imagen original y la máscara redimensionada
            result_img = plot_result(
                np.array(original_img), 
                mask_resized
            )
            # Mostrar directamente la imagen resultante sin usar matplotlib
            st.image(result_img, use_container_width=True)
            
        # Información adicional sobre el modelo
        st.sidebar.subheader("Información del modelo")
        
        # Extraer métricas del nombre del modelo si es posible                
        model_info = selected_model.split('_')
        for part in model_info:
            if 'dice' in part.lower():
                # Extraer el valor entre paréntesis usando expresiones regulares
                import re
                dice_match = re.search(r'\(([^)]+)\)', part)
                dice_value = dice_match.group(1) if dice_match else part
                st.sidebar.info(f"Dice Score: {dice_value}")
                
            elif 'iou' in part.lower():
                # Extraer el valor entre paréntesis
                iou_match = re.search(r'\(([^)]+)\)', part)
                iou_value = iou_match.group(1) if iou_match else part
                st.sidebar.info(f"IoU: {iou_value}")
                
            elif 'epoch' in part.lower():
                # Extraer el valor entre paréntesis
                epoch_match = re.search(r'\(([^)]+)\)', part)
                epoch_value = epoch_match.group(1) if epoch_match else part
                st.sidebar.info(f"Épocas: {epoch_value}")
        
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {str(e)}")

# Información adicional en el sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Acerca de esta aplicación")
st.sidebar.info(
    """
    Esta aplicación permite segmentar imágenes de coches utilizando modelos U-NET entrenados.
    
    - Selecciona un modelo de la lista desplegable
    - Selecciona una imagen de prueba
    - Observa el resultado de la segmentación
    
    Los modelos están ubicados en la carpeta `/models/` y las imágenes de prueba en `/data/test/`.
    """
)