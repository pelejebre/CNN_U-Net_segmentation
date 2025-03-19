""" Índice del Módulo helpers.py

## 1. Componentes de la Arquitectura U-NET
- `Conv_3_k`: Bloque de convolución con kernel 3x3
- `Double_Conv`: Bloque de doble convolución
- `Down_Conv`: Bloque de codificación (pooling + convolución)
- `Up_Conv`: Bloque de decodificación (upsampling + convolución)
- `U_NET`: Arquitectura completa de la red

## 2. Carga y Preprocesamiento de Datos
- `Car_Dataset`: Dataset personalizado para imágenes y máscaras

## 3. Optimización de Hiperparámetros
- `find_lr`: Búsqueda de tasa de aprendizaje óptima

## 4. Métricas de Evaluación
- `dice`: Cálculo del coeficiente Dice (F1-score)
- `iou`: Cálculo de Intersección sobre Unión (Jaccard)
- `accuracy`: Evaluación del modelo con múltiples métricas

## 5. Entrenamiento
- `train`: Entrenamiento completo del modelo con validación


## 6. Visualización
- `plot_mini_batch`: Visualización de lotes de imágenes y máscaras
"""

import os
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# Definir la transformación a tensor
to_tensor = transforms.ToTensor()

# ESTRUCTURA DE LA U-NET ------------------------------
# Conv_3_k: bloque de convolución con un kernel 3x3
class Conv_3_k(nn.Module):  
    """Bloque de convolución con un kernel 3x3"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):        
        return self.conv1(x)

# Double_Conv: bloque de doble convolución que se repite en cada "piso" de la U-Net
class Double_Conv(nn.Module):
    """Bloque de doble convolución para U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
                                Conv_3_k(in_channels, out_channels),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(),
                                Conv_3_k(out_channels, out_channels),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU()                                
                            )
            
    def forward(self, x):
        return self.double_conv(x)

# Down_Conv: bloque que contiene Double_Conv + Pooling    
class Down_Conv(nn.Module):
    """Bloque para hacer Max pooling y convolución"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Empezaremos con un bloque de pooling porque la salida de la convolución será la que linquemos con 
        # el otro "piso" en la etapa de decodificación
        self.encoder = nn.Sequential(
                        nn.MaxPool2d(2,2),
                        Double_Conv(in_channels, out_channels)  
                        )
        
    def forward(self, x):        
        return self.encoder(x)
    
class Up_Conv(nn.Module):
    """Bloque para la etapa de decodificación"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #Upsample va a recibir la salida de la convolución de la etapa de codificación
        # y la salida de la convolución de la etapa de decodificación
        # por eso el out_chanels indicamos in_channels/2
        self.upsample_layer = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='bicubic'),  # Upsample scale_factor=2 duplica dimensiones
                            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1),            
                        )
        self.decoder = Double_Conv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: salida de la etapa de decodificación
        x2: salida de la etapa de codificación
        """        
        x1 = self.upsample_layer(x1)
        x = torch.cat((x2,x1), dim=1)       # La dimensión 1 es la de los canales          
        return self.decoder(x)  # Concatenación y convolución
    
class U_NET(nn.Module):
    
    def __init__(self, in_channels, channels, num_classes):
        super().__init__()
        # Encoder
        self.input_conv = Double_Conv(in_channels, channels)  # Primer bloque de convolución #64, 224, 224
        self.down_conv1 = Down_Conv(channels, 2*channels)          # Primer bloque de pooling y convolución #128, 112, 112
        self.down_conv2 = Down_Conv(2*channels, 4*channels)        # Segundo bloque de pooling y convolución #256, 56, 56  
        self.down_conv3 = Down_Conv(4*channels, 8*channels)        # Tercer bloque de pooling y convolución #512, 28, 28
        
        # Bottleneck
        self.middle_conv = Down_Conv(8*channels, 16*channels)       # Cuarto bloque de pooling y convolución #1024, 14, 14

        # Decoder
        self.up_conv1 = Up_Conv(16*channels, 8*channels)            # Primer bloque de decodificación #512, 28, 28
        self.up_conv2 = Up_Conv(8*channels, 4*channels)             # Segundo bloque de decodificación #256, 56, 56
        self.up_conv3 = Up_Conv(4*channels, 2*channels)             # Tercer bloque de decodificación #128, 112, 112
        self.up_conv4 = Up_Conv(2*channels, channels)               # Cuarto bloque de decodificación #64, 224, 224

        # Output
        self.out_conv = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)

    def forward(self, x):        
        x1 = self.input_conv(x)           # Primer bloque de convolución
        x2 = self.down_conv1(x1)           # Primer bloque de pooling y convolución
        x3 = self.down_conv2(x2)           # Segundo bloque de pooling y convolución
        x4 = self.down_conv3(x3)           # Tercer bloque de pooling y convolución
                
        x5 = self.middle_conv(x4)          # Cuarto bloque de pooling y convolución
        
        u1 = self.up_conv1(x5, x4)         
        u2 = self.up_conv2(u1, x3)
        u3 = self.up_conv3(u2, x2)
        u4 = self.up_conv4(u3, x1)
        
        return self.out_conv(u4)
    
# CARGA DE DATOS -----------------------------
class Car_Dataset(Dataset):
    def __init__(self, data, masks=None, imgs_transform=None, mask_transform=None):        
        self.train_data = data      
        self.train_masks = masks
        
        self.imgs_transform = imgs_transform
        self.mask_transform = mask_transform
        
        # Ordenamos las imágenes para asegurar que coincidan con las máscaras
        self.images = sorted(os.listdir(self.train_data))
        
        # También necesitamos ordenar las máscaras si existen
        if self.train_masks is not None:
            self.masks = sorted(os.listdir(self.train_masks))
        
    def __len__(self):
        if self.train_masks is not None:
            # Si hay máscaras, aseguramos que la longitud sea la misma
            assert len(self.images) == len(self.masks), 'ATENCIÓN. Número de imágenes y máscaras distintas'
                
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.train_data, self.images[idx])
        img = Image.open(image_name)
        
        # (1) Si hay que aplicar transformaciones lo pasamos por la transformación
        # (2) Si no hay que aplicar transformaciones, al menos convertimos a tensor de PyTorch
        
        # Imágenes
        if self.imgs_transform is not None:     # (1) 
            img = self.imgs_transform(img)
        else:                                   # (2) 
            img = to_tensor(img)
        
        # Máscaras
        if self.train_masks is not None:        # Si hay máscaras (imágenes de entrenamiento y validación)         
            mask_name = os.path.join(self.train_masks, self.masks[idx])
            mask = Image.open(mask_name)
            if self.mask_transform is not None: # (1)
                mask = self.mask_transform(mask)
            else:                               # (2)     
                mask = to_tensor(mask)
                             
            # Normalización para asegurarnos de que la máscara está entre 0 y 1
            mask_max = mask.max().item()        
            mask = mask / mask_max        
        else:
            return img
        
        return img, mask

# OPTIMIZACIONES DE HIPERPARÁMETROS -----------------------------
# En helpers.py, modifica la función find_lr
def find_lr(device, model, optimizer, start_val=1e-6, end_val=1, beta=0.99, train_loader=None):
    # Si no se proporciona un loader, muestra un error
    if train_loader is None:
        raise ValueError("Debe proporcionar un loader de datos")
    
    n = len(train_loader) - 1
    factor = (end_val / start_val)**(1/n)
    lr = start_val
    optimizer.param_groups[0]['lr'] = lr 
    avg_loss, loss, acc, = 0., 0., 0.
    lowest_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    accuracies = []
    model = model.to(device=device)
    for i, (x, y) in enumerate(train_loader, start=1):
        x = x.to(device = device, dtype = torch.float32)
        y = y.to(device = device, dtype = torch.long).squeeze(1)
        optimizer.zero_grad()
        scores = model(x)
        cost = F.cross_entropy(input=scores, target=y)
        loss = beta*loss + (1-beta)*cost.item()
        
        avg_loss = loss/(1 - beta**i)

        preds = torch.argmax(scores, dim=1)
        acc_ = (preds == y).sum()/torch.numel(scores)

        if i > 1 and avg_loss > 4 * lowest_loss:
            print(f'from here({i, cost.item()})')
            return log_lrs, losses, accuracies
        if avg_loss < lowest_loss or i == 1:
            lowest_loss = avg_loss

        accuracies.append(acc_.item())

        losses.append(avg_loss)
        log_lrs.append(lr)
        
        cost.backward()
        optimizer.step()
        
        print(f'cost:{cost.item():.4f}, lr: {lr:.4f}, acc: {acc_.item():.4f}')
        lr *= factor
        optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses, accuracies
    
# MÉTRICAS DE EVALUACIÓN -----------------------------    
def dice(preds, targets, smooth=1e-6):
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    intersection = (preds * targets).sum()
    dice_score = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    
    return dice_score.item()

def iou(preds, targets, smooth=1e-6):
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou_score = (intersection + smooth) / (union + smooth)
    
    return iou_score.item()

def accuracy(device, model, loader):
    model.eval()
    cost = 0
    correct = 0
    total = 0
    dice_score_acum = 0
    iou_score_acum = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)  # Convertir a float32
            y = y.to(device, dtype=torch.long).squeeze(1)  # Convertir a long y eliminar la dimensión de canal            
            
            scores = model(x)
            cost += (F.cross_entropy(scores, y)).item()
            
            # Accuracy standard (no óptimo para segmentación pero ayuda a dar una idea)
            predictions = torch.argmax(scores, dim=1)
            correct += (predictions == y).sum().item()
            total += torch.numel(predictions)
            
            # Métricas de segmentación
            dice_score_acum += dice(predictions, y)
            iou_score_acum += iou(predictions, y)
    
    accuracy = correct / total
    dice_score = dice_score_acum / len(loader)
    iou_score = iou_score_acum / len(loader)
    
    return cost / len(loader), accuracy, dice_score, iou_score

# ENTRENAMIENTO DEL MODELO-----------------------------
def train(device, model, optimizer, scheduler = None, epochs = 100, store_every = 25, train_loader = None, val_loader = None):   
    # Si no se proporciona un loader, muestra un error
    if train_loader is None:
        raise ValueError("Debe proporcionar un loader de datos")
    if val_loader is None:
        raise ValueError("Debe proporcionar un loader de datos")
    
    # Registrar el tiempo de inicio del entrenamiento
    import time
    start_time = time.time()
        
    model = model.to(device)
    print(f"Modelo en dispositivo: {next(model.parameters()).device}")  # Verificar dispositivo del modelo    
    for epoch in range(epochs):
        train_correct_num = 0
        train_total = 0
        train_cost_acum = 0
        for i, (x, y) in enumerate(train_loader, start=1):
            model.train()
            x = x.to(device, dtype=torch.float32)               # Convertir a float32
            y = y.to(device, dtype=torch.long).squeeze(1)       # Convertir a long y eliminar la dimensión de canal            
            
            scores = model(x)                                   # Forward pass
            cost = F.cross_entropy(input=scores, target=y)      # Calcular la función de pérdida
            optimizer.zero_grad()                               # Reiniciar los gradientes
            cost.backward()                                     # Backward pass
            optimizer.step()                                    # Actualizar los pesos 
            
            if scheduler: 
                scheduler.step()                      
            
            train_predictions = torch.argmax(scores, dim=1)             # Nos quedamos con el valor máximo de las dos salidas (probabilidades) que nos da el modelo en la salida
            train_correct_num += (train_predictions == y).sum().item()  # Acumulamos el número de aciertos
            train_total += torch.numel(train_predictions)               # Acumulamos el número total de píxeles 
            train_cost_acum += cost.item()                              # Acumulamos el coste
                        
            if i%store_every == 0:
                val_cost, val_acc, dice, iou = accuracy(device, model, val_loader)  # Evaluar el modelo en el conjunto de validación cada 'store_ever' ciclos para evitar que se ralentice el entrenamiento
                train_acc = float(train_correct_num)/train_total
                train_cost_every = float(train_cost_acum)/i
                print(f"Epoch {epoch+1}/{epochs}, Batch {i}, "
                      f"Train Cost: {train_cost_every:.4f}, "
                      f"Train Accuracy: {train_acc:.4f}, "
                      f"Validation Cost: {val_cost:.4f}, "
                      f"Validation Accuracy: {val_acc:.4f}, "
                      f"DICE: {dice:.4f}, "
                      f"IoU: {iou:.4f}")                
                
    # Calcular el tiempo total de entrenamiento
    end_time = time.time()
    total_seconds = end_time - start_time
    
    # Formatear el tiempo como horas:minutos:segundos
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    time_train = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    return time_train, epoch+1, dice, iou  # Devuelvo epoch+1 para que sea el número real de épocas     

# REPRESENTACIONES DE IMÁGENES ------------------------------
def plot_mini_batch(imgs, masks, BATCH_SIZE=32):
    plt.figure(figsize=(20,10))
    for i in range(BATCH_SIZE):
        plt.subplot(4, 8, i+1)                          # 4 filas y 8 columnas
        img=imgs[i,...].permute(1,2,0).numpy()          
        mask = masks[i,...].permute(1,2,0).numpy()      
        plt.imshow(img)                                 # Mostramos la imagen
        plt.imshow(mask, alpha=0.5)                     # Mostramos la máscara con transparencia
        plt.axis('Off')
    plt.tight_layout()
    plt.show()