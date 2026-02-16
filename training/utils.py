"""Utilidades compartidas para entrenamiento de modelos BERT."""

import os
import re
import warnings
import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidores
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'


# === MODELO ===

class BERTClassifier(torch.nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.1, unfreeze_layers=None):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

        # Congelar todo
        for param in self.bert.parameters():
            param.requires_grad = False

        # Descongelar capas especificadas
        if unfreeze_layers is None:
            unfreeze_layers = [10, 11]
        for name, param in self.bert.named_parameters():
            if any(f'encoder.layer.{i}' in name for i in unfreeze_layers) or 'pooler' in name:
                param.requires_grad = True

        hidden_size = self.bert.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        logits = self.classifier(pooled_output)
        return logits


# === DATASET ===

class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.labels = labels
        print(f'Pre-tokenizando {len(texts)} textos...')
        self.encodings = tokenizer(
            [str(t) for t in texts],
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        print('Tokenizacion completa!')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


# === DATA LOADING ===

def group_labels(label):
    if 'sucursal' in str(label).lower():
        return 'sucursales'
    elif 'cac' in str(label).lower():
        return 'CAC'
    else:
        return label


def load_and_prepare_data(csv_path, min_samples, max_samples=None):
    """Carga y prepara el dataset."""
    df = pd.read_csv(csv_path, usecols=['descripcion', 'areas_resolutora'])
    print(f'Filas totales: {len(df)}')

    df = df.dropna(subset=['descripcion', 'areas_resolutora'])
    df = df.reset_index(drop=True)
    print(f'Filas despues de limpiar nulos: {len(df)}')

    df['areas_resolutora'] = df['areas_resolutora'].str.split(', ').str[-1]
    df['labels'] = df['areas_resolutora'].apply(group_labels)

    class_counts = df['labels'].value_counts()
    print(f'Clases unicas: {len(class_counts)}')

    # Filtrar clases
    if max_samples is not None:
        valid_classes = class_counts[(class_counts >= min_samples) & (class_counts <= max_samples)].index
    else:
        valid_classes = class_counts[class_counts >= min_samples].index

    df_filter = df[df['labels'].isin(valid_classes)].reset_index(drop=True)
    print(f'Clases que se mantienen: {len(valid_classes)}')
    print(f'Filas despues de filtrar: {len(df_filter)}')
    print(f'Clases: {list(df_filter["labels"].unique())}')

    return df_filter


def prepare_splits(df_filter):
    """Codifica labels y divide en train/val/test."""
    label_encoder = LabelEncoder()
    df_filter['label_encoded'] = label_encoder.fit_transform(df_filter['labels'])
    num_classes = len(label_encoder.classes_)

    print(f'\nNumero de clases: {num_classes}')
    for i, cls in enumerate(label_encoder.classes_):
        print(f'  {i}: {cls}')

    texts = df_filter['descripcion'].tolist()
    labels = df_filter['label_encoded'].tolist()

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels,
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels,
    )

    print(f'\nTrain: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}')

    return (train_texts, train_labels, val_texts, val_labels,
            test_texts, test_labels, label_encoder, num_classes)


# === CHECKPOINT ===

def save_checkpoint(model, optimizer, scheduler, epoch, step, val_loss, val_acc, path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, device, optimizer=None, scheduler=None, path=None):
    if path is None or not os.path.exists(path):
        print(f'No se encontro checkpoint en: {path}')
        return 0, 0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f'Checkpoint cargado: epoca {checkpoint["epoch"]}, paso {checkpoint["step"]}')
    print(f'Val Loss: {checkpoint["val_loss"]:.4f}, Val Acc: {checkpoint["val_acc"]:.4f}')
    return checkpoint['epoch'], checkpoint['step']


# === TRAINING ===

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, output_dir):
    """Entrena el modelo y guarda checkpoints, metricas e imagenes."""
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'imagenes'), exist_ok=True)

    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    # Intentar reanudar
    resume_path = os.path.join(checkpoint_dir, 'last_checkpoint.pt')
    start_epoch, global_step = load_checkpoint(model, device, optimizer, scheduler, resume_path)

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')

    print(f'\nIniciando entrenamiento desde epoca {start_epoch}...')
    print(f'Epocas: {start_epoch} -> {num_epochs}')
    print(f'Batches por epoca: {len(train_loader)}')
    print(f'Dispositivo: {device}')
    print('=' * 60)

    for epoch in range(start_epoch, num_epochs):
        # --- TRAIN ---
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f'Epoca {epoch+1}/{num_epochs} [Train]')
        for batch_idx, batch in enumerate(train_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })

            global_step += 1
            if global_step % 500 == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, 0, 0,
                              os.path.join(checkpoint_dir, 'last_checkpoint.pt'))

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoca {epoch+1}/{num_epochs} [Val]'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(logits, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f'\nEpoca {epoch+1}/{num_epochs}')
        print(f'  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}')

        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step,
                       avg_val_loss, val_acc,
                       os.path.join(checkpoint_dir, 'last_checkpoint.pt'))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step,
                           avg_val_loss, val_acc,
                           os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f'  >>> Nuevo mejor modelo guardado! (Val Loss: {avg_val_loss:.4f})')

        print('=' * 60)

    print(f'\nEntrenamiento completado! Mejor Val Loss: {best_val_loss:.4f}')

    # Guardar graficas de entrenamiento
    save_training_plots(train_losses, val_losses, val_accuracies, output_dir)

    return train_losses, val_losses, val_accuracies


# === EVALUATION ===

def evaluate_model(model, test_loader, label_encoder, device, output_dir):
    """Evalua el modelo en el test set y guarda metricas e imagenes."""
    os.makedirs(os.path.join(output_dir, 'imagenes'), exist_ok=True)

    # Cargar mejor modelo
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    load_checkpoint(model, device, path=os.path.join(checkpoint_dir, 'best_model.pt'))

    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluando en test set'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, dim=1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calcular metricas
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    num_classes = len(label_encoder.classes_)
    top3_acc = top_k_accuracy_score(all_labels, all_probabilities, k=min(3, num_classes))
    top5_acc = top_k_accuracy_score(all_labels, all_probabilities, k=min(5, num_classes))

    # Imprimir metricas
    metrics_text = []
    metrics_text.append('=' * 50)
    metrics_text.append('METRICAS DE EVALUACION EN TEST SET')
    metrics_text.append('=' * 50)
    metrics_text.append(f'Accuracy:             {accuracy:.4f} ({accuracy*100:.2f}%)')
    metrics_text.append(f'Precision (macro):    {precision:.4f} ({precision*100:.2f}%)')
    metrics_text.append(f'Recall (macro):       {recall:.4f} ({recall*100:.2f}%)')
    metrics_text.append(f'F1-Score (macro):     {f1_macro:.4f} ({f1_macro*100:.2f}%)')
    metrics_text.append(f'F1-Score (weighted):  {f1_weighted:.4f} ({f1_weighted*100:.2f}%)')
    metrics_text.append(f'Top-3 Accuracy:       {top3_acc:.4f} ({top3_acc*100:.2f}%)')
    metrics_text.append(f'Top-5 Accuracy:       {top5_acc:.4f} ({top5_acc*100:.2f}%)')
    metrics_text.append('=' * 50)

    metrics_str = '\n'.join(metrics_text)
    print(metrics_str)

    # Classification report
    unique_labels = np.unique(all_labels)
    class_names = label_encoder.inverse_transform(unique_labels)
    report = classification_report(all_labels, all_predictions,
                                   target_names=class_names, zero_division=0)
    print('\nREPORTE DE CLASIFICACION POR CLASE')
    print('=' * 80)
    print(report)

    # Guardar metricas en txt
    metrics_path = os.path.join(output_dir, 'metricas.txt')
    with open(metrics_path, 'w') as f:
        f.write(metrics_str)
        f.write('\n\nREPORTE DE CLASIFICACION POR CLASE\n')
        f.write('=' * 80 + '\n')
        f.write(report)
    print(f'\nMetricas guardadas en: {metrics_path}')

    # Guardar metricas como JSON
    metrics_json = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'top3_accuracy': float(top3_acc),
        'top5_accuracy': float(top5_acc),
    }
    json_path = os.path.join(output_dir, 'metricas.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)

    # Guardar confusion matrix
    save_confusion_matrix(all_labels, all_predictions, label_encoder, output_dir)

    return metrics_json


# === PLOTS ===

def save_training_plots(train_losses, val_losses, val_accuracies, output_dir):
    """Guarda graficas de entrenamiento como imagenes."""
    img_dir = os.path.join(output_dir, 'imagenes')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train Loss', marker='o')
    axes[0].plot(val_losses, label='Val Loss', marker='s')
    axes[0].set_title('Loss por Epoca')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_accuracies, label='Val Accuracy', marker='s', color='green')
    axes[1].set_title('Accuracy de Validacion por Epoca')
    axes[1].set_xlabel('Epoca')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'curvas_entrenamiento.png'), dpi=150)
    plt.close()
    print(f'Graficas guardadas en: {img_dir}/curvas_entrenamiento.png')


def save_confusion_matrix(all_labels, all_predictions, label_encoder, output_dir):
    """Guarda la matriz de confusion como imagen."""
    img_dir = os.path.join(output_dir, 'imagenes')
    top_n = min(40, len(label_encoder.classes_))

    top_classes_idx = pd.Series(all_labels).value_counts().head(top_n).index.tolist()
    mask = np.isin(all_labels, top_classes_idx)
    filtered_labels = all_labels[mask]
    filtered_preds = all_predictions[mask]

    cm = confusion_matrix(filtered_labels, filtered_preds, labels=top_classes_idx)
    top_class_names = label_encoder.inverse_transform(top_classes_idx)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=top_class_names, yticklabels=top_class_names)
    plt.title(f'Matriz de Confusion (Top {top_n} clases)')
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f'Confusion matrix guardada en: {img_dir}/confusion_matrix.png')


def save_class_distribution(df_filter, output_dir):
    """Guarda grafico de distribucion de clases."""
    img_dir = os.path.join(output_dir, 'imagenes')
    os.makedirs(img_dir, exist_ok=True)

    plt.figure(figsize=(14, 6))
    top_classes = df_filter['labels'].value_counts().head(15)
    top_classes.plot(kind='barh', color='steelblue')
    plt.title('Top 15 clases mas frecuentes')
    plt.xlabel('Cantidad de ejemplos')
    plt.ylabel('Area resolutora')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'distribucion_clases.png'), dpi=150)
    plt.close()
    print(f'Distribucion guardada en: {img_dir}/distribucion_clases.png')
