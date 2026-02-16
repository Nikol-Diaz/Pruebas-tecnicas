"""
Entrenamiento del modelo SMALL.
- Clases con 200-2000 ejemplos (excluye las mas grandes y las mas chicas).
- BATCH_SIZE = 8
- NUM_EPOCHS = 5
- Descongelar capas 10-11 de BERT.
"""

import sys
import os
import pickle

# Agregar el directorio padre al path para importar utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from utils import (
    MODEL_NAME, BERTClassifier, TicketDataset,
    load_and_prepare_data, prepare_splits,
    train_model, evaluate_model, save_class_distribution,
)

# === CONFIGURACION ===
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'test.csv')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

MIN_SAMPLES = 200
MAX_SAMPLES = 2000
MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
DROPOUT_RATE = 0.1
UNFREEZE_LAYERS = [10, 11]


def main():
    print('=' * 60)
    print('ENTRENAMIENTO MODELO SMALL')
    print(f'Clases: {MIN_SAMPLES}-{MAX_SAMPLES} ejemplos')
    print('=' * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Cargar datos
    print('\n--- Cargando datos ---')
    df_filter = load_and_prepare_data(CSV_PATH, MIN_SAMPLES, MAX_SAMPLES)
    save_class_distribution(df_filter, OUTPUT_DIR)

    # Preparar splits
    print('\n--- Preparando splits ---')
    (train_texts, train_labels, val_texts, val_labels,
     test_texts, test_labels, label_encoder, num_classes) = prepare_splits(df_filter)

    # Guardar label_encoder para prediccion posterior
    encoder_path = os.path.join(OUTPUT_DIR, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f'Label encoder guardado en: {encoder_path}')

    # Guardar config para prediccion
    config = {
        'model_name': MODEL_NAME,
        'num_classes': num_classes,
        'max_length': MAX_LENGTH,
        'dropout_rate': DROPOUT_RATE,
        'unfreeze_layers': UNFREEZE_LAYERS,
        'min_samples': MIN_SAMPLES,
        'max_samples': MAX_SAMPLES,
        'classes': list(label_encoder.classes_),
    }
    import json
    config_path = os.path.join(OUTPUT_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f'Config guardada en: {config_path}')

    # Tokenizador
    print('\n--- Cargando tokenizador ---')
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Datasets
    print('\n--- Creando datasets ---')
    train_dataset = TicketDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TicketDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = TicketDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo
    print('\n--- Creando modelo ---')
    model = BERTClassifier(
        model_name=MODEL_NAME,
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE,
        unfreeze_layers=UNFREEZE_LAYERS,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parametros totales: {total_params:,}')
    print(f'Parametros entrenables: {trainable_params:,}')
    print(f'Porcentaje entrenables: {trainable_params/total_params*100:.2f}%')

    # Class weights + loss
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # Entrenar
    print('\n--- Iniciando entrenamiento ---')
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, NUM_EPOCHS, OUTPUT_DIR)

    # Evaluar
    print('\n--- Evaluando modelo ---')
    metrics = evaluate_model(model, test_loader, label_encoder, device, OUTPUT_DIR)

    print('\n' + '=' * 60)
    print('ENTRENAMIENTO SMALL COMPLETADO')
    print(f'Accuracy: {metrics["accuracy"]*100:.2f}%')
    print(f'F1 Macro: {metrics["f1_macro"]*100:.2f}%')
    print(f'Archivos guardados en: {OUTPUT_DIR}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
