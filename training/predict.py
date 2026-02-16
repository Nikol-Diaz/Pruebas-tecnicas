"""
Script para cargar un modelo entrenado y hacer predicciones.

Uso:
    python predict.py small "El cliente solicita bloqueo de tarjeta"
    python predict.py big "No puedo ingresar a la app movil"
"""

import sys
import os
import json
import pickle

import torch
from transformers import BertTokenizer

# Agregar directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import BERTClassifier


def load_model(model_dir):
    """Carga modelo, tokenizador y label_encoder desde un directorio."""
    # Cargar config
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    # Cargar label encoder
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar modelo
    model = BERTClassifier(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate'],
        unfreeze_layers=config['unfreeze_layers'],
    )
    model = model.to(device)

    # Cargar pesos
    checkpoint_path = os.path.join(model_dir, 'checkpoints', 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Cargar tokenizador
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])

    return model, tokenizer, label_encoder, config, device


def predict(text, model, tokenizer, label_encoder, config, device, top_k=3):
    """Predice la clase de un texto."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config['max_length'],
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probabilities = torch.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    top_classes = label_encoder.inverse_transform(top_indices)

    return list(zip(top_classes, top_probs.tolist()))


def main():
    if len(sys.argv) < 3:
        print('Uso: python predict.py <small|big> "texto a clasificar"')
        sys.exit(1)

    model_type = sys.argv[1]  # "small" o "big"
    text = sys.argv[2]

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_type)

    if not os.path.exists(model_dir):
        print(f'Error: no existe el directorio {model_dir}')
        sys.exit(1)

    print(f'Cargando modelo {model_type}...')
    model, tokenizer, label_encoder, config, device = load_model(model_dir)

    print(f'\nTexto: "{text[:100]}..."')
    print(f'\nTop 3 predicciones:')

    results = predict(text, model, tokenizer, label_encoder, config, device)
    for i, (cls, prob) in enumerate(results):
        bar = '█' * int(prob * 30)
        print(f'  {i+1}. {cls}: {prob:.4f} ({prob*100:.1f}%) {bar}')


if __name__ == '__main__':
    main()
