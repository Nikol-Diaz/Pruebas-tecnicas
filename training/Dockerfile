FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar codigo
COPY utils.py .
COPY predict.py .
COPY small/ small/
COPY big/ big/

# Copiar datos
COPY test.csv .

# Por defecto entrena el modelo small
CMD ["python", "small/train.py"]
